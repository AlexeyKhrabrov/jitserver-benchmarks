#!/usr/bin/env python3

import argparse
import getpass
import itertools
import os
import os.path

import acmeair
import daytrader
import jitserver
import petclinic
import remote
import shared
import util


# client_threads, activation_factor
thread_configs = (
	(7,  2),
	(15, 4),
	(31, 8),
	(63, 16)
)

# delay_us
latency_configs = (
	(
		(0,    dict()),
		(200,  dict()),
	), (
		(400,  dict()),
		(800,  dict()),
	), (
		(1200, dict()),
		(2000, dict()),
	), (
		(3200, dict()),
		(4400, dict()),
	),
)

jmeter_durations = {
	"acmeair":   4  * 60, # seconds
	"daytrader": 10 * 60, # seconds
	"petclinic": 4  * 60, # seconds
}


run_experiments = (
	jitserver.Experiment.JITServer,
)

result_experiments = (
	jitserver.Experiment.LocalJIT,
	jitserver.Experiment.JITServer,
)


bench_cls = {
	"acmeair":   acmeair.AcmeAir,
	"daytrader": daytrader.DayTrader,
	"petclinic": petclinic.PetClinic,
}

def get_config(benchmark, threads, factor, delay_us, jmeter, n_runs, localjit=False, skip_complete=False):
	result = bench_cls[benchmark]().small_config()
	result.name = "nthreads_{}_{}_{}".format(threads, "full" if jmeter else "start",
	                                         "localjit" if localjit else delay_us)

	result.jitserver_config.server_resource_stats = True
	result.jitserver_config.client_threads = None if localjit else threads
	result.jitserver_config.client_thread_activation_factor = None if localjit else factor
	result.jitserver_config.forceaot = True

	result.db_config.use_internal_addr = True

	result.application_config.use_internal_addr = True
	result.application_config.start_interval = float("+inf") # seconds

	result.jmeter_config.duration = jmeter_durations[benchmark]

	result.n_instances = 1
	result.cache_extra_instance = True
	result.run_jmeter = jmeter
	result.n_runs = n_runs
	result.skip_complete = skip_complete

	return result

def make_cluster(benchmark, threads, factor, delay_us, jmeter, n_runs,
                 hosts, subset, localjit=False, skip_complete=False):
	host0 = hosts[(2 * subset) % len(hosts)]
	host1 = hosts[(2 * subset + 1) % len(hosts)]

	return shared.BenchmarkCluster(
		get_config(benchmark, threads, factor, delay_us, jmeter, n_runs, localjit, skip_complete),
		bench_cls[benchmark](), jitserver_hosts=[host0], db_hosts=[host0],
		application_hosts=[host1], jmeter_hosts=[host0]
	)

def measure_latency(benchmark, cluster, *, passwd=None):
	path = os.path.join(remote.RemoteHost.logs_dir, benchmark, cluster.config.name, "latency.log")
	if cluster.config.skip_complete and os.path.isfile(path):
		return

	latency = cluster.application_hosts[0].get_latency(cluster.jitserver_hosts[0], passwd=passwd)

	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w") as f:
		print("{}".format(latency), file=f)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("benchmark")
	parser.add_argument("hosts_file", nargs="?")
	parser.add_argument("subset", type=int, nargs="?")

	parser.add_argument("-n", "--n-runs", type=int, nargs="?", const=5)
	parser.add_argument("--skip-complete", action="store_true")
	parser.add_argument("-c", "--cleanup", action="store_true")
	parser.add_argument("-j", "--jmeter", action="store_true")
	parser.add_argument("-l", "--localjit", action="store_true")
	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument("-L", "--logs-path")
	parser.add_argument("-r", "--result", type=int, nargs="?", const=-1)
	parser.add_argument("-R", "--results-path")
	parser.add_argument("-f", "--format")
	parser.add_argument("-d", "--details", action="store_true")
	parser.add_argument("-S", "--stdin-passwd", action="store_true")
	parser.add_argument("--single-legend", action="store_true")
	parser.add_argument("--same-limits", action="store_true") # unused

	args = parser.parse_args()
	remote.RemoteHost.logs_dir = args.logs_path or remote.RemoteHost.logs_dir

	bench = bench_cls[args.benchmark]()

	if args.result is not None:
		import results

		results.results_dir = args.results_path or results.results_dir
		results.plot_format = args.format or results.plot_format

		all_latency_configs = list(c for c in itertools.chain.from_iterable(latency_configs))

		if args.result >= 0:
			lc = all_latency_configs[args.result]
			for tc in thread_configs:
				results.LatencyExperimentResult(
					result_experiments, bench,
					get_config(args.benchmark, *tc, *lc[:-1], args.jmeter, args.n_runs), args.details, **lc[-1]
				).save_results()
			return

		if args.details:
			cmd = [__file__, args.benchmark, "-n", str(args.n_runs), "-d"]
			if args.jmeter:
				cmd.append("-j")
			if args.logs_path is not None:
				cmd.extend(("-L", args.logs_path))
			if args.results_path is not None:
				cmd.extend(("-R", args.results_path))
			if args.format is not None:
				cmd.extend(("-f", args.format))

			util.parallelize(lambda i: util.run(cmd + ["-r", str(i)], check=True), range(len(all_latency_configs)))

		all_configs = [[get_config(args.benchmark, *tc, *lc[:-1], args.jmeter, args.n_runs)
		                for lc in all_latency_configs] for tc in thread_configs]

		results.NThreadsAllExperimentsResult(
			result_experiments, bench, all_configs, args.details, [c[-1] for c in all_latency_configs]
		).save_results(
			legends={"jitserver_peak_cpu_p": True, "peak_mem": True, "jitserver_mem": True}
			if args.single_legend else None
		)
		return

	hosts = [bench.new_host(*h) for h in remote.load_hosts(args.hosts_file)]
	#NOTE: assuming same credentials for all hosts
	if args.stdin_passwd:
		passwd = input("Password: ")
		print()
	else:
		passwd = getpass.getpass()

	util.verbose = args.verbose
	util.set_sigint_handler()

	if args.cleanup:
		cluster = make_cluster(args.benchmark, *thread_configs[0][0], *latency_configs[0][0][:-1],
		                       args.jmeter, args.n_runs, hosts, args.subset)
		cluster.check_sudo_passwd(passwd)
		cluster.full_cleanup(passwd=passwd)
		cluster.application_hosts[0].reset_net_delay("eth0", passwd=passwd)
		return

	if args.localjit:
		cluster = make_cluster(args.benchmark, *thread_configs[0][0], *latency_configs[0][0][:-1],
		                       args.jmeter, args.n_runs, hosts, args.subset, True, args.skip_complete)
		cluster.run_all_experiments([jitserver.Experiment.LocalJIT])
		return

	for c in latency_configs[args.subset]:
		for t in thread_configs:
			cluster = make_cluster(args.benchmark, *t, *c[:-1], args.jmeter, args.n_runs,
			                       hosts, args.subset, False, args.skip_complete)
			cluster.check_sudo_passwd(passwd)

			src = "../latency_{}_localjit/localjit".format("full" if args.jmeter else "start")
			dst_dir = os.path.join(remote.RemoteHost.logs_dir, args.benchmark, cluster.config.name)
			os.makedirs(dst_dir, exist_ok=True)
			dst = os.path.join(dst_dir, "localjit")
			if not os.path.islink(dst):
				os.symlink(src, dst)

			h = cluster.application_hosts[0]
			if c[0]:
				h.reset_net_delay("eth0", passwd=passwd)
				h.set_net_delay("eth0", c[0], passwd=passwd)
			try:
				measure_latency(args.benchmark, cluster, passwd=passwd)
				cluster.run_all_experiments(run_experiments)
			finally:
				if c[0]:
					h.reset_net_delay("eth0", check=True, passwd=passwd)


if __name__ == "__main__":
	main()
