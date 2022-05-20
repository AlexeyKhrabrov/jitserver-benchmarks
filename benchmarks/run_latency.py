#!/usr/bin/env python3

import argparse
import copy
import getpass
import itertools
import os
import os.path

import acmeair
import daytrader
import jitserver
import petclinic
import results
import remote
import shared
import util


# local, ib, delay_us
configurations = ((
		(False, True, 0, dict()),
		(False, False, 0, dict()),
		(False, False, 100, dict()),
	), (
		(False, False, 200, dict()),
		(False, False, 400, dict()),
		(False, False, 800, dict()),
	), (
		(False, False, 1200, dict()),
		(False, False, 2000, dict()),
		(False, False, 3200, dict()),
	), (
		(False, False, 4400, dict()),
		(False, False, 6000, dict()),
		(False, False, 8000, dict()),
	),
)

jmeter_durations = {
	"acmeair": 6 * 60,# seconds
	"daytrader": 15 * 60,# seconds
	"petclinic": 3 * 60,# seconds
}


run_experiments = (
	jitserver.Experiment.AOTCache,
)

result_experiments = (
	jitserver.Experiment.LocalJIT,
	jitserver.Experiment.AOTCache,
	jitserver.Experiment.AOTCacheWarm,
)


bench_cls = {
	"acmeair": acmeair.AcmeAir,
	"daytrader": daytrader.DayTrader,
	"petclinic": petclinic.PetClinic
}

def get_config(benchmark, local, ib, delay_us, jmeter, n_runs, localjit=False):
	result = bench_cls[benchmark]().small_config(False)
	result.name = "latency_{}_{}".format(
		"full" if jmeter else "start",
		"localjit" if localjit else "{}_{}_{}".format(
			"tcp", "local" if local else ("ib" if ib else "eth"), delay_us
		)
	)

	result.jitserver_config.server_extra_stats = True
	result.jitserver_config.client_extra_stats = True
	result.jitserver_config.use_internal_addr = ib or local
	result.jitserver_config.client_threads = 15

	result.db_config.use_internal_addr = True

	result.application_config.use_internal_addr = True
	result.application_config.start_interval = float("+inf")# seconds
	result.application_config.sleep_time = 1.0# seconds

	result.jmeter_config.duration = jmeter_durations[benchmark]

	result.n_instances = 1
	result.aotcache_extra_instance = True
	result.run_jmeter = jmeter
	result.n_runs = n_runs

	return result

def make_cluster(benchmark, local, ib, delay_us, jmeter,
                 n_runs, hosts, subset, localjit=False):
	host0 = hosts[(2 * subset) % len(hosts)]
	host1 = hosts[(2 * subset + 1) % len(hosts)]

	if local:
		h = copy.deepcopy(host1)
		h.internal_addr = "localhost"
		jitserver_hosts = [h]
	else:
		jitserver_hosts = [host0]

	return shared.BenchmarkCluster(
		get_config(benchmark, local, ib, delay_us, jmeter, n_runs, localjit),
		bench_cls[benchmark](), jitserver_hosts=jitserver_hosts,
		db_hosts=[host0], application_hosts=[host1], jmeter_hosts=[host0]
	)

def measure_latency(benchmark, local, ib, cluster, *, passwd=None):
	latency = cluster.application_hosts[0].get_latency(
		cluster.jitserver_hosts[0], use_internal_addr=local or ib, passwd=passwd
	)

	path = os.path.join(remote.RemoteHost.logs_dir, benchmark,
	                    cluster.config.name, "latency.log")
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w") as f:
		print("{}".format(latency), file=f)

def measure_tcp_bandwidth(benchmark, local, ib, cluster, *, passwd=None):
	bandwidth = cluster.application_hosts[0].get_tcp_bandwidth(
		cluster.jitserver_hosts[0], use_internal_addr=local or ib
	)

	path = os.path.join(remote.RemoteHost.logs_dir, benchmark,
	                    cluster.config.name, "bandwidth.log")
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w") as f:
		print("{}".format(bandwidth), file=f)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("benchmark")
	parser.add_argument("hosts_file", nargs="?")
	parser.add_argument("subset", type=int, nargs="?")

	parser.add_argument("-n", "--n-runs", type=int, nargs="?", const=5)
	parser.add_argument("-i", "--ib", action="store_true")
	parser.add_argument("-c", "--cleanup", action="store_true")
	parser.add_argument("-j", "--jmeter", action="store_true")
	parser.add_argument("-l", "--localjit", action="store_true")
	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument("-L", "--logs-path")
	parser.add_argument("-r", "--result", type=int, nargs="?", const=-1)
	parser.add_argument("-R", "--results-path")
	parser.add_argument("-f", "--full-init", action="store_true")
	parser.add_argument("-F", "--format")
	parser.add_argument("-d", "--details", action="store_true")
	parser.add_argument("-S", "--stdin-passwd", action="store_true")

	args = parser.parse_args()
	remote.RemoteHost.logs_dir = args.logs_path or remote.RemoteHost.logs_dir
	results.results_dir = args.results_path or results.results_dir
	results.plot_format = args.format or results.plot_format

	bench = bench_cls[args.benchmark]()

	if args.result is not None:
		all_configs = list(
			c for c in itertools.chain.from_iterable(configurations) if args.ib or not c[1]
		)

		if args.result >= 0:
			c = all_configs[args.result]
			results.LatencyExperimentResult(
				result_experiments, bench,
				get_config(args.benchmark, *c[:-1], args.jmeter, args.n_runs),
				full_init=args.full_init, **c[-1]
			).save_results()

		else:
			if args.details:
				cmd = [__file__, args.benchmark]
				if args.jmeter:
					cmd.append("-j")
				if args.logs_path is not None:
					cmd.extend(("-L", args.logs_path))
				if args.results_path is not None:
					cmd.extend(("-R", args.results_path))
				if args.full_init:
					cmd.append("-f")
				if args.format is not None:
					cmd.extend(("-F", args.format))

				util.parallelize(
					lambda i: util.run(cmd + ["-r", str(i)], check=True),
					range(len(all_configs))
				)

			results.LatencyAllExperimentsResult(
				result_experiments, bench,
				[get_config(args.benchmark, *c[:-1], args.jmeter, args.n_runs)
				 for c in all_configs],
				[c[-1] for c in all_configs], full_init=args.full_init
			).save_results(
				legends={"full_warmup_time": args.benchmark == "petclinic"}
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
		c = configurations[args.subset][0]
		cluster = make_cluster(args.benchmark, *c[:-1], args.jmeter,
		                       args.n_runs, hosts, args.subset)
		cluster.check_sudo_passwd(passwd)
		cluster.full_cleanup(passwd=passwd)
		cluster.application_hosts[0].reset_net_delay("eth0", passwd=passwd)
		return

	if args.localjit:
		c = configurations[args.subset][0]
		cluster = make_cluster(args.benchmark, *c[:-1], args.jmeter,
		                       args.n_runs, hosts, args.subset, True)
		cluster.run_all_experiments([jitserver.Experiment.LocalJIT])
		return

	for c in configurations[args.subset]:
		if c[1] and not args.ib:
			continue

		cluster = make_cluster(args.benchmark, *c[:-1], args.jmeter,
		                       args.n_runs, hosts, args.subset)
		cluster.check_sudo_passwd(passwd)

		src = "../latency_{}_localjit/localjit".format("full" if args.jmeter
		                                               else "start")
		dst_dir = os.path.join(remote.RemoteHost.logs_dir, args.benchmark,
		                       cluster.config.name)
		os.makedirs(dst_dir, exist_ok=True)
		dst = os.path.join(dst_dir, "localjit")
		if not os.path.islink(dst):
			os.symlink(src, dst)

		h = cluster.application_hosts[0]
		if c[2]:
			h.reset_net_delay("eth0", passwd=passwd)
			h.set_net_delay("eth0", c[2], passwd=passwd)
		try:
			measure_latency(args.benchmark, c[0], c[1], cluster, passwd=passwd)
			measure_tcp_bandwidth(args.benchmark, c[0], c[1], cluster)
			cluster.run_all_experiments(run_experiments)
		finally:
			if c[2]:
				h.reset_net_delay("eth0", check=True, passwd=passwd)


if __name__ == "__main__":
	main()
