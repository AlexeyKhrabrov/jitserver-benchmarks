#!/usr/bin/env python3

import argparse
import getpass

import acmeair
import daytrader
import jitserver
import petclinic
import results
import remote
import shared
import util


# name, start_interval, duration, n_invocations, idle_time
configurations = {
	"acmeair": (
		("short",  10.0, 2  * 60, 25, 1.0, dict()),
		("medium", 10.0, 5  * 60, 10, 1.0, dict()),
		("long",   10.0, 10 * 60, 5,  1.0, dict()),
	),
	"daytrader": (
		("short",  10.0, 2  * 60, 25, 1.0, dict()),
		("medium", 10.0, 5  * 60, 10, 1.0, dict()),
		("long",   10.0, 10 * 60, 5,  1.0, dict()),
	),
	"petclinic": (
		("short",  10.0, 2  * 60, 25, 1.0, dict()),
		("medium", 10.0, 5  * 60, 10, 1.0, dict()),
		("long",   10.0, 10 * 60, 5,  1.0, dict()),
	),
}


experiments = (
	jitserver.Experiment.LocalJIT,
	jitserver.Experiment.JITServer,
	jitserver.Experiment.AOTCache,
)


bench_cls = {
	"acmeair": acmeair.AcmeAir,
	"daytrader": daytrader.DayTrader,
	"petclinic": petclinic.PetClinic,
}

dbs_per_host = {"acmeair": 1, "daytrader": 4, "petclinic": 1}

# jitserver_hosts, db_hosts, application_hosts, jmeter_hosts
hosts_lists = {
	"acmeair":   ([0], [1, 2],    [3, 4, 5, 6], [8, 9, 10]),
	"daytrader": ([0], [1, 2, 7], [3, 4, 5, 6], [8, 9, 10]),
	"petclinic": ([0], [0],       [1, 2, 3, 4], [8, 9, 10]),
}

def get_config(benchmark, name, interval, duration, n_invocations, idle_time, scc, n_runs, skip_complete=False):
	result = bench_cls[benchmark]().small_config()
	result.name = "density_{}_{}".format("scc" if scc else "noscc", name)

	result.jitserver_config.server_threads = 128
	result.jitserver_config.session_purge_time = (duration + 10) * 1000 # milliseconds
	result.jitserver_config.session_purge_interval = 10 * 1000 # milliseconds

	result.application_config.start_interval = interval
	if scc:
		result.application_config.populate_scc = True
		result.application_config.populate_scc_run_jmeter = False

	result.jmeter_config.duration = duration
	result.jmeter_config.stop_timeout = 3 * 60 # seconds
	result.jmeter_config.duration_includes_start = True

	result.n_dbs = len(hosts_lists[benchmark][1]) * dbs_per_host[benchmark]
	result.n_instances = 64
	result.run_jmeter = True
	result.n_runs = n_runs
	result.skip_complete = skip_complete
	result.n_invocations = n_invocations
	result.idle_time = idle_time
	result.invocation_attempts = 2

	return result

def make_cluster(benchmark, hosts, *args):
	config = get_config(benchmark, *args)
	db_hosts = hosts_lists[benchmark][1]
	if config.n_dbs > len(db_hosts):
		#NOTE: assuming db hosts are homogeneous
		config.db_config.docker_config.ncpus = hosts[db_hosts[0]].get_ncpus() // (config.n_dbs // len(db_hosts))

	return shared.BenchmarkCluster(
		config, bench_cls[benchmark](), jitserver_hosts=[hosts[i] for i in hosts_lists[benchmark][0]],
		db_hosts=[hosts[i] for i in db_hosts], application_hosts=[hosts[i] for i in hosts_lists[benchmark][2]],
		jmeter_hosts=[hosts[i] for i in hosts_lists[benchmark][3]]
	)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("benchmark")
	parser.add_argument("hosts_file", nargs="?")

	parser.add_argument("-s", "--scc", action="store_true")
	parser.add_argument("-n", "--n-runs", type=int, default=3)
	parser.add_argument("--skip-complete", action="store_true")
	parser.add_argument("-c", "--cleanup", action="store_true")
	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument("-L", "--logs-path")
	parser.add_argument("-r", "--result", type=int, nargs="?", const=-1)
	parser.add_argument("-R", "--results-path")
	parser.add_argument("-f", "--format")
	parser.add_argument("-d", "--details", action="store_true")
	parser.add_argument("--single-legend", action="store_true")
	parser.add_argument("--same-limits", action="store_true")
	parser.add_argument("-o", "--overlays", action="store_true")

	args = parser.parse_args()
	remote.RemoteHost.logs_dir = args.logs_path or remote.RemoteHost.logs_dir
	results.results_dir = args.results_path or results.results_dir
	results.plot_format = args.format or results.plot_format

	configs = configurations[args.benchmark]
	bench = bench_cls[args.benchmark]()

	if args.result is not None:
		if args.result >= 0:
			c = configs[args.result]
			results.DensityExperimentResult(
				experiments, bench, get_config(args.benchmark, *c[:-1], args.scc, args.n_runs), args.details, **c[-1]
			).save_results()
			return

		cmd = [__file__, args.benchmark, "-n", str(args.n_runs)]
		if args.scc:
			cmd.append("-s")
		if args.logs_path is not None:
			cmd.extend(("-L", args.logs_path))
		if args.results_path is not None:
			cmd.extend(("-R", args.results_path))
		if args.format is not None:
			cmd.extend(("-f", args.format))
		if args.details:
			cmd.append("-d")

		util.parallelize(lambda i: util.run(cmd + ["-r", str(i)], check=True), range(len(configs)))

		result = results.DensityAllExperimentsResult(
			experiments, bench, [get_config(args.benchmark, *c[:-1], args.scc, args.n_runs) for c in configs],
			args.details, [c[-1] for c in configs]
		)

		limits = None
		if args.same_limits:
			other_result = results.DensityAllExperimentsResult(
				experiments, bench,
				[get_config(args.benchmark, *c[:-1], not args.scc, args.n_runs) for c in configs],
				args.details, [c[-1] for c in configs]
			)
			current_limits = result.save_results(dry_run=True)
			other_limits = other_result.save_results(dry_run=True)
			limits = {f: max(current_limits[f], other_limits[f]) for f in current_limits.keys()}

		result.save_results(
			limits=limits, legends={
				"cpu_time_per_req": args.benchmark == "daytrader",
				"total_peak_mem": False,
			} if args.single_legend else None,
			overlays=args.overlays
		)
		return

	hosts = [bench.new_host(*h) for h in remote.load_hosts(args.hosts_file)]

	util.verbose = args.verbose
	util.set_sigint_handler()

	if args.cleanup:
		cluster = make_cluster(args.benchmark, hosts, *configs[0][:-1], args.scc, args.n_runs)
		#NOTE: assuming same credentials for all hosts
		passwd = getpass.getpass()
		cluster.check_sudo_passwd(passwd)
		cluster.full_cleanup(passwd=passwd)
		return

	for c in configs:
		cluster = make_cluster(args.benchmark, hosts, *c[:-1], args.scc, args.n_runs, args.skip_complete)
		cluster.run_all_density_experiments(experiments)


if __name__ == "__main__":
	main()
