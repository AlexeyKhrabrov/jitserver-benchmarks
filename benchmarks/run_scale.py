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


# n_instances, n_dbs, jitserver_hosts, db_hosts, application_hosts, jmeter_hosts
configurations = {
	"acmeair": (
		(64, 2, [0], [1, 2], [3, 4, 5, 6], [8, 9, 10], dict()),

		(48, 2, [0], [1, 2], [3, 4, 5], [8, 9, 10], dict()),
		(1, 1, [6], [6], [7], [6], dict()),

		(32, 1, [0], [1], [2, 3], [8, 9], dict()),
		(12, 1, [4], [5], [6], [10], dict()),

		(24, 1, [0], [1], [2, 3], [8, 9], dict()),
		(16, 1, [4], [5], [6], [10], dict()),

		(8, 1, [0], [1], [2], [8], dict()),
		(4, 1, [3], [4], [5], [9], dict()),
		(2, 1, [6], [6], [7], [10], dict()),
	),
	"daytrader": (
		(64, 12, [0], [1, 2, 3], [4, 5, 6, 7], [8, 9, 10], dict()),

		(48, 12, [0], [1, 2, 3], [4, 5, 6], [8, 9, 10], dict()),

		(32, 8, [0], [1, 2], [3, 4], [8, 9], dict()),
		(12, 3, [5], [6], [7], [10], dict()),

		(24, 6, [0], [1, 2], [3, 4], [8, 9], dict()),
		(16, 4, [5], [6], [7], [10], dict()),

		(8, 2, [0], [1], [2], [8], dict()),
		(4, 1, [3], [4], [5], [9], dict()),
		(2, 1, [6], [6], [7], [10], dict()),
		(1, 1, [7], [7], [6], [7], dict()),
	),
	"petclinic": (
		(64, 1, [0], [0], [1, 2, 3, 4], [5, 8, 9, 10], dict()),
		(2, 1, [6], [6], [7], [6], dict()),
		(1, 1, [7], [7], [6], [7], dict()),

		(48, 1, [0], [0], [1, 2, 3], [8, 9, 10], dict()),
		(16, 1, [4], [4], [5], [6, 7], dict()),

		(32, 1, [0], [0], [1, 2], [3, 4], dict()),
		(24, 1, [5], [5], [6, 7], [8, 9], dict()),

		(12, 1, [0], [0], [1], [8, 9], dict()),
		(8, 1, [2], [2], [3], [4], dict()),
		(4, 1, [5], [6], [6], [7], dict()),

		(80, 1, [0], [0], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], dict()),
	),
}

jmeter_durations = {
	"acmeair": 6 * 60,# seconds
	"daytrader": 15 * 60,# seconds
	"petclinic": 2 * 60# seconds
}


experiments = (
	jitserver.Experiment.LocalJIT,
	jitserver.Experiment.JITServer,
	jitserver.Experiment.AOTCache,
)


bench_cls = {
	"acmeair": acmeair.AcmeAir,
	"daytrader": daytrader.DayTrader,
	"petclinic": petclinic.PetClinic
}

def get_config(benchmark, n_instances, n_dbs, jmeter,
               n_runs, skip_complete_runs=False):
	result = bench_cls[benchmark]().small_config(False)
	result.name = "scale_{}_{}".format("full" if jmeter else "start", n_instances)

	result.jitserver_config.server_threads = 128
	result.application_config.sleep_time = 1.0# seconds
	result.jmeter_config.duration = jmeter_durations[benchmark]

	result.n_instances = n_instances
	result.n_dbs = n_dbs
	result.run_jmeter = jmeter
	result.n_runs = n_runs
	result.skip_complete_runs = skip_complete_runs

	return result

def make_cluster(benchmark, hosts, n_instances, n_dbs,
                 jitserver_hosts, db_hosts, application_hosts, jmeter_hosts,
                 jmeter, n_runs, skip_complete_runs=False):
	config = get_config(benchmark, n_instances, n_dbs,
	                    jmeter, n_runs, skip_complete_runs)
	if config.n_dbs > len(db_hosts):
		#NOTE: assuming db hosts are homogeneous
		config.db_config.docker_config.ncpus = (
			hosts[db_hosts[0]].get_ncpus() // (config.n_dbs // len(db_hosts))
		)

	return shared.BenchmarkCluster(
		config, bench_cls[benchmark](),
		jitserver_hosts=[hosts[i] for i in jitserver_hosts],
		db_hosts=[hosts[i] for i in db_hosts],
		application_hosts=[hosts[i] for i in application_hosts],
		jmeter_hosts=[hosts[i] for i in jmeter_hosts]
	)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("benchmark")
	parser.add_argument("hosts_file", nargs="?")
	parser.add_argument("config_idx", type=int, nargs="?")

	parser.add_argument("-n", "--n-runs", type=int, nargs="?", const=5)
	parser.add_argument("--skip-complete-runs", action="store_true")
	parser.add_argument("-c", "--cleanup", action="store_true")
	parser.add_argument("-j", "--jmeter", action="store_true")
	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument("-L", "--logs-path")
	parser.add_argument("-r", "--result", type=int, nargs="?", const=-1)
	parser.add_argument("-R", "--results-path")
	parser.add_argument("-f", "--full-init", action="store_true")
	parser.add_argument("-F", "--format")
	parser.add_argument("-d", "--details", action="store_true")
	parser.add_argument("--single-legend", action="store_true")

	args = parser.parse_args()
	remote.RemoteHost.logs_dir = args.logs_path or remote.RemoteHost.logs_dir
	results.results_dir = args.results_path or results.results_dir
	results.plot_format = args.format or results.plot_format

	configs = configurations[args.benchmark]
	bench = bench_cls[args.benchmark]()

	if args.result is not None:
		if args.result >= 0:
			c = configs[args.result]
			results.ScaleExperimentResult(
				experiments, bench_cls[args.benchmark](),
				get_config(args.benchmark, c[0], c[1], args.jmeter, args.n_runs),
				full_init=args.full_init, **(c[-1] or {})
			).save_results()

		else:
			if args.details:
				cmd = [__file__, args.benchmark, "-n", str(args.n_runs)]
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
					range(len(configs))
				)

			sorted_configs = sorted(configs, key=lambda c: c[0])
			results.ScaleAllExperimentsResult(
				experiments, bench,
				[get_config(args.benchmark, c[0], c[1], args.jmeter, args.n_runs)
				 for c in sorted_configs],
				[c[-1] for c in sorted_configs], full_init=args.full_init
			).save_results(
				legends={
					"full_warmup_time_normalized": args.benchmark == "petclinic"
				} if args.single_legend else None
			)

		return

	hosts = [bench.new_host(*h) for h in remote.load_hosts(args.hosts_file)]
	c = configs[args.config_idx]

	util.verbose = args.verbose
	util.set_sigint_handler()

	if args.cleanup:
		cluster = make_cluster(args.benchmark, hosts, *c[:-1],
		                       args.jmeter, args.n_runs)
		#NOTE: assuming same credentials for all hosts
		passwd = getpass.getpass()
		cluster.check_sudo_passwd(passwd)
		cluster.full_cleanup(passwd=passwd)
		return

	cluster = make_cluster(args.benchmark, hosts, *c[:-1], args.jmeter,
	                       args.n_runs, args.skip_complete_runs)
	cluster.run_all_experiments(experiments, skip_cleanup=True)


if __name__ == "__main__":
	main()
