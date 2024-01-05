#!/usr/bin/env python3

import argparse
import getpass

import acmeair
import daytrader
import jitserver
import petclinic
import remote
import results
import shared
import util


# size, warm, duration, scc_extra_duration
configurations = {
	"acmeair": (
		("XS", False, 7 * 60, 0,      dict()),
		("XS", True,  4 * 60, 3 * 60, dict()),
		("S",  False, 6 * 60, 0,      dict()),
		("S",  True,  3 * 60, 3 * 60, dict()),
		("M",  False, 3 * 60, 0,      dict()),
		("M",  True,  150,    30,     dict()),
		("L",  False, 2 * 60, 0,      dict()),
		("L",  True,  90,     30,     dict()),
	),
	"daytrader": (
		("XS", False, 25 * 60, 0,       dict()),
		("XS", True,  15 * 60, 10 * 60, dict()),
		("S",  False, 15 * 60, 0,       dict()),
		("S",  True,  10 * 60, 5 * 60,  dict()),
		("M",  False, 7 * 60,  0,       dict()),
		("M",  True,  4 * 60,  3 * 60,  dict()),
		("L",  False, 5 * 60,  0,       dict()),
		("L",  True,  3 * 60,  2 * 60,  dict()),
	),
	"petclinic": (
		("XS", False, 8 * 60, 0,      dict()),
		("XS", True,  6 * 60, 2 * 60, dict()),
		("S",  False, 5 * 60, 0,      dict()),
		("S",  True,  3 * 60, 2 * 60, dict()),
		("M",  False, 3 * 60, 0,      dict()),
		("M",  True,  150,    30,     dict()),
		("L",  False, 150,    0,      dict()),
		("L",  True,  2 * 60, 30,     dict()),
	),
}


run_experiments = (
	jitserver.Experiment.LocalJIT,
	jitserver.Experiment.AOTCache,
)

result_experiments = (
	jitserver.Experiment.LocalJIT,
	jitserver.Experiment.AOTCache,
	jitserver.Experiment.AOTCacheWarm,
)


def get_config(bench, jmeter, size, warm, duration, scc_extra_duration, n_runs, skip_complete_runs=False):
	if size == "XS":
		c = bench.xsmall_config()
	elif size == "S":
		c = bench.small_config()
	elif size == "M":
		c = bench.medium_config()
	elif size == "L":
		c = bench.large_config()

	if warm:
		config = shared.warm_full_config(c)
	else:
		config = shared.cold_config(c)

	config.name = "single_{}_{}".format("full" if jmeter else "start", config.name)

	config.application_config.start_interval = float("+inf") # seconds
	config.jmeter_config.duration = duration
	config.jmeter_config.scc_extra_duration = scc_extra_duration

	config.n_instances = 1
	config.cache_extra_instance = True
	config.run_jmeter = jmeter
	config.n_runs = n_runs
	config.skip_complete_runs = skip_complete_runs

	return config

bench_cls = {
	"acmeair": acmeair.AcmeAir,
	"daytrader": daytrader.DayTrader,
	"petclinic": petclinic.PetClinic
}

def make_cluster(bench, hosts, subset, jmeter, size, warm, duration,
                 scc_extra_duration, n_runs, skip_complete_runs=False):
	host0 = hosts[subset % len(hosts)]
	host1 = hosts[(subset + (1 if (subset % 2 == 0) else -1)) % len(hosts)]

	return shared.BenchmarkCluster(
		get_config(bench, jmeter, size, warm, duration, scc_extra_duration, n_runs, skip_complete_runs),
		bench, jitserver_hosts=[host0], db_hosts=[host0], application_hosts=[host1], jmeter_hosts=[host0]
	)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("benchmark")
	parser.add_argument("hosts_file", nargs="?")
	parser.add_argument("config_idx", type=int, nargs="?")
	parser.add_argument("subset", type=int, nargs="?")

	parser.add_argument("-n", "--n-runs", type=int, nargs="?", const=5)
	parser.add_argument("--skip-complete-runs", action="store_true")
	parser.add_argument("-c", "--cleanup", action="store_true")
	parser.add_argument("-j", "--jmeter", action="store_true")
	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument("-L", "--logs-path")
	parser.add_argument("-r", "--result", type=int, nargs="?", const=-1)
	parser.add_argument("-R", "--results-path")
	parser.add_argument("-f", "--format")
	parser.add_argument("-d", "--details", action="store_true")
	parser.add_argument("--single-legend", action="store_true")
	parser.add_argument("--same-limits", action="store_true")

	args = parser.parse_args()
	remote.RemoteHost.logs_dir = args.logs_path or remote.RemoteHost.logs_dir
	results.results_dir = args.results_path or results.results_dir
	results.plot_format = args.format or results.plot_format

	configs = configurations[args.benchmark]
	bench = bench_cls[args.benchmark]()

	if args.result is not None:
		if args.result >= 0:
			c = configs[args.result]

			results.SingleInstanceExperimentResult(
				result_experiments, bench, get_config(bench, args.jmeter, *c[:-1], args.n_runs), **c[-1]
			).save_results(details=args.details)

		else:
			cmd = [__file__, args.benchmark, "-n", str(args.n_runs)]
			if args.jmeter:
				cmd.append("-j")
			if args.logs_path is not None:
				cmd.extend(("-L", args.logs_path))
			if args.results_path is not None:
				cmd.extend(("-R", args.results_path))
			if args.format is not None:
				cmd.extend(("-f", args.format))
			if args.details:
				cmd.append("-d")

			util.parallelize(lambda i: util.run(cmd + ["-r", str(i)], check=True), range(len(configs)))

			cold_configs = configs[0::2]
			warm_configs = configs[1::2]

			cold_result = results.SingleInstanceAllExperimentsResult(
				result_experiments, bench, "cold",
				[get_config(bench, args.jmeter, *c[:-1], args.n_runs) for c in cold_configs],
				["XS", "S", "M", "L"], [c[-1] for c in cold_configs],
			)
			warm_result = results.SingleInstanceAllExperimentsResult(
				result_experiments, bench, "warm",
				[get_config(bench, args.jmeter, *c[:-1], args.n_runs) for c in warm_configs],
				["XS", "S", "M", "L"], [c[-1] for c in warm_configs]
			)

			limits = None
			if args.same_limits:
				cold_limits = cold_result.save_results(dry_run=True)
				warm_limits = warm_result.save_results(dry_run=True)
				limits = {f: max(cold_limits[f], warm_limits[f]) for f in cold_limits.keys()}

			cold_result.save_results(
				limits=limits,
				legends={"start_time": False, "warmup_time": False, "peak_mem": False} if args.single_legend else None
			)
			warm_result.save_results(
				limits=limits,
				legends={"start_time": True, "warmup_time": False, "peak_mem": False} if args.single_legend else None
			)

		return

	hosts = [bench.new_host(*h) for h in remote.load_hosts(args.hosts_file)]
	c = configs[args.config_idx]

	util.verbose = args.verbose
	util.set_sigint_handler()

	if args.cleanup:
		cluster = make_cluster(bench, hosts, args.subset, args.jmeter, *c[:-1], args.n_runs)
		#NOTE: assuming same credentials for all hosts
		passwd = getpass.getpass()
		cluster.check_sudo_passwd(passwd)
		cluster.full_cleanup(passwd=passwd)
		return

	cluster = make_cluster(bench, hosts, args.subset, args.jmeter, *c[:-1], args.n_runs, args.skip_complete_runs)
	cluster.run_all_experiments(run_experiments, skip_cleanup=True)


if __name__ == "__main__":
	main()
