#!/usr/bin/env python3

import argparse
import getpass

import jitserver
import remote
import renaissance
import shared
import util


# n_iterations (full warmup)
configurations = {
	"akka-uct":         (8,   dict()),
	"als":              (20,  dict()),
	"chi-square":       (24,  dict()),
	"db-shootout":      (8,   dict()),
	"dec-tree":         (40,  dict()),
	"dotty":            (100, dict()),
	"finagle-chirper":  (40,  dict()),
	"finagle-http":     (12,  dict()),
	"fj-kmeans":        (8,   dict()),
	"future-genetic":   (12,  dict()),
	"gauss-mix":        (20,  dict()),
	"log-regression":   (80,  dict()),
	"mnemonics":        (8,   dict()),
	"movie-lens":       (16,  dict()),
	"naive-bayes":      (12,  dict()),
	"page-rank":        (8,   dict()),
	"par-mnemonics":    (8,   dict()),
	"philosophers":     (40,  dict()),
	"reactors":         (8,   dict()),
	"rx-scrabble":      (50,  dict()),
	"scala-doku":       (8,   dict()),
	"scala-kmeans":     (30,  dict()),
	"scala-stm-bench7": (30,  dict()),
	"scrabble":         (24,  dict()),
}

all_workloads = list(configurations)


run_experiments = (
	jitserver.Experiment.LocalJIT,
	jitserver.Experiment.AOTCache,
	jitserver.Experiment.ProfileCache,
	jitserver.Experiment.AOTPrefetcher,
	jitserver.Experiment.FullCache,
)

result_experiments = (
	jitserver.Experiment.LocalJIT,
	jitserver.Experiment.AOTCache,
	jitserver.Experiment.AOTCacheWarm,
	jitserver.Experiment.ProfileCacheWarm,
	jitserver.Experiment.AOTPrefetcherWarm,
	jitserver.Experiment.FullCacheWarm,
)


def get_config(workload, n_runs, full, skip_complete=False):
	config = renaissance.Renaissance.base_config()
	config.name = "".join(("single_", "full_" if full else "1st_", workload or ""))
	if config.name.endswith("_"):
		config.name = config.name[:-1]

	config.jitserver_config.client_threads = 31
	config.jitserver_config.client_thread_activation_factor = 16
	config.jitserver_config.server_resource_stats = True

	config.application_config.start_interval = float("+inf")
	config.n_instances = 1
	config.cache_extra_instance = True
	config.n_runs = n_runs
	config.skip_complete = skip_complete

	return config

def make_cluster(workload, hosts, subset, n_runs, full, skip_complete=False):
	config = get_config(workload, n_runs, full, skip_complete)

	host0 = hosts[subset % len(hosts)]
	host1 = hosts[(subset + (1 if (subset % 2 == 0) else -1)) % len(hosts)]

	return shared.BenchmarkCluster(
		config, renaissance.Renaissance,
		jitserver_hosts=[host0], db_hosts=[host0], application_hosts=[host1], jmeter_hosts=[host0],
		extra_args=renaissance.Renaissance.extra_args(workload, config.application_config,
		                                              configurations[workload][0] if full else 1),
		fix_log_cmd=renaissance.Renaissance.fix_log_cmd()
	)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("workload", nargs="?")
	parser.add_argument("hosts_file", nargs="?")
	parser.add_argument("subset", type=int, nargs="?")

	parser.add_argument("-n", "--n-runs", type=int, default=5)
	parser.add_argument("--skip-complete", action="store_true")
	parser.add_argument("-c", "--cleanup", action="store_true")
	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument("-L", "--logs-path")
	parser.add_argument("-r", "--result", action="store_true")
	parser.add_argument("-R", "--results-path")
	parser.add_argument("-f", "--format")
	parser.add_argument("-d", "--details", action="store_true")
	parser.add_argument("--single-legend", action="store_true")
	parser.add_argument("--same-limits", action="store_true") # unused
	parser.add_argument("-F", "--full", action="store_true")
	parser.add_argument("-w", "--workload-runs", action="append")

	args = parser.parse_args()
	remote.RemoteHost.logs_dir = args.logs_path or remote.RemoteHost.logs_dir

	if args.result:
		import results

		results.results_dir = args.results_path or results.results_dir
		results.plot_format = args.format or results.plot_format
		results.throughput_time_index = False
		results.throughput_marker_interval = 1

		if args.workload is not None:
			results.SingleInstanceExperimentResult(
				result_experiments, renaissance.Renaissance,
				get_config(args.workload, args.n_runs, args.full),
				args.details, True, **configurations[args.workload][-1]
			).save_results()
			return

		cmd_args = ["-r"]
		if args.logs_path is not None:
			cmd_args.extend(("-L", args.logs_path))
		if args.results_path is not None:
			cmd_args.extend(("-R", args.results_path))
		if args.format is not None:
			cmd_args.extend(("-f", args.format))
		if args.details:
			cmd_args.append("-d")
		if args.full:
			cmd_args.append("-F")

		workload_runs = dict(w.split(":") for w in args.workload_runs or ())
		util.parallelize(
			lambda w: util.run([__file__, w, "-n", workload_runs.get(w) or str(args.n_runs)] + cmd_args, check=True),
			all_workloads
		)

		results.RenaissanceSingleInstanceAllWorkloadsResult(
			result_experiments, all_workloads, get_config(None, args.n_runs, args.full),
			workload_runs, args.details, {w: configurations[w][-1] for w in all_workloads}
		).save_results(
			legends={0: True} if args.single_legend else None
		)
		return

	hosts = [renaissance.Renaissance.new_host(*h) for h in remote.load_hosts(args.hosts_file)]

	util.verbose = args.verbose
	util.set_sigint_handler()

	if args.cleanup:
		cluster = make_cluster(args.workload, hosts, args.subset, args.n_runs)
		#NOTE: assuming same credentials for all hosts
		passwd = getpass.getpass()
		cluster.check_sudo_passwd(passwd)
		cluster.full_cleanup(passwd=passwd)
		return

	cluster = make_cluster(args.workload, hosts, args.subset, args.n_runs, args.full, args.skip_complete)
	cluster.run_all_experiments(run_experiments, skip_cleanup=True)


if __name__ == "__main__":
	main()
