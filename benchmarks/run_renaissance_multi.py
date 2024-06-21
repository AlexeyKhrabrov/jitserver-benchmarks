#!/usr/bin/env python3

import argparse
import getpass

import jitserver
import remote
import renaissance
import shared
import util


num_hosts = 4
#NOTE: assuming 16 cpus per host, 1 cpu per container
num_instances = 48

# start_interval, full_iterations, invocation_interval
configurations = {
	"akka-uct":         (21.0, 8,   70.0,  dict()),
	"als":              (27.0, 20,  105.0, dict()),
	"chi-square":       (15.0, 24,  40.0,  dict()),
	"db-shootout":      (20.0, 8,   65.0,  dict()),
	"dec-tree":         (22.0, 40,  75.0,  dict()),
	"dotty":            (19.0, 100, 60.0,  dict()),
	"finagle-chirper":  (17.0, 40,  45.0,  dict()),
	"finagle-http":     (17.0, 12,  45.0,  dict()),
	"fj-kmeans":        (13.0, 8,   35.0,  dict()),
	"future-genetic":   (9.0,  12,  20.0,  dict()),
	"gauss-mix":        (18.0, 20,  55.0,  dict()),
	"log-regression":   (20.0, 80,  65.0,  dict()),
	"mnemonics":        (9.0,  8,   20.0,  dict()),
	"movie-lens":       (25.0, 16,  85.0,  dict()),
	"naive-bayes":      (35.0, 12,  170.0, dict()),
	"page-rank":        (22.0, 8,   75.0,  dict()),
	"par-mnemonics":    (9.0,  8,   20.0,  dict()),
	"philosophers":     (5.0,  40,  10.0,  dict()),
	"reactors":         (21.0, 8,   70.0,  dict()),
	"rx-scrabble":      (7.0,  50,  15.0,  dict()),
	"scala-doku":       (9.0,  8,   20.0,  dict()),
	"scala-kmeans":     (5.0,  30,  10.0,  dict()),
	"scala-stm-bench7": (7.0,  30,  15.0,  dict()),
	"scrabble":         (7.0,  24,  15.0,  dict()),
}

all_workloads = list(configurations)


experiments = (
	jitserver.Experiment.LocalJIT,
	jitserver.Experiment.JITServer,
	jitserver.Experiment.AOTCache,
	jitserver.Experiment.ProfileCache,
	jitserver.Experiment.AOTPrefetcher,
	jitserver.Experiment.FullCache,
)


def get_config(workload, n_runs, full, skip_complete=False):
	config = renaissance.Renaissance.base_config()
	config.name = "".join(("multi_", "full_" if full else "1st_", workload or ""))
	if config.name.endswith("_"):
		config.name = config.name[:-1]

	config.jitserver_config.client_threads = 15
	config.jitserver_config.client_thread_activation_factor = 8
	config.jitserver_config.server_threads = 256
	config.jitserver_config.server_resource_stats = True

	config.application_config.start_interval = configurations[workload][0] if workload else None
	config.n_instances = num_instances
	config.n_runs = n_runs
	config.skip_complete = skip_complete

	return config

def make_cluster(workload, hosts, subset, n_runs, full, skip_complete=False):
	config = get_config(workload, n_runs, full, skip_complete)

	host0 = hosts[subset * num_hosts]
	app_hosts = hosts[subset * num_hosts + 1:(subset + 1) * num_hosts]

	return shared.BenchmarkCluster(
		config, renaissance.Renaissance,
		jitserver_hosts=[host0], db_hosts=[host0], application_hosts=app_hosts, jmeter_hosts=[host0],
		extra_args=renaissance.Renaissance.extra_args(workload, config.application_config,
		                                              configurations[workload][1] if full else 1),
		fix_log_cmd=renaissance.Renaissance.fix_log_cmd()
	)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("workload", nargs="?")
	parser.add_argument("hosts_file", nargs="?")
	parser.add_argument("subset", type=int, nargs="?")

	parser.add_argument("-n", "--n-runs", type=int, default=3)
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
	parser.add_argument("-o", "--overlays", action="store_true")

	args = parser.parse_args()
	remote.RemoteHost.logs_dir = args.logs_path or remote.RemoteHost.logs_dir

	if args.result:
		import results

		results.results_dir = args.results_path or results.results_dir
		results.plot_format = args.format or results.plot_format
		results.throughput_time_index = False
		results.throughput_marker_interval = 1

		if args.workload is not None:
			config = get_config(args.workload, args.n_runs, args.full)
			results.ScaleExperimentResult(
				experiments, renaissance.Renaissance, config, args.details, True, **configurations[args.workload][-1]
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

		results.RenaissanceMultiInstanceAllWorkloadsResult(
			experiments, all_workloads, get_config(None, args.n_runs, args.full),
			workload_runs, args.details, {w: configurations[w][-1] for w in all_workloads}
		).save_results(
			legends={0: True} if args.single_legend else None, overlays=args.overlays
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
	cluster.run_all_experiments(experiments)

if __name__ == "__main__":
	main()
