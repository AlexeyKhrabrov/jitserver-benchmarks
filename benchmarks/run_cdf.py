#!/usr/bin/env python3

import argparse
import getpass

import acmeair
import daytrader
import docker
import jitserver
import petclinic
import remote
import results
import shared
import util


jmeter_durations = {
	"acmeair": 6 * 60, # seconds
	"daytrader": 15 * 60, # seconds
	"petclinic": 3 * 60, # seconds
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


def get_config(bench, jmeter, experiment, n_runs,
               equal_resources, skip_complete_runs=False):
	result = bench.small_config(False)
	result.name = "cdf_{}_{}".format("eq" if equal_resources else "ne", "full" if jmeter else "start")

	result.jitserver_config.server_vlog = True
	result.jitserver_config.client_vlog = True
	result.jitserver_config.server_extra_stats = True
	result.jitserver_config.client_extra_stats = True

	result.application_config.start_interval = float("+inf") # seconds
	result.application_config.save_jitdump = (bench.name() == "petclinic")
	result.jmeter_config.duration = jmeter_durations[bench.name()]

	result.n_instances = 1
	result.aotcache_extra_instance = True
	result.run_jmeter = jmeter
	result.n_runs = n_runs
	result.skip_complete_runs = skip_complete_runs

	if equal_resources:
		result.jitserver_docker_config = docker.DockerConfig(ncpus=1, pin_cpus=True)
		ncpus = 1 if experiment.is_jitserver() else 2
		result.application_config.docker_config.ncpus = ncpus

	return result

bench_cls = {
	"acmeair": acmeair.AcmeAir,
	"daytrader": daytrader.DayTrader,
	"petclinic": petclinic.PetClinic
}

def make_cluster(bench, hosts, subset, jmeter, experiment, n_runs, equal_resources, skip_complete_runs=False):
	host0 = hosts[(2 * subset) % len(hosts)]
	host1 = hosts[(2 * subset + 1) % len(hosts)]

	return shared.BenchmarkCluster(get_config(bench, jmeter, experiment, n_runs, equal_resources, skip_complete_runs),
	                               bench, jitserver_hosts=[host0], db_hosts=[host0],
	                               application_hosts=[host1], jmeter_hosts=[host0])


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("benchmark")
	parser.add_argument("hosts_file", nargs="?")
	parser.add_argument("subset", type=int, nargs="?")

	parser.add_argument("-e", "--equal-resources", action="store_true")
	parser.add_argument("-n", "--n-runs", type=int, nargs="?", const=5)
	parser.add_argument("--skip-complete-runs", action="store_true")
	parser.add_argument("-c", "--cleanup", action="store_true")
	parser.add_argument("-j", "--jmeter", action="store_true")
	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument("-L", "--logs-path")
	parser.add_argument("-r", "--result", action="store_true")
	parser.add_argument("-R", "--results-path")
	parser.add_argument("-f", "--format")
	parser.add_argument("--single-legend", action="store_true")
	parser.add_argument("--same-limits", action="store_true") # unused

	args = parser.parse_args()
	remote.RemoteHost.logs_dir = args.logs_path or remote.RemoteHost.logs_dir
	results.results_dir = args.results_path or results.results_dir
	results.plot_format = args.format or results.plot_format

	bench = bench_cls[args.benchmark]()

	if args.result:
		c = get_config(bench, args.jmeter, result_experiments[0], args.n_runs, args.equal_resources)
		results.SingleInstanceExperimentResult(result_experiments, bench, c).save_results(
			legends={"comp_times": False} if args.single_legend else None, cdf_plots=True
		)
		return

	hosts = [bench.new_host(*h) for h in remote.load_hosts(args.hosts_file)]

	util.verbose = args.verbose
	util.set_sigint_handler()

	if args.cleanup:
		cluster = make_cluster(bench, hosts, args.subset, args.jmeter,
		                       run_experiments[0], args.n_runs, args.equal_resources)
		#NOTE: assuming same credentials for all hosts
		passwd = getpass.getpass()
		cluster.check_sudo_passwd(passwd)
		cluster.full_cleanup(passwd=passwd)
		return

	for e in run_experiments:
		cluster = make_cluster(bench, hosts, args.subset, args.jmeter, e, args.n_runs,
		                       args.equal_resources, args.skip_complete_runs)
		cluster.run_all_experiments([e], skip_cleanup=True)


if __name__ == "__main__":
	main()
