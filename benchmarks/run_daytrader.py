#!/usr/bin/env python3

import argparse
import getpass

import daytrader
import docker
import jitserver
import openj9
import remote
import shared
import util


experiments = (
	jitserver.Experiment.LocalJIT,
	jitserver.Experiment.JITServer,
	jitserver.Experiment.AOTCache,
)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("hosts_file")
	parser.add_argument("n_instances", type=int)
	parser.add_argument("n_runs", type=int)

	parser.add_argument("-c", "--cleanup", action="store_true")
	parser.add_argument("-j", "--jmeter", action="store_true")
	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument("-L", "--logs-path")

	args = parser.parse_args()
	remote.RemoteHost.logs_dir = args.logs_path or remote.RemoteHost.logs_dir

	config = shared.BenchmarkConfig(
		name="test",
		jitserver_config=jitserver.JITServerConfig(
			server_vlog=True,
			client_vlog=True,
			detailed_vlog=True,
			server_extra_stats=True,
			client_extra_stats=True,
			jdk_ver=8,
			debug=False,
			portable_scc=False,
			forceaot=True,
			nodelay_aotload=True,
			svm_at_startup=False,
			client_threads=None,
			localjit_memlimit=None,
			server_threads=None,
			server_codecache=None,
			server_memlimit=None,
			require_jitserver=True,
			disable_active_thread_thresholds=True,
			disable_gcr_threshold=False,
			server_scratch_space_factor=1,
			reconnect_wait_time=None,
			client_socket_timeout=None,
			server_socket_timeout=None,
			encryption=False,
			use_internal_addr=False,
			share_romclasses=True,
			romclass_cache_partitions=None,
			aotcache_name=None,
			stop_sleep_time=2.0,# seconds
			stop_timeout=10.0,# seconds
			stop_attempts=6,
			kill_remote_on_timeout=False,
			save_javacore=True,
		),
		jitserver_docker_config=None,
		db_config=shared.DBConfig(
			docker_config=docker.DockerConfig(
				ncpus=None,
				memory=None,
				pin_cpus=False,
				network="host",
			),
			use_internal_addr=False,
		),
		application_config=shared.ApplicationConfig(
			docker_config=docker.DockerConfig(
				ncpus=1,
				memory="1g",
				pin_cpus=True,
				network="host",
			),
			jvm_config=openj9.JVMConfig(
				heap_size=None,
				virtualized=False,
				scc_size="192m",
				nojit=False,
			),
			populate_scc=False,
			populate_scc_no_aot=False,
			populate_scc_run_jmeter=False,
			populate_scc_bench=None,
			use_internal_addr=False,
			share_scc=False,
			start_interval=None,# seconds
			start_timeout=2 * 60.0,# seconds
			sleep_time=1.0,# seconds
			stop_timeout=10.0,# seconds
			stop_attempts=6,
			kill_remote_on_timeout=False,
			javacore_interval=None,
			save_javacore=True,
			save_scc_stats=True,
		),
		jmeter_config=shared.JMeterConfig(
			docker_config=docker.DockerConfig(
				ncpus=1,
				memory="4g",
				pin_cpus=True,
				network="host",
			),
			jvm_config=openj9.JVMConfig(
				# defaults
			),
			nthreads=6,
			duration=6 * 60,# seconds
			summariser_interval=6,# seconds; minimum is 6
			latency_data=False,
			report_data=False,
			keep_running=True,
			keep_scc=True,
			stop_timeout=3 * 60,# seconds
			scc_extra_duration=None,
			duration_includes_start=False,
		),
		n_jitservers=1,
		n_dbs=1,
		n_instances=args.n_instances,
		aotcache_extra_instance=False,
		populate_aotcache_bench=None,
		run_jmeter=args.jmeter,
		n_runs=args.n_runs,
		attempts=1,
		skip_runs=(),
		skip_complete_runs=False,
		n_invocations=None,
		idle_time=None,
		collect_stats=True,
	)

	hosts = [daytrader.DayTraderHost(*h)
	         for h in remote.load_hosts(args.hosts_file)]

	cluster = shared.BenchmarkCluster(
		config, daytrader.DayTrader,
		jitserver_hosts=[hosts[0]], db_hosts=[hosts[0]],
		application_hosts=[hosts[1]], jmeter_hosts=[hosts[0]]
	)

	util.verbose = args.verbose
	util.set_sigint_handler()

	if args.cleanup:
		#NOTE: assuming same credentials for all hosts
		cluster.full_cleanup(passwd=getpass.getpass())
		return

	cluster.run_all_experiments(experiments)


if __name__ == "__main__":
	main()
