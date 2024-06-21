#!/usr/bin/env python3

import argparse
import getpass

import docker
import jitserver
import openj9
import remote
import renaissance
import shared
import util


experiments = (
	jitserver.Experiment.LocalJIT,
	jitserver.Experiment.JITServer,
	jitserver.Experiment.AOTCache,
	jitserver.Experiment.ProfileCache,
	jitserver.Experiment.AOTPrefetcher,
	jitserver.Experiment.FullCache,
)

result_experiments = (
	jitserver.Experiment.LocalJIT,
	jitserver.Experiment.JITServer,
	jitserver.Experiment.AOTCache,
	jitserver.Experiment.AOTCacheWarm,
	jitserver.Experiment.ProfileCache,
	jitserver.Experiment.ProfileCacheWarm,
	jitserver.Experiment.AOTPrefetcher,
	jitserver.Experiment.AOTPrefetcherWarm,
	jitserver.Experiment.FullCache,
	jitserver.Experiment.FullCacheWarm,
)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("hosts_file")
	parser.add_argument("subset", type=int)
	parser.add_argument("n_instances", type=int)
	parser.add_argument("n_runs", type=int)
	parser.add_argument("workload")
	parser.add_argument("repetitions", type=int, nargs="?")

	parser.add_argument("-c", "--cleanup", action="store_true")
	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument("-L", "--logs-path")
	parser.add_argument("-r", "--result", action="store_true")
	parser.add_argument("-R", "--results-path")
	parser.add_argument("-f", "--format")
	parser.add_argument("-d", "--details", action="store_true")

	args = parser.parse_args()
	remote.RemoteHost.logs_dir = args.logs_path or remote.RemoteHost.logs_dir

	config = shared.BenchmarkConfig(
		name="test_" + args.workload,
		jitserver_config=jitserver.JITServerConfig(
			server_vlog=True,
			detailed_server_vlog=True,
			client_vlog=True,
			detailed_client_vlog=True,
			server_extra_stats=True,
			client_extra_stats=True,
			server_resource_stats=True,
			jdk_ver=8,
			debug=False,
			portable_scc=False,
			noaot=False,
			forceaot=False,
			nodelay_aotload=True,
			aot_default_counts=False,
			store_remote_aot=False,
			svm_at_startup=False,
			client_threads=None,
			client_thread_activation_factor=None,
			localjit_memlimit=None,
			server_threads=None,
			server_codecache=None,
			server_memlimit=None,
			require_jitserver=True,
			disable_active_thread_thresholds=True,
			disable_gcr_threshold=False,
			server_scratch_space_factor=1,
			reconnect_wait_time=None, # milliseconds
			client_socket_timeout=0, # milliseconds
			server_socket_timeout=0, # milliseconds
			session_purge_time=0, # milliseconds
			session_purge_interval=0, # milliseconds
			encryption=False,
			use_internal_addr=False,
			share_romclasses=True,
			romclass_cache_partitions=None,
			aotcache_name=None,
			stop_sleep_time=2.0, # seconds
			stop_timeout=10.0, # seconds
			stop_attempts=6,
			kill_remote_on_timeout=False,
			save_jitdump=True, # since stats output at shutdown can be truncated
			save_javacore=False,
			prefetch_all=True,
			prefetch_start_only=True,
			pcount=0,
			scount=0,
			profile_reuse_threshold=1,
			bytecode_keep_other_weight=True,
			fanin_keep_other_weight=True,
			disable_inlining=False,
			disable_fanin=False,
			disable_jit_profiling=True,
			disable_recompilation=False,
			disable_preexistence=True,
			disable_known_objects=True,
			disable_nooptserver=True,
			disable_inlining_aggressiveness=True,
			disable_unresolved_is_cold=True,
			noclassgc=True,
			throughput_mode=False,
			profile_more=False,
			client_malloc_trim_time=None,
			client_duplicate_stdouterr=True, # workaround for stdout and stderr redirected to /dev/null before shutdown
			comp_stats_on_jitdump=False,
			exclude_methods=None,
			aotcache_detailed_memory_usage=False,
		),
		jitserver_docker_config=None,
		db_config=shared.DBConfig(docker_config=docker.DockerConfig()), # unused
		application_config=shared.ApplicationConfig(
			docker_config=docker.DockerConfig(
				ncpus=1,
				memory="3g",
				pin_cpus=True,
				network="host",
			),
			jvm_config=openj9.JVMConfig(
				heap_size=None,
				virtualized=False,
				scc_size="256m",
				nojit=False,
			),
			populate_scc=False,
			populate_scc_no_aot=False,
			populate_scc_run_jmeter=False,
			populate_scc_bench=None,
			use_internal_addr=False,
			share_scc=False,
			start_interval=float("+inf"), # seconds
			start_timeout=60.0, # seconds
			sleep_time=1.0, # seconds
			stop_timeout=30 * 60.0, # seconds
			stop_attempts=None,
			kill_remote_on_timeout=False,
			javacore_interval=None,
			save_jitdump=False,
			save_javacore=False,
			save_scc_stats=False,
		),
		jmeter_config=shared.JMeterConfig(docker_config=docker.DockerConfig(), jvm_config=openj9.JVMConfig(),
		                                  nthreads=0, duration=0, summariser_interval=0), # unused
		n_jitservers=1,
		n_dbs=1,
		n_instances=args.n_instances,
		cache_extra_instance=True,
		populate_cache_bench=None,
		run_jmeter=False,
		n_runs=args.n_runs,
		attempts=3,
		skip_runs=(),
		skip_complete=True,
		n_invocations=None,
		idle_time=None,
		invocation_interval=None,
		collect_stats=True,
	)

	if args.result:
		import results

		results.results_dir = args.results_path or results.results_dir
		results.plot_format = args.format or results.plot_format
		results.throughput_time_index = False
		results.throughput_marker_interval = 1

		assert args.n_instances == 1
		results.SingleInstanceExperimentResult(
			result_experiments, renaissance.Renaissance, config, args.details,
		).save_results()
		return

	hosts = [renaissance.RenaissanceHost(*h) for h in remote.load_hosts(args.hosts_file)]
	host0 = hosts[(2 * args.subset) % len(hosts)]
	host1 = hosts[(2 * args.subset + 1) % len(hosts)]

	cluster = shared.BenchmarkCluster(
		config, renaissance.Renaissance,
		jitserver_hosts=[host0], db_hosts=[host0], application_hosts=[host1], jmeter_hosts=[host0],
		extra_args=renaissance.Renaissance.extra_args(args.workload, config.application_config, args.repetitions,
		                                              no_forced_gc=not config.jitserver_config.noclassgc),
		fix_log_cmd=renaissance.Renaissance.fix_log_cmd()
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
