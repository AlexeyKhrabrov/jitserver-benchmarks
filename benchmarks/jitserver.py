import enum
import os.path
import signal
import time

import docker
import openj9
import remote
import util

module_dir = os.path.dirname(__file__)


class Experiment(enum.IntEnum):
	LocalJIT = 0
	JITServer = 1
	AOTCache = 2
	AOTCacheWarm = 3
	ProfileCache = 4
	ProfileCacheWarm = 5
	AOTPrefetcher = 6
	AOTPrefetcherWarm = 7
	FullCache = 8
	FullCacheWarm = 9

	def is_jitserver(self):
		return self is not Experiment.LocalJIT

	def is_aotcache(self):
		return self in (Experiment.AOTCache, Experiment.AOTCacheWarm,
		                Experiment.AOTPrefetcher, Experiment.AOTPrefetcherWarm,
		                Experiment.FullCache, Experiment.FullCacheWarm)

	def is_profilecache(self):
		return self in (Experiment.ProfileCache, Experiment.ProfileCacheWarm,
		                Experiment.FullCache, Experiment.FullCacheWarm)

	def is_aotprefetcher(self):
		return self in (Experiment.AOTPrefetcher, Experiment.AOTPrefetcherWarm,
		                Experiment.FullCache, Experiment.FullCacheWarm)

	def is_eager(self):
		return self.is_profilecache() or self.is_aotprefetcher()

	def is_cache(self):
		return self.is_aotcache() or self.is_profilecache()

	def is_warm_cache(self):
		return self in (Experiment.AOTCacheWarm, Experiment.ProfileCacheWarm,
		                Experiment.AOTPrefetcherWarm, Experiment.FullCacheWarm)

	def to_single_instance(self):
		if self.is_warm_cache():
			return (Experiment(self.value - 1), 1)
		else:
			return (self, 0)

	def to_non_warm(self):
		if self.is_warm_cache():
			return Experiment(self.value - 1)
		elif self.is_cache():
			return Experiment.JITServer
		else:
			return self


class JITServerConfig:
	def __init__(self, *,
		server_vlog=False, detailed_server_vlog=False, client_vlog=False, detailed_client_vlog=False,
		server_extra_stats=False, client_extra_stats=False, server_resource_stats=False, jdk_ver=8, debug=False,
		portable_scc=False, noaot=False, forceaot=False, nodelay_aotload=False, aot_default_counts=False,
		store_remote_aot=False, svm_at_startup=False, client_threads=None, client_thread_activation_factor=None,
		localjit_memlimit=None, server_threads=None, server_codecache=None, server_memlimit=None,
		require_jitserver=False, disable_active_thread_thresholds=False, disable_gcr_threshold=False,
		server_scratch_space_factor=None, reconnect_wait_time=None, client_socket_timeout=None,
		server_socket_timeout=None, session_purge_time=None, session_purge_interval=None, encryption=False,
		use_internal_addr=False, share_romclasses=False, romclass_cache_partitions=None, aotcache_name=None,
		stop_sleep_time=None, stop_timeout=None, stop_attempts=None, kill_remote_on_timeout=False, save_jitdump=False,
		save_javacore=False, prefetch_all=False, prefetch_start_only=False, scount=None, pcount=None,
		profile_reuse_threshold=None, aot_prefetcher_no_profile_cache=False, bytecode_keep_other_weight=False,
		fanin_keep_other_weight=False, disable_inlining=False, disable_fanin=False, disable_jit_profiling=False,
		disable_per_client_allocators=False, disable_recompilation=False, disable_preexistence=False,
		disable_known_objects=False, disable_nooptserver=False, disable_inlining_aggressiveness=False,
		disable_unresolved_is_cold=False, noclassgc=False, throughput_mode=False, profile_more=False,
		client_malloc_trim_time=None, client_duplicate_stdouterr=False, comp_stats_on_jitdump=False,
		exclude_methods=None, aotcache_detailed_memory_usage=False
	):
		self.server_vlog = server_vlog
		self.detailed_server_vlog = detailed_server_vlog
		self.client_vlog = client_vlog
		self.detailed_client_vlog = detailed_client_vlog
		self.server_extra_stats = server_extra_stats
		self.client_extra_stats = client_extra_stats
		self.server_resource_stats = server_resource_stats
		self.jdk_ver = jdk_ver
		self.debug = debug
		self.portable_scc = portable_scc
		self.noaot = noaot
		self.forceaot = forceaot
		self.nodelay_aotload = nodelay_aotload
		self.aot_default_counts = aot_default_counts
		self.store_remote_aot = store_remote_aot
		self.svm_at_startup = svm_at_startup
		self.client_threads = client_threads
		self.client_thread_activation_factor = client_thread_activation_factor
		self.localjit_memlimit = localjit_memlimit
		self.server_threads = server_threads
		self.server_codecache = server_codecache
		self.server_memlimit = server_memlimit
		self.require_jitserver = require_jitserver
		self.disable_active_thread_thresholds = disable_active_thread_thresholds
		self.disable_gcr_threshold = disable_gcr_threshold
		self.server_scratch_space_factor = server_scratch_space_factor
		self.reconnect_wait_time = reconnect_wait_time # milliseconds
		self.client_socket_timeout = client_socket_timeout # milliseconds
		self.server_socket_timeout = server_socket_timeout # milliseconds
		self.session_purge_time = session_purge_time # milliseconds
		self.session_purge_interval = session_purge_interval # milliseconds
		self.encryption = encryption
		self.use_internal_addr = use_internal_addr
		self.share_romclasses = share_romclasses
		self.romclass_cache_partitions = romclass_cache_partitions
		self.aotcache_name = aotcache_name
		self.stop_sleep_time = stop_sleep_time # seconds
		self.stop_timeout = stop_timeout # seconds
		self.stop_attempts = stop_attempts
		self.kill_remote_on_timeout = kill_remote_on_timeout
		self.save_jitdump = save_jitdump
		self.save_javacore = save_javacore
		self.prefetch_all = prefetch_all
		self.prefetch_start_only = prefetch_start_only
		self.scount = scount
		self.pcount = pcount
		self.profile_reuse_threshold = profile_reuse_threshold
		self.bytecode_keep_other_weight = bytecode_keep_other_weight
		self.fanin_keep_other_weight = fanin_keep_other_weight
		self.disable_inlining = disable_inlining
		self.disable_fanin = disable_fanin
		self.disable_jit_profiling = disable_jit_profiling
		self.disable_per_client_allocators = disable_per_client_allocators
		self.disable_recompilation = disable_recompilation
		self.disable_preexistence = disable_preexistence
		self.disable_known_objects = disable_known_objects
		self.disable_nooptserver = disable_nooptserver
		self.disable_inlining_aggressiveness = disable_inlining_aggressiveness
		self.disable_unresolved_is_cold = disable_unresolved_is_cold
		self.noclassgc = noclassgc
		self.throughput_mode = throughput_mode
		self.profile_more = profile_more
		self.client_malloc_trim_time = client_malloc_trim_time # milliseconds
		self.client_duplicate_stdouterr = client_duplicate_stdouterr
		self.comp_stats_on_jitdump = comp_stats_on_jitdump
		self.exclude_methods = exclude_methods
		self.aotcache_detailed_memory_usage = aotcache_detailed_memory_usage

	def verbose_args(self, vlog_path, detailed):
		tags = ["compilePerformance"]
		if detailed:
			tags.extend(("failures", "JITServer"))
		return "verbose={{{}}},vlog={}".format("|".join(tags), vlog_path)

	def jitserver_args(self, experiment, *, vlog_path, jitserver_port=None, cert_path=None, key_path=None):
		args = ["-Xshareclasses:none"]
		jit_opts = []

		if self.server_vlog:
			jit_opts.append(self.verbose_args(vlog_path, self.detailed_server_vlog))
		if jitserver_port is not None:
			args.append("-XX:JITServerPort={}".format(jitserver_port))
		if self.server_threads is not None:
			args.append("-XcompilationThreads{}".format(self.server_threads))
		if self.server_codecache is not None:
			args.append("-Xcodecachetotal{}".format(self.server_codecache))
		if self.server_memlimit is not None:
			jit_opts.append("scratchSpaceLimit={}".format(
			                util.size_to_bytes(self.server_memlimit) // 1024))
		if self.disable_active_thread_thresholds:
			jit_opts.extend(("highActiveThreadThreshold=1000000000", "veryHighActiveThreadThreshold=1000000000"))
		if self.server_scratch_space_factor is not None:
			jit_opts.append("scratchSpaceFactorWhenJITServerWorkload={}".format(self.server_scratch_space_factor))
		if self.server_socket_timeout is not None:
			args.append("-XX:JITServerTimeout={}".format(self.server_socket_timeout))
		if self.session_purge_time is not None:
			jit_opts.extend(("oldAge={}".format(self.session_purge_time),
			                 "oldAgeUnderLowMemory={}".format(self.session_purge_time)))
		if self.session_purge_interval is not None:
			jit_opts.append("timeBetweenPurges={}".format(self.session_purge_interval))
		if self.encryption:
			args.extend(("-XX:JITServerSSLCert={}".format(cert_path), "-XX:JITServerSSLKey={}".format(key_path)))
		if self.share_romclasses:
			args.append("-XX:+JITServerShareROMClasses")
		if self.romclass_cache_partitions is not None:
			jit_opts.append("sharedROMClassCacheNumPartitions={}".format(self.romclass_cache_partitions))
		if self.save_jitdump:
			args.append("-Xdump:jit:events=user")
		if self.save_javacore:
			args.append("-Xdump:java:events=user,file=javacore.txt")

		if experiment.is_aotcache():
			args.append("-XX:+JITServerUseAOTCache")
		if experiment.is_profilecache() or experiment.is_aotprefetcher():
			args.append("-XX:+JITServerShareProfilingData")
		if experiment.is_aotprefetcher():
			args.append("-XX:+JITServerUseAOTPrefetcher")
		if experiment.is_profilecache():
			args.append("-XX:+JITServerPreCompileProfiledMethods")
		if self.profile_reuse_threshold is not None:
			args.append("-XX:JITServerProfileReuseThresholdClients={}".format(self.profile_reuse_threshold))

		if self.disable_jit_profiling:
			jit_opts.extend(("disableJProfiling", "disableJProfilingThread",
			                 "disableProfiling", "disableSamplingJProfiling"))
		if self.disable_recompilation or self.disable_preexistence:
			jit_opts.append("disableInvariantArgumentPreexistence")
		if self.disable_known_objects:
			jit_opts.append("disableKnownObjectTable")
		if self.disable_inlining:
			jit_opts.extend(("disableInlining", "disableInliningDuringVPAtWarm"))
		if self.disable_fanin:
			jit_opts.append("disableInlinerFanIn")

		if jit_opts:
			opts = ",".join(jit_opts)
			args.extend(("-Xjit:{}".format(opts), "-Xaot:{}".format(opts)))
		return args

	def jvm_args(self, experiment, *, vlog_path, jitserver_addr, jitserver_port=None,
	             cert_path=None, scc_no_aot=False, save_jitdump=False, save_javacore=False):
		args = ["-XX:{}PortableSharedCache".format("+" if self.portable_scc else "-")]
		jit_opts = []

		if self.client_vlog:
			jit_opts.append(self.verbose_args(vlog_path, self.detailed_client_vlog))
		if self.noaot or scc_no_aot:
			jit_opts.extend(("noload", "nostore"))
		if self.forceaot:
			jit_opts.append("forceAOT")
		if self.nodelay_aotload:
			jit_opts.append("disableDelayRelocationForAOTCompilations")
		if self.aot_default_counts:
			jit_opts.append("dontLowerCountsForAotCold")
		if self.disable_gcr_threshold:
			jit_opts.append("GCRQueuedThresholdForCounting=1000000000")
		if save_jitdump:
			args.append("-Xdump:jit:events=user")
		if save_javacore:
			args.append("-Xdump:java:events=user,file=javacore.txt")

		if experiment.is_jitserver():
			args.extend(("-XX:+UseJITServer", "-XX:JITServerAddress={}".format(jitserver_addr)))
			if jitserver_port is not None:
				args.append("-XX:JITServerPort={}".format(jitserver_port))
			if self.client_threads is not None:
				args.append("-XcompilationThreads{}".format(self.client_threads))
			if self.client_thread_activation_factor is not None:
				jit_opts.append("jitclientThreadActivationQueueWeightFactor={}".format(
				                self.client_thread_activation_factor))
			if self.require_jitserver:
				args.append("-XX:+RequireJITServer")
			if self.reconnect_wait_time is not None:
				jit_opts.append("reconnectWaitTimeMs={}".format(self.reconnect_wait_time))
			if self.client_socket_timeout is not None:
				args.append("-XX:JITServerTimeout={}".format(self.client_socket_timeout))
			if self.encryption:
				args.append("-XX:JITServerSSLRootCerts={}".format(cert_path))
		else:
			if self.localjit_memlimit is not None:
				jit_opts.append("scratchSpaceLimit={}".format(util.size_to_bytes(self.localjit_memlimit) // 1024))

		if experiment.is_aotcache():
			args.append("-XX:+JITServerUseAOTCache")
		if experiment.is_eager():
			args.append("-XX:+JITServerShareProfilingData") # need full profiling data for recompilations after AOT loads
			if self.prefetch_all:
				args.append("-XX:+JITServerPrefetchAllData")
			if self.prefetch_start_only:
				jit_opts.append("aotPrefetcherDoNotRequestAtClassLoad")
		if experiment.is_aotprefetcher():
			args.append("-XX:+JITServerUseAOTPrefetcher")
			if self.scount is not None:
				jit_opts.append("aotPrefetcherSCount={}".format(self.scount))
		if experiment.is_profilecache():
			args.append("-XX:+JITServerPreCompileProfiledMethods")
			if self.pcount is not None:
				jit_opts.append("aotPrefetcherPCount={}".format(self.pcount))
		if experiment.is_cache() and self.aotcache_name:
			args.append("-XX:JITServerAOTCacheName={}".format(self.aotcache_name))

		if self.disable_jit_profiling:
			jit_opts.extend(("disableJProfiling", "disableJProfilingThread",
			                 "disableProfiling", "disableSamplingJProfiling"))
		if self.disable_recompilation:
			jit_opts.append("inhibitRecompilation")
		if self.disable_recompilation or self.disable_preexistence:
			jit_opts.append("disableInvariantArgumentPreexistence")
		if self.disable_known_objects:
			jit_opts.append("disableKnownObjectTable")
		if self.disable_inlining:
			jit_opts.extend(("disableInlining", "disableInliningDuringVPAtWarm"))
		if self.disable_fanin:
			jit_opts.append("disableInlinerFanIn")
		if self.disable_nooptserver:
			jit_opts.append("disableSelectiveNoServer")
		if self.disable_inlining_aggressiveness:
			jit_opts.append("dontVaryInlinerAggressivenessWithTime")#TODO: ? can remove (should be off by default)
		if self.noclassgc:
			args.append("-Xnoclassgc")

		if self.exclude_methods:
			jit_opts.extend("exclude={{{}}}".format(m) for m in self.exclude_methods)
			jit_opts.extend("dontInline={{{}}}".format(m) for m in self.exclude_methods)

		if jit_opts:
			opts = ",".join(jit_opts)
			args.extend(("-Xjit:{}".format(opts), "-Xaot:{}".format(opts)))
		return args

	common_stats_env_vars = ("TR_silentEnv", "TR_PrintResourceUsageStats", "TR_PrintJITServerAOTCacheStats")
	extra_stats_env_vars = ("TR_PrintJITServerMsgStats", "TR_PrintJITServerIPMsgStats")

	@staticmethod
	def env(env_vars):
		return {k: 1 for k in env_vars}

	def jitserver_env(self):
		env_vars = list(self.common_stats_env_vars) + ["TR_PrintCompStats", "TR_PrintCompTime"]

		if self.server_extra_stats:
			env_vars.extend(self.extra_stats_env_vars)
		if self.disable_per_client_allocators:
			env_vars.append("TR_DisablePerClientPersistentAllocation")
		if self.bytecode_keep_other_weight:
			env_vars.append("TR_SharedBytecodeDataKeepOtherWeight")
		if self.fanin_keep_other_weight:
			env_vars.append("TR_SharedFanInDataKeepOtherWeight")
		if self.disable_unresolved_is_cold:
			env_vars.append("TR_DisableUnresolvedIsCold")
		if self.throughput_mode:
			env_vars.append("TR_ThroughputMode")
		if self.aotcache_detailed_memory_usage:
			env_vars.append("TR_JITServerAOTCacheDetailedMemoryUsage")

		return self.env(env_vars)

	def jvm_env(self):
		env_vars = list(self.common_stats_env_vars)

		if self.comp_stats_on_jitdump:
			env_vars.extend(("TR_PrintCompStatsOnJITDump", "TR_PrintCompTimeOnJITDump"))
		else:
			env_vars.extend(("TR_PrintCompStats", "TR_PrintCompTime"))

		if self.client_extra_stats:
			env_vars.extend(self.extra_stats_env_vars)
		if self.store_remote_aot:
			env_vars.append("TR_enableRemoteAOTMethodStorage")
		if self.svm_at_startup:
			env_vars.append("TR_DontDisableSVMDuringStartup")
		if self.disable_unresolved_is_cold:
			env_vars.append("TR_DisableUnresolvedIsCold")
		if self.throughput_mode:
			env_vars.append("TR_ThroughputMode")
		if self.profile_more:
			env_vars.append("TR_IProfileMore")
		if self.client_duplicate_stdouterr:
			env_vars.append("TR_DuplicateStdOutErr")

		result = self.env(env_vars)
		if self.client_malloc_trim_time is not None:
			result["TR_MallocTrimTimeMs"] = self.client_malloc_trim_time

		return result


#NOTE: assuming single instance per host
class JITServerInstance(remote.ServerInstance):
	def __init__(self, config, host, benchmark, config_name, instance_id, *, reserve_cpus=True, collect_stats=False):
		super().__init__(host, "jitserver", benchmark, config_name, instance_id,
		                 start_log_line="JITServer is ready to accept incoming requests",
		                 reserve_cpus=reserve_cpus, collect_stats=collect_stats or config.server_resource_stats)
		self.config = config

	def jitserver_port(self):
		return 38400 + self.instance_id

	def cleanup(self):
		self.host.cleanup_process("jitserver")
		super().cleanup()

	def start(self, experiment, run_id, attempt_id):
		jdk_path = self.host.jdk_path(self.config.jdk_ver, self.config.debug)

		cmd = [os.path.join(jdk_path, "bin/jitserver")]
		cmd.extend(self.config.jitserver_args(experiment, vlog_path=self.output_path("vlog_server"),
		           jitserver_port=self.jitserver_port(), cert_path=os.path.join(jdk_path, "cert.pem"),
		           key_path=os.path.join(jdk_path, "key.pem")))

		return super().start(cmd, experiment.name.lower(), run_id, attempt_id,
		                     env=self.config.jitserver_env(), timeout=5.0)

	def store_openj9_crash_files(self):
		self.store_crash_files(".", store_patterns=openj9.crash_store_patterns,
		                       keep_patterns=openj9.crash_keep_patterns)

	def stop(self, prefix=None, store=True):
		util.sleep(self.config.stop_sleep_time)

		if self.config.save_jitdump or self.config.save_javacore:
			self.remote_proc.kill(signal.SIGQUIT)
			time.sleep(1.0)

		exc = super().stop(store=False, timeout=self.config.stop_timeout, attempts=self.config.stop_attempts,
		                   raise_on_failure=False, kill_remote_on_timeout=self.config.kill_remote_on_timeout)

		if exc is None:
			cmd = []
			if self.config.save_javacore:
				cmd.extend(("mv", "javacore.txt", self.output_dir(), "&&"))
			cmd.extend(("rm", "-f", "javacore*.txt"))
			self.host.run(cmd, globs=True)
		else:
			self.store_openj9_crash_files()

		if store:
			self.host.run(["mv", self.output_path("vlog_server*"), self.output_path("vlog_server.log")], globs=True)
			self.store_output(exc is None, prefix)
		return exc


class JITServerHost(docker.DockerHost, openj9.OpenJ9Host):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def jitserver_setup(self, *, scripts_only=False, buildkit=False, sudo=False, passwd=None):
		self.rsync_put(os.path.join(module_dir, "jitserver/"), "jitserver/")
		self.update_scripts()
		if scripts_only:
			return

		output_path = self.log_path("jitserver_setup")
		t0 = time.monotonic()
		if sudo:
			self.run_sudo(["jitserver/build.sh"], output=output_path, check=True, passwd=passwd)
		else:
			self.run(["jitserver/build.sh"], output=output_path, check=True,
			         env={"DOCKER_BUILDKIT": 1} if buildkit else None)
		t1 = time.monotonic()

		print("jitserver setup on {} took {:.2f} seconds".format(self.addr, t1 - t0))


class JITServerContainerInstance(openj9.OpenJ9ContainerInstance, JITServerInstance):
	def __init__(self, docker_config, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.docker_config = docker_config
		self.reserved_cpus = self.get_reserved_cpus(docker_config)

	def store_output(self, success, prefix=None):
		vlog = "vlog_server" if self.config.server_vlog else None
		super().store_output(success, prefix, vlog=vlog)

	def start(self, experiment, run_id, attempt_id):
		cmd = [
			"jitserver/run.sh",
			self.benchmark,
			str(self.instance_id),
			self.host.jdk_path(self.config.jdk_ver, self.config.debug),
			util.args_str(self.config.jitserver_args(experiment, vlog_path="/output/vlogs/vlog_server",
			              jitserver_port=self.jitserver_port(), cert_path="/cert.pem", key_path="/key.pem")),
			util.args_str("{}={}".format(k, v) for k, v in self.config.jitserver_env().items())
		] + self.docker_config.docker_args(self.host, self.reserved_cpus)

		return super(JITServerInstance, self).start(cmd, experiment.name.lower(), run_id, attempt_id)

	def stop(self, prefix=None):
		exc = super(openj9.OpenJ9ContainerInstance, self).stop(store=False)
		if exc is None and self.config.save_javacore:
			self.host.cp_from_container(self.get_name(), "/output/javacore.txt", self.output_dir())
		self.host.remove_container(self.get_name())
		self.store_output(exc is None, prefix)
		return exc


def jitserver_instance(docker_config, *args, **kwargs):
	if docker_config is not None:
		return JITServerContainerInstance(docker_config, *args, **kwargs)
	else:
		return JITServerInstance(*args, **kwargs)
