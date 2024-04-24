import itertools
import json
import math
import os
import os.path
import shutil
import signal
import stat
import time

import docker
import jitserver
import openj9
import remote
import util

module_dir = os.path.dirname(__file__)


class BenchmarkHost(jitserver.JITServerHost):
	def __init__(self, benchmark, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.benchmark = benchmark

	def update_benchmark(self, exclude=None):
		self.rsync_put(os.path.join(module_dir, self.benchmark + "/"), self.benchmark + "/", exclude=exclude)

	def benchmark_prereqs(self, *, passwd=None, exclude=None):
		self.update_benchmark(exclude)

		cmd = [os.path.join(self.benchmark, "prereqs.sh")]
		self.run_sudo(cmd, passwd=passwd, output=self.log_path("{}_prereqs".format(self.benchmark)), check=True)

	def benchmark_setup(self, args=None, *, scripts_only=False, clean=False, prune=False,
	                    exclude=None, buildkit=False, sudo=False, passwd=None):
		if clean:
			self.clean_images()
		self.update_benchmark(exclude)
		self.jitserver_setup(scripts_only=scripts_only, buildkit=buildkit, sudo=sudo, passwd=passwd)
		if scripts_only:
			return

		cmd = [os.path.join(self.benchmark, "build_all.sh")]
		if args is not None:
			cmd.extend(args)

		output_path = self.log_path("{}_setup".format(self.benchmark))
		t0 = time.monotonic()
		if sudo:
			self.run_sudo(cmd, output=output_path, check=True, passwd=passwd)
		else:
			self.run(cmd, output=output_path, check=True, env={"DOCKER_BUILDKIT": 1} if buildkit else None)
		t1 = time.monotonic()

		print("{} setup on {} took {:.2f} seconds".format(self.benchmark, self.addr, t1 - t0))

		if prune:
			self.prune_images()

	def scc_path(self, component, instance_id):
		return os.path.join(self.benchmark, "{}_scc_{}".format(component, instance_id))

	def scc_cleanup(self, component):
		self.run(["rm", "-rf", self.scc_path(component, "*")], globs=True)

	def stored_scc_path(self, component, instance_id):
		return self.scc_path(component, "{}_{}".format(instance_id, "stored"))

	def store_scc(self, component, instance_id, *, dst_name=None):
		src = self.scc_path(component, instance_id)
		dst = self.stored_scc_path(component, dst_name or instance_id)
		self.run(["rm", "-rf", dst, "&&", "mv", src, dst], check=True)

	def load_scc(self, component, instance_id, *, src_name=None, check=False):
		src = self.stored_scc_path(component, src_name or instance_id)
		dst = self.scc_path(component, instance_id)
		result = self.run(["rm", "-rf", dst, "&&", "cp", "-a", src, dst])
		return result.returncode == 0

	def full_cleanup(self, db_name=None, *, passwd=None):
		container_ids = self.get_output(["docker", "ps", "-a", "-q"]).split()
		if container_ids:
			self.run(["docker", "rm", "-f"] + container_ids)
		self.cleanup_process("jitserver")
		self.run(["killall", "top", "docker"])

		names = ["jitserver", self.benchmark, "jmeter"] + ([db_name] if db_name is not None else [])
		cmd = ["rm", "-rf"]
		cmd.extend(os.path.join(self.benchmark, "{}_*/".format(name)) for name in names)

		if passwd is not None:
			self.run_sudo(cmd, passwd=passwd, globs=True)
		else:
			self.run(cmd, globs=True)


class DBConfig:
	def __init__(self, *, docker_config, use_internal_addr=False):
		self.docker_config = docker_config
		self.use_internal_addr = use_internal_addr


class ApplicationConfig:
	def __init__(self, *,
		docker_config, jvm_config, populate_scc=False, populate_scc_no_aot=False, populate_scc_run_jmeter=False,
		populate_scc_bench=None, use_internal_addr=False, share_scc=False, start_interval=None,
		start_timeout=None, sleep_time=None, stop_timeout=None, stop_attempts=None, kill_remote_on_timeout=False,
		javacore_interval=None, save_jitdump=False, save_javacore=False, save_scc_stats=False
	):
		self.docker_config = docker_config
		self.jvm_config = jvm_config
		self.populate_scc = populate_scc
		self.populate_scc_no_aot = populate_scc_no_aot
		self.populate_scc_run_jmeter = populate_scc_run_jmeter
		self.populate_scc_bench = populate_scc_bench
		self.use_internal_addr = use_internal_addr
		self.share_scc = share_scc
		self.start_interval = start_interval # seconds
		self.start_timeout = start_timeout # seconds
		self.sleep_time = sleep_time # seconds
		self.stop_timeout = stop_timeout # seconds
		self.stop_attempts = stop_attempts
		self.kill_remote_on_timeout = kill_remote_on_timeout
		self.javacore_interval = javacore_interval
		self.save_jitdump = save_jitdump
		self.save_javacore = save_javacore
		self.save_scc_stats = save_scc_stats


class ApplicationInstance(openj9.OpenJ9ContainerInstance):
	def __init__(self, config, host, bench, config_name, instance_id, db_instance, jitserver_instance, *,
	             benchmark=None, reserve_cpus=True, collect_stats=False, extra_args=None, fix_log_cmd=None):
		super().__init__(host, bench.name(), benchmark or bench.name(), config_name, instance_id,
		                 start_log_line=bench.start_log_line(), error_log_line=bench.error_log_line(),
		                 stop_signal=bench.stop_signal(), reserve_cpus=reserve_cpus, collect_stats=collect_stats)
		self.config = config
		self.bench = bench
		self.jitclient_config = jitserver_instance.config
		self.db_addr = self.host.get_host_addr(db_instance.host, db_instance.config.use_internal_addr)
		self.db_port = db_instance.get_port()
		self.jitserver_addr = self.host.get_host_addr(jitserver_instance.host,
		                                              jitserver_instance.config.use_internal_addr)
		self.jitserver_instance = jitserver_instance
		self.reserved_cpus = self.get_reserved_cpus(config.docker_config)
		self.javacore_proc = None
		self.extra_args = extra_args
		self.fix_log_cmd = fix_log_cmd

	def scc_path(self):
		if not self.config.populate_scc and not self.config.share_scc:
			return None
		return self.host.scc_path(self.benchmark, "shared" if self.config.share_scc else self.instance_id)

	def store_output(self, success, prefix=None, invocation_attempt=None):
		vlog = "vlog_client" if self.jitclient_config.client_vlog else None
		super().store_output(success, prefix, invocation_attempt, vlog=vlog)

	def start(self, experiment, run_id, attempt_id, scc_run=False):
		if self.config.populate_scc and not self.config.share_scc and not scc_run:
			self.host.load_scc(self.benchmark, self.instance_id, src_name="populated", check=True)

		cmd = [
			os.path.join(self.bench.name(), "run_{}.sh".format(self.bench.name())),
			str(self.instance_id),
			self.db_addr,
			str(self.db_port),
			self.host.jdk_path(self.jitclient_config.jdk_ver, self.jitclient_config.debug),
			self.scc_path() or "",
			util.args_str(
				self.config.jvm_config.jvm_args() + (
					self.jitclient_config.jvm_args(
						experiment, vlog_path="/output/vlogs/vlog_client",
						jitserver_addr=self.jitserver_addr,
						jitserver_port=self.jitserver_instance.jitserver_port(),
						cert_path="/cert.pem",
						scc_no_aot=self.config.populate_scc_no_aot if scc_run else False,
						save_jitdump=self.config.save_jitdump,
						save_javacore=self.config.save_javacore
					))
			),
			util.args_str("{}={}".format(k, v) for k, v in self.jitclient_config.jvm_env().items()),
			self.extra_args or ""
		] + self.config.docker_config.docker_args(self.host, self.reserved_cpus)

		exc = super().start(cmd, experiment.name.lower(), run_id, attempt_id,
		                    timeout=self.config.start_timeout, raise_on_failure=False)
		if exc is None:
			if self.config.javacore_interval is not None:
				cmd = ["scripts/collect_javacore.sh", self.get_name(), str(self.remote_proc.remote_pid),
				       str(self.config.javacore_interval), self.output_dir()]
				self.javacore_proc = self.host.start(cmd)
			util.sleep(self.config.sleep_time)

		return exc

	def stop(self, success=True, prefix=None, invocation_attempt=None):
		if self.bench.stop_signal() is not None:
			if self.javacore_proc is not None:
				self.javacore_proc.stop(signal.SIGINT)
				self.javacore_proc = None

			if self.config.save_jitdump or self.config.save_javacore:
				self.remote_proc.kill(signal.SIGQUIT)
				time.sleep(0.5)

			if self.config.save_scc_stats:
				cmd = ["docker", "exec", self.get_name(), "/opt/ibm/java/bin/java",
				       "-Xshareclasses:printStats,name={},cacheDir=/output/.classCache".format(self.bench.name())]
				self.host.run(cmd, remote_output=self.output_path("scc_stats.log"))

			exc = super(openj9.OpenJ9ContainerInstance, self).stop(
				store=False, timeout=self.config.stop_timeout, attempts=self.config.stop_attempts,
				raise_on_failure=False, kill_remote_on_timeout=self.config.kill_remote_on_timeout
			)

		else:
			exc = self.wait(store=False, timeout=self.config.stop_timeout, raise_on_failure=False,
			                kill_remote_on_timeout=self.config.kill_remote_on_timeout,
			                prefix=prefix, invocation_attempt=invocation_attempt)

			if self.javacore_proc is not None:
				self.javacore_proc.stop(signal.SIGINT)
				self.javacore_proc = None

		if exc is None:
			if self.config.save_javacore:
				self.host.cp_from_container(self.get_name(), "/output/javacore.txt", self.output_dir())
		else:
			self.store_openj9_crash_files()

		if self.fix_log_cmd is not None:
			self.host.run(self.fix_log_cmd + [self.log_path()], check=True)

		self.host.remove_container(self.get_name())
		self.store_output(success and (exc is None), prefix, invocation_attempt)
		return exc


class JMeterConfig:
	def __init__(self, *, docker_config, jvm_config, nthreads, duration, summariser_interval,
	             latency_data=False, report_data=False, keep_running=False, stop_timeout=None,
	             scc_extra_duration=None, duration_includes_start=False):
		self.docker_config = docker_config
		self.jvm_config = jvm_config
		self.nthreads = nthreads
		self.duration = duration # seconds
		self.summariser_interval = summariser_interval # seconds
		self.latency_data = latency_data
		self.report_data = report_data
		self.keep_running = keep_running
		self.stop_timeout = stop_timeout # seconds
		self.scc_extra_duration = scc_extra_duration or 0
		self.duration_includes_start = duration_includes_start


class JMeterInstance(docker.ContainerInstance):
	def __init__(self, config, host, bench, config_name, application_instance, n_instances, n_dbs, *,
	             benchmark=None, extra_duration=0, reserve_cpus=True, collect_stats=False):
		super().__init__(host, "jmeter", benchmark or bench.name(), config_name, application_instance.instance_id,
		                 reserve_cpus=reserve_cpus, collect_stats=collect_stats)
		self.config = config
		self.bench = bench
		self.application_instance = application_instance
		self.application_addr = self.host.get_host_addr(application_instance.host,
		                                                application_instance.config.use_internal_addr)
		self.n_instances = n_instances
		self.n_dbs = n_dbs
		self.duration = config.duration + extra_duration
		if not config.duration_includes_start:
			self.duration += config.summariser_interval
		self.reserved_cpus = self.get_reserved_cpus(config.docker_config)

	def get_remote_pid(self):
		pid = util.retry_loop(lambda: self.host.get_host_pid(self.get_name(), "java", check=False,
		                                                     cmd_filter=lambda s: "-version" not in s),
		                      attempts=50, sleep_time=0.1)
		if pid is None:
			raise Exception("No java process started in jmeter container")
		return pid

	def run(self, experiment, run_id, attempt_id, scc_run=False, prefix=None, invocation_attempt=None):
		duration = self.duration
		if scc_run:
			duration += self.config.scc_extra_duration
		if self.config.duration_includes_start:
			duration -= int(self.application_instance.start_time)
		print("jmeter instance {} duration: {} seconds".format(self.instance_id, duration))

		cmd = [
			os.path.join(self.bench.name(), "run_jmeter.sh"),
			str(self.instance_id),
			str(self.n_instances),
			str(self.n_dbs),
			self.application_addr,
			str(self.config.nthreads),
			str(duration),
			str(self.config.summariser_interval),
			str(self.config.latency_data).lower(),
			str(self.config.report_data).lower(),
			util.args_str(self.config.jvm_config.jvm_args())
		] + self.config.docker_config.docker_args(self.host, self.reserved_cpus)

		timeout = ((duration + self.config.stop_timeout) if self.config.stop_timeout is not None else None)

		super().start(cmd, experiment.name.lower(), run_id, attempt_id)
		return self.wait(timeout=timeout, raise_on_failure=False, kill_remote_on_timeout=True,
		                 prefix=prefix, invocation_attempt=invocation_attempt)


class BenchmarkConfig:
	def __init__(self, *,
		name, jitserver_config, jitserver_docker_config, db_config, application_config, jmeter_config,
		n_jitservers, n_dbs, n_instances, cache_extra_instance=False, populate_cache_bench=None,
		run_jmeter, n_runs, attempts, skip_runs=None, skip_complete=False, n_invocations=None,
		idle_time=None, invocation_interval=None, invocation_attempts=None, collect_stats=False
	):
		self.name = name
		self.jitserver_config = jitserver_config
		self.jitserver_docker_config = jitserver_docker_config
		self.db_config = db_config
		self.application_config = application_config
		self.jmeter_config = jmeter_config
		self.n_jitservers = n_jitservers
		self.n_dbs = n_dbs
		self.n_instances = n_instances
		self.cache_extra_instance = cache_extra_instance
		self.populate_cache_bench = populate_cache_bench
		self.run_jmeter = run_jmeter
		self.n_runs = n_runs
		self.attempts = attempts
		self.skip_runs = skip_runs or ()
		self.skip_complete = skip_complete
		self.n_invocations = n_invocations
		self.idle_time = idle_time
		self.invocation_interval = invocation_interval
		self.invocation_attempts = invocation_attempts
		self.collect_stats = collect_stats

	def get_n_instances(self, is_cache):
		if self.cache_extra_instance and is_cache:
			return self.n_instances + 1
		else:
			return self.n_instances

	def jmeter_extra_duration(self, instance_id):
		if (self.run_jmeter and self.jmeter_config.keep_running and
		    (self.application_config.start_interval != float("+inf"))
		):
			return math.ceil((self.application_config.start_interval or 0.0) * (self.n_instances - instance_id - 1))
		else:
			return 0

	def save_json(self, benchmark, experiment=None, run_id=None):
		assert (experiment is None) == (run_id is None)
		path = os.path.join(
			remote.RemoteHost.logs_dir, benchmark, self.name, experiment.name.lower() if experiment is not None else "",
			"run_{}".format(run_id) if run_id is not None else "", "config.json"
		)

		os.makedirs(os.path.dirname(path), exist_ok=True)
		with open(path, "w") as f:
			json.dump(self, f, default=vars, indent="\t")

		if run_id is not None:
			os.chmod(path, os.stat(path).st_mode & ~(stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH))


class BenchmarkCluster(openj9.OpenJ9Cluster):
	def __init__(self, config, bench, *, jitserver_hosts, db_hosts, application_hosts,
	             jmeter_hosts, extra_args=None, fix_log_cmd=None):
		super().__init__(list(set(jitserver_hosts + db_hosts + application_hosts + jmeter_hosts)))
		self.config = config
		self.bench = bench
		self.jitserver_hosts = jitserver_hosts
		self.db_hosts = db_hosts
		self.application_hosts = application_hosts
		self.jmeter_hosts = jmeter_hosts

		self.for_each(docker.DockerHost.reset_reserved_cpus)

		self.jitserver_instances = [
			jitserver.jitserver_instance(config.jitserver_docker_config, config.jitserver_config, jitserver_hosts[i],
			                             bench.name(), config.name, i, collect_stats=config.collect_stats)
			for i in range(config.n_jitservers)
		]

		self.db_instances = [
			bench.new_db_instance(config.db_config, h, bench.name(), config.name, i, collect_stats=config.collect_stats)
			for i, h in zip(
				range(config.n_dbs),
				itertools.cycle(db_hosts)
			)
		]

		self.application_instances = [
			ApplicationInstance(config.application_config, h, bench, config.name, i, d, j,
			                    collect_stats=config.collect_stats, extra_args=extra_args, fix_log_cmd=fix_log_cmd)
			for i, h, d, j in zip(
				range(config.get_n_instances(True)),
				itertools.cycle(application_hosts),
				itertools.cycle(self.db_instances),
				itertools.cycle(self.jitserver_instances)
			)
		]

		self.jmeter_instances = [
			JMeterInstance(config.jmeter_config, h, bench, config.name, l, config.get_n_instances(True), config.n_dbs,
			               extra_duration=self.config.jmeter_extra_duration(i), collect_stats=config.collect_stats)
			for i, h, l in zip(
				range(config.get_n_instances(True)),
				itertools.cycle(jmeter_hosts),
				self.application_instances
			)
		]

	def run_application_and_jmeter(self, jmeter_instance, experiment, run_jmeter, run_id, attempt_id, *,
	                               scc_run=False, prefix=None, invocation_attempt=None):
		application_instance = jmeter_instance.application_instance
		success = not experiment.is_jitserver() or application_instance.jitserver_instance.is_running()

		if success:
			exc = application_instance.start(experiment, run_id, attempt_id, scc_run)
			success = exc is None

		if success and experiment.is_jitserver():
			success = application_instance.jitserver_instance.is_running()

		if success and run_jmeter:
			exc = jmeter_instance.run(experiment, run_id, attempt_id, scc_run, prefix, invocation_attempt)
			success = exc is None

		exc = application_instance.stop(success, prefix, invocation_attempt)
		return success and (exc is None)

	def populate_application_scc(self, host_id):
		bench = self.config.application_config.populate_scc_bench or self.bench

		application_host = self.application_hosts[host_id]
		db_host = (application_host if len(self.application_hosts) > len(self.db_hosts) else self.db_hosts[host_id])
		jmeter_host = (application_host if len(self.application_hosts) > len(self.jmeter_hosts)
		               else self.jmeter_hosts[host_id])

		#NOTE: unused
		jitserver_instance = jitserver.jitserver_instance(
			self.config.jitserver_docker_config, self.config.jitserver_config, application_host, self.bench.name(),
			self.config.name, 0, reserve_cpus=False, collect_stats=self.config.collect_stats
		)
		db_instance = bench.new_db_instance(self.config.db_config, db_host, self.bench.name(), self.config.name,
		                                    host_id, reserve_cpus=False, collect_stats=self.config.collect_stats)
		db_instance.start(jitserver.Experiment.LocalJIT, 0, 0)

		application_instance = ApplicationInstance(
			self.config.application_config, application_host, bench, self.config.name, host_id, db_instance,
			jitserver_instance, benchmark=self.bench.name(), reserve_cpus=False, collect_stats=self.config.collect_stats
		)
		jmeter_instance = JMeterInstance(
			self.config.jmeter_config, jmeter_host, bench, self.config.name,
			application_instance, len(self.application_hosts), 1, benchmark=self.bench.name(),
			reserve_cpus=False, collect_stats=self.config.collect_stats
		)
		success = self.run_application_and_jmeter(jmeter_instance, jitserver.Experiment.LocalJIT,
		                                          self.config.application_config.populate_scc_run_jmeter,
		                                          0, 0, scc_run=True, prefix="scc")

		db_instance.stop(prefix="scc")
		if success:
			application_host.store_scc(self.bench.name(), host_id, dst_name="populated")
		return success

	def populate_all_application_scc(self):
		print("Populating {} scc...".format(self.bench.name()))

		t0 = time.monotonic()
		results = util.parallelize(self.populate_application_scc, range(len(self.application_hosts)))
		t1 = time.monotonic()

		if not all(r for r in results):
			remote.ServerInstance.rename_failed_run(self.hosts[0], self.bench.name(), self.config.name,
			                                        "localjit", 0, 0, "scc")
			raise Exception("Failed to populate {} scc".format(self.bench.name()))

		print("Populated {} scc in {:.2f} seconds".format(self.bench.name(), t1 - t0))

	def populate_cache(self, instance_id, experiment, run_id, attempt_id):
		bench = self.config.populate_cache_bench or self.bench

		db_instance = bench.new_db_instance(self.config.db_config, self.db_hosts[instance_id], self.bench.name(),
		                                    self.config.name, instance_id, reserve_cpus=False,
		                                    collect_stats=self.config.collect_stats)
		db_instance.start(experiment, run_id, attempt_id)

		application_instance = ApplicationInstance(
			self.config.application_config, self.application_hosts[instance_id], bench, self.config.name,
			instance_id, db_instance, self.jitserver_instances[instance_id], benchmark=self.bench.name(),
			reserve_cpus=False, collect_stats=self.config.collect_stats
		)
		jmeter_instance = JMeterInstance(
			self.config.jmeter_config, self.jmeter_hosts[instance_id], self.bench, self.config.name,
			application_instance, len(self.jitserver_hosts), 1, benchmark=self.bench.name(),
			reserve_cpus=False, collect_stats=self.config.collect_stats
		)
		success = self.run_application_and_jmeter(jmeter_instance, experiment, self.config.run_jmeter,
		                                          run_id, attempt_id, prefix="cache")

		db_instance.stop(prefix="cache")
		return success

	def populate_all_cache(self, experiment, run_id, attempt_id):
		print("Populating cache...")

		results = util.parallelize(self.populate_cache, range(len(self.jitserver_hosts)),
		                           experiment, run_id, attempt_id)
		success = all(r for r in results)

		if not success:
			util.parallelize(lambda i: i.stop(prefix="cache"), self.jitserver_instances)
			remote.ServerInstance.rename_failed_run(self.hosts[0], self.bench.name(), self.config.name,
			                                        experiment.name.lower(), run_id, attempt_id, "cache")

		return success

	def run_single_experiment(self, experiment, run_id, attempt_id):
		if experiment.is_jitserver():
			util.parallelize(lambda i: i.start(experiment, run_id, attempt_id), self.jitserver_instances)

			if experiment.is_warm_cache() and not self.populate_all_cache(experiment, run_id, attempt_id):
				return False

		util.parallelize(lambda i: i.start(experiment, run_id, attempt_id), self.db_instances)

		if self.config.application_config.share_scc:
			if self.config.application_config.populate_scc:
				util.parallelize(BenchmarkHost.load_scc, self.application_hosts,
				                 self.bench.name(), "shared", src_name="populated")
			else:
				util.parallelize(BenchmarkHost.scc_cleanup, self.application_hosts, self.bench.name())

		n_instances = self.config.get_n_instances(experiment.is_cache())
		results = util.parallelize(self.run_application_and_jmeter, self.jmeter_instances[0:n_instances],
		                           experiment, self.config.run_jmeter, run_id, attempt_id,
		                           sleep_time=self.config.application_config.start_interval)
		success = all(r for r in results)

		util.parallelize(lambda i: i.stop(), self.db_instances)

		if experiment.is_jitserver():
			results = util.parallelize(lambda i: i.stop(), self.jitserver_instances)
			success = success and all(r is None for r in results)

		if not success:
			remote.ServerInstance.rename_failed_run(self.hosts[0], self.bench.name(), self.config.name,
			                                        experiment.name.lower(), run_id, attempt_id)

		return success

	def cleanup(self):
		util.parallelize(JMeterInstance.cleanup, self.jmeter_instances)
		util.parallelize(ApplicationInstance.cleanup, self.application_instances)
		util.parallelize(BenchmarkHost.scc_cleanup, self.application_hosts, self.bench.name())
		util.parallelize(lambda i: i.cleanup(), self.db_instances)
		util.parallelize(lambda i: i.cleanup(), self.jitserver_instances)
		self.for_each(remote.RemoteHost.run, ["killall", "top", "docker"])

	def full_cleanup(self, *, passwd=None):
		self.for_each(lambda h: h.full_cleanup(passwd=passwd), parallel=True)

	def run_logs_path(self, experiment, run_id):
		return os.path.join(remote.RemoteHost.logs_dir, self.bench.name(), self.config.name,
		                    experiment.name.lower(), "run_{}".format(run_id))

	def is_complete_run(self, experiment, run_id, components, nums_of_instances, files):
		for c in range(len(components)):
			for i in range(nums_of_instances[c]):
				for f in files[c]:
					path = os.path.join(self.run_logs_path(experiment, run_id), "{}_{}".format(components[c], i), f)
					if not os.path.isfile(path):
						print("Missing output file {}".format(path))
						return False

		return True

	def skip_run(self, experiment, run_id, is_density):
		if run_id in self.config.skip_runs:
			print("Skipping experiment {} {} {} run {}".format(
			      self.bench.name(), self.config.name, experiment.name, run_id))
			return True

		if not self.config.skip_complete:
			return False

		components = ("jitserver", self.bench.db_name(), self.bench.name(), "jmeter")

		n_instances = self.config.n_instances * (self.config.n_invocations if is_density else 1)
		nums_of_instances = (
			self.config.n_jitservers if experiment.is_jitserver() else 0,
			self.config.n_dbs if self.bench.db_name() is not None else 0,
			n_instances, # application
			n_instances if self.config.run_jmeter else 0, # jmeter
		)

		jitserver_files = ["jitserver.log"]
		if self.config.jitserver_docker_config is not None:
			jitserver_files.append("cgroup_rusage.log")
		if self.config.jitserver_config.server_vlog:
			jitserver_files.append("vlog_server.log")

		db_files = [(self.bench.db_name() or "") + ".log", "cgroup_rusage.log"]

		app_files = [self.bench.name() + ".log", self.bench.start_stop_ts_file()]
		if self.bench.stop_signal():
			app_files.append("cgroup_rusage.log")
		if self.config.jitserver_config.client_vlog:
			app_files.append("vlog_client.log")

		jmeter_files = ["jmeter.log"]
		if self.config.jmeter_config.latency_data or self.config.jmeter_config.report_data:
			jmeter_files.append("results.jtl")

		if self.config.collect_stats or self.config.jitserver_config.server_resource_stats:
			jitserver_files.append("stats.log")
			if self.config.jitserver_docker_config is not None:
				jitserver_files.append("docker_stats.log")

		if self.config.collect_stats:
			db_files.extend(("stats.log", "docker_stats.log"))
			app_files.extend(("stats.log", "docker_stats.log"))
			jmeter_files.extend(("stats.log", "docker_stats.log"))

		files = (jitserver_files, db_files, app_files, jmeter_files)

		if not self.is_complete_run(experiment, run_id, components, nums_of_instances, files):
			return False

		print("Skipping complete experiment {} {} {} run {}".format(
		      self.bench.name(), self.config.name, experiment.name, run_id))
		return True

	def run_experiment(self, experiment):
		for r in range(self.config.n_runs):
			if self.skip_run(experiment, r, False):
				continue

			shutil.rmtree(self.run_logs_path(experiment, r), ignore_errors=True)
			self.config.save_json(self.bench.name(), experiment, r)

			for i in range(self.config.attempts):
				print("Running experiment {} {} {} run {}/{} attempt {}/{}...".format(self.bench.name(),
				      self.config.name, experiment.name, r, self.config.n_runs, i, self.config.attempts))

				t0 = time.monotonic()
				if self.run_single_experiment(experiment, r, i):
					t1 = time.monotonic()
					print("Finished experiment {} {} {} run {}/{} in {:.2f} seconds".format(
					      self.bench.name(), self.config.name, experiment.name, r, self.config.n_runs, t1 - t0))
					break

				if i == self.config.attempts - 1:
					print("Experiment {} {} {} failed".format(self.bench.name(), self.config.name, experiment.name))
					return False

		return True

	def run_all_experiments(self, experiments, *, skip_cleanup=False):
		if all(self.skip_run(e, r, False) for e in experiments for r in range(self.config.n_runs)):
			print("Skipping complete benchmark {} configuration {}".format(self.bench.name(), self.config.name))
			return

		self.config.save_json(self.bench.name())

		print("Running benchmark {} configuration {}...".format(self.bench.name(), self.config.name))

		if not skip_cleanup:
			self.cleanup()
		try:
			t0 = time.monotonic()

			if self.config.application_config.populate_scc:
				self.populate_all_application_scc()
			for e in experiments:
				self.run_experiment(e)

			t1 = time.monotonic()
			print("Finished benchmark {} configuration {} in {:.2f} seconds".format(
			      self.bench.name(), self.config.name, t1 - t0))

		except KeyboardInterrupt:
			print("Interrupting...")
			self.cleanup()
			raise
		except:
			util.print_exception()
			raise

		if not skip_cleanup:
			self.cleanup()

	def run_invocation(self, invocation_id, instance_id, experiment, run_id, attempt_id):
		current_id = invocation_id * self.config.n_instances + instance_id
		jmeter_instance = self.jmeter_instances[instance_id]
		jmeter_instance.instance_id = current_id
		application_instance = jmeter_instance.application_instance
		application_instance.instance_id = current_id

		t0 = time.monotonic()
		for i in range(self.config.invocation_attempts):
			success = self.run_application_and_jmeter(jmeter_instance, experiment, self.config.run_jmeter,
			                                          run_id, attempt_id, invocation_attempt=i)
			t1 = time.monotonic()

			if success:
				if self.config.invocation_interval:
					sleep_time = max(0.0, self.config.invocation_interval - (t1 - t0))
					print("{} {} {} instance {} invocation {}: sleeping for {:.2f}/{:.2f} seconds".format(
					      self.bench.name(), self.config.name, experiment.name, instance_id,
					      invocation_id, sleep_time, self.config.invocation_interval))
					util.sleep(sleep_time)
				else:
					util.sleep(self.config.idle_time)

				return True

		return False

	def run_instance(self, instance_id, experiment, run_id, attempt_id):
		for i in range(self.config.n_invocations):
			if not self.run_invocation(i, instance_id, experiment, run_id, attempt_id):
				return False
		return True

	def run_single_density_experiment(self, experiment, run_id, attempt_id):
		if experiment.is_jitserver():
			util.parallelize(lambda i: i.start(experiment, run_id, attempt_id), self.jitserver_instances)

		util.parallelize(lambda i: i.start(experiment, run_id, attempt_id), self.db_instances)

		results = util.parallelize(self.run_instance, range(self.config.n_instances), experiment, run_id, attempt_id,
		                           sleep_time=self.config.application_config.start_interval)
		success = all(r for r in results)

		util.parallelize(lambda i: i.stop(), self.db_instances)

		if experiment.is_jitserver():
			results = util.parallelize(lambda i: i.stop(), self.jitserver_instances)
			success = success and all(r is None for r in results)

		if not success:
			remote.ServerInstance.rename_failed_run(self.hosts[0], self.bench.name(), self.config.name,
			                                        experiment.name.lower(), run_id, attempt_id)

		return success

	def run_density_experiment(self, experiment):
		for r in range(self.config.n_runs):
			if self.skip_run(experiment, r, True):
				continue

			shutil.rmtree(self.run_logs_path(experiment, r), ignore_errors=True)
			self.config.save_json(self.bench.name(), experiment, r)

			for i in range(self.config.attempts):
				print("Running experiment {} {} {} run {}/{} attempt {}/{}...".format(self.bench.name(),
				      self.config.name, experiment.name, r, self.config.n_runs, i, self.config.attempts))

				t0 = time.monotonic()
				if self.run_single_density_experiment(experiment, r, i):
					t1 = time.monotonic()
					print("Finished experiment {} {} {} run {}/{} in {:.2f} seconds".format(
					      self.bench.name(), self.config.name, experiment.name, r, self.config.n_runs, t1 - t0))
					break

				if i == self.config.attempts - 1:
					print("Experiment {} {} {} failed".format(self.bench.name(), self.config.name, experiment.name))
					return False

		return True

	def run_all_density_experiments(self, experiments):
		if all(self.skip_run(e, r, True) for e in experiments for r in range(self.config.n_runs)):
			print("Skipping complete benchmark {} configuration {}".format(self.bench.name(), self.config.name))
			return

		self.config.save_json(self.bench.name())

		print("Running benchmark {} configuration {}...".format(self.bench.name(), self.config.name))

		self.full_cleanup()
		try:
			if self.config.application_config.populate_scc:
				self.populate_all_application_scc()
			for e in experiments:
				self.run_density_experiment(e)

		except KeyboardInterrupt:
			print("Interrupting...")
			self.full_cleanup()
			raise
		except:
			util.print_exception()
			raise

		self.full_cleanup()


class DummyDBInstance(remote.ServerInstance):
	def __init__(self, config, host, *args, **kwargs):
		super().__init__(host, "dummy", *args, **kwargs)
		self.config = config

	def get_port(self): return 0
	def start(self, *args): pass
	def stop(self, *args, **kwargs): pass
	def cleanup(self): pass


def base_config():
	return BenchmarkConfig(
		name=None,
		jitserver_config=jitserver.JITServerConfig(
			jdk_ver=8,
			nodelay_aotload=True,
			disable_active_thread_thresholds=True,
			server_scratch_space_factor=1,
			share_romclasses=True,
			stop_sleep_time=2.0, # seconds
			stop_timeout=10.0, # seconds
			stop_attempts=6,
			kill_remote_on_timeout=True,
			save_jitdump=True, # since stats output at shutdown can be truncated
			disable_jit_profiling=True,
		),
		jitserver_docker_config=None,
		db_config=DBConfig(docker_config=docker.DockerConfig()),
		application_config=ApplicationConfig(
			docker_config=None,
			jvm_config=None,
			sleep_time=1.0,# seconds
			kill_remote_on_timeout=True,
		),
		jmeter_config=JMeterConfig(
			docker_config=None,
			jvm_config=openj9.JVMConfig(),
			nthreads=None,
			duration=None, # seconds
			summariser_interval=6, # seconds; minimum is 6
			stop_timeout=60, # seconds
		),
		n_jitservers=1,
		n_dbs=1,
		n_instances=None,
		run_jmeter=False,
		n_runs=1,
		attempts=3,
	)

def update_config(config, name, application_cpus, application_mem, jmeter_cpus,
                  jmeter_mem, jmeter_threads, jmeter_pin_cpus=False):
	config.name = name
	config.application_config.docker_config = docker.DockerConfig(ncpus=application_cpus, memory=application_mem,
	                                                              pin_cpus=True)
	config.jmeter_config.docker_config = docker.DockerConfig(ncpus=jmeter_cpus, memory=jmeter_mem,
	                                                         pin_cpus=jmeter_pin_cpus)
	config.jmeter_config.nthreads = jmeter_threads
	return config


def xxsmall_config(config, jmeter_threads, jmeter_cpus=1, jmeter_pin_cpus=False):
	return update_config(config, "xxsmall", 0.5, "256m", jmeter_cpus, "4g", jmeter_threads, jmeter_pin_cpus)

def xsmall_config(config, jmeter_threads, jmeter_cpus=1, jmeter_pin_cpus=False):
	return update_config(config, "xsmall", 0.5, "512m", jmeter_cpus, "4g", jmeter_threads, jmeter_pin_cpus)

def small_config(config, jmeter_threads, jmeter_cpus=1, jmeter_pin_cpus=False):
	return update_config(config, "small", 1, "1g", jmeter_cpus, "4g", jmeter_threads, jmeter_pin_cpus)

def medium_config(config, jmeter_threads, jmeter_cpus=2, jmeter_pin_cpus=False):
	return update_config(config, "medium", 2, "2g", jmeter_cpus, "4g", jmeter_threads, jmeter_pin_cpus)

def large_config(config, jmeter_threads, jmeter_cpus=4, jmeter_pin_cpus=False):
	return update_config(config, "large", 4, "4g", jmeter_cpus, "4g", jmeter_threads, jmeter_pin_cpus)


def cold_config(config, name=None):
	config.name += name or "_cold"
	return config

def warm_start_config(config, portable=False, name=None):
	config = cold_config(config, name or ("_warm_start" + ("_portable" if portable else "")))
	config.jitserver_config.portable_scc = portable
	config.application_config.populate_scc = True
	return config

def warm_full_config(config, portable=False):
	config = warm_start_config(config, portable, "_warm_full" + ("_portable" if portable else ""))
	config.application_config.populate_scc_run_jmeter = True
	return config
