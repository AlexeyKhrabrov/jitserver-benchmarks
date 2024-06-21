import datetime

import docker
import openj9
import shared


class RenaissanceHost(shared.BenchmarkHost):
	def __init__(self, *args, **kwargs):
		super().__init__("renaissance", *args, **kwargs)

	def benchmark_prereqs(self, *, passwd=None):
		super().benchmark_prereqs(passwd=passwd, exclude=("renaissance.jar", "JITServerPlugin.jar"))

	def benchmark_setup(self, jdk_ver, *, update=False, **kwargs):
		args = [self.jdk_path(jdk_ver)]
		if update:
			args.append("--update")

		super().benchmark_setup(args, exclude=("renaissance.jar", "JITServerPlugin.jar"), **kwargs)


class Renaissance:
	@staticmethod
	def name(): return "renaissance"

	@staticmethod
	def new_host(*args, **kwargs): return RenaissanceHost(*args, **kwargs)

	@staticmethod
	def db_name(): return None

	@staticmethod
	def new_db_instance(*args, **kwargs): return shared.DummyDBInstance(*args, **kwargs)

	@staticmethod
	def start_log_line(): return "Z Setup complete"

	@staticmethod
	def error_log_line(): return None

	@staticmethod
	def stop_signal(): return None

	@staticmethod
	def start_stop_ts_file(): return "renaissance.log"

	@staticmethod
	def stop_log_line(): return "Z Benchmark complete"

	@staticmethod
	def parse_start_stop_ts(line):
		return datetime.datetime.strptime(line.split(maxsplit=1)[0][:-1],
		                                  "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc)

	@staticmethod
	def extra_args(workload, application_config, repetitions=None, *, no_forced_gc=False):
		args = [workload]

		if repetitions is not None:
			args.extend(("--repetitions", repetitions))
		if no_forced_gc:
			args.append("--no-forced-gc")
		if application_config.sleep_time:
			args.extend(("--with-arg", "-S={}".format(int(application_config.sleep_time * 1000)))) # milliseconds
		if application_config.save_jitdump or application_config.save_javacore:
			args.extend(("--with-arg", "-q"))
		if application_config.save_scc_stats:
			args.extend(("--with-arg", "-s"))

		return " ".join(str(arg) for arg in args)

	@staticmethod
	def fix_log_cmd():
		return ["sed", "-i", "-z",
		        "s/WARNING: This benchmark provides no result that can be validated.\\n"
			    "There is no way to check that no silent failure occurred.\\n//g"]

	@staticmethod
	def base_config():
		result = shared.base_config()

		result.jitserver_config.server_scratch_space_factor = None
		result.jitserver_config.disable_preexistence = True
		result.jitserver_config.disable_known_objects = True
		result.jitserver_config.disable_nooptserver = True
		result.jitserver_config.disable_inlining_aggressiveness = True
		result.jitserver_config.disable_unresolved_is_cold = True
		result.jitserver_config.noclassgc = True
		# Workaround for stdout and stderr redirected to /dev/null before shutdown
		result.jitserver_config.client_duplicate_stdouterr = True

		result.application_config.docker_config = docker.DockerConfig(
			ncpus=1,
			memory="4g",
			pin_cpus=True,
			network="host",
		)
		result.application_config.jvm_config = openj9.JVMConfig(
			scc_size="256m",
		)
		result.application_config.start_timeout = 60.0 # seconds
		result.application_config.sleep_time = 1.0 # seconds
		result.application_config.stop_timeout = 30 * 60.0 # seconds
		result.application_config.stop_attempts = None

		result.jmeter_config.docker_config = docker.DockerConfig() # unused
		result.jmeter_config.duration = 0 # unused

		return result
