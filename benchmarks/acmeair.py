import docker
import liberty
import openj9
import shared


class AcmeAirHost(shared.BenchmarkHost):
	def __init__(self, *args, **kwargs):
		super().__init__("acmeair", *args, **kwargs)

	def full_cleanup(self, *, passwd=None):
		super().full_cleanup("mongo", passwd=passwd)


class MongoInstance(docker.ContainerInstance):
	def __init__(self, config, host, benchmark, config_name, instance_id, *, reserve_cpus=True, collect_stats=False):
		super().__init__(host, "mongo", benchmark, config_name, instance_id, start_log_line="Waiting for connections",
		                 reserve_cpus=reserve_cpus, collect_stats=collect_stats)
		self.config = config
		self.reserved_cpus = self.get_reserved_cpus(config.docker_config)

	def get_port(self):
		return 27017 + self.instance_id

	def start(self, experiment, run_id, attempt_id):
		cmd = (["acmeair/run_mongo.sh", str(self.instance_id)] +
		       self.config.docker_config.docker_args(self.host, self.reserved_cpus))

		super().start(cmd, experiment.name.lower(), run_id, attempt_id, timeout=5.0)

		cmd = ["acmeair/mongo_init.sh", str(self.instance_id)]
		self.host.run(cmd, remote_output=self.output_path("mongo_init.log"), check=True)


class AcmeAir(liberty.Liberty):
	@staticmethod
	def name(): return "acmeair"

	@staticmethod
	def new_host(*args, **kwargs): return AcmeAirHost(*args, **kwargs)

	@staticmethod
	def db_name(): return "mongo"

	@staticmethod
	def new_db_instance(*args, **kwargs): return MongoInstance(*args, **kwargs)

	@staticmethod
	def base_config():
		result = shared.base_config()
		result.application_config.jvm_config = openj9.JVMConfig(
			scc_size="96m",
		)
		result.application_config.start_timeout = 60.0 # seconds
		result.application_config.stop_timeout = 10.0 # seconds
		result.application_config.stop_attempts = 6
		return result

	@staticmethod
	def xxsmall_config(jmeter_pin_cpus=False):
		return shared.xxsmall_config(AcmeAir.base_config(), 2, 1, jmeter_pin_cpus)

	@staticmethod
	def xsmall_config(jmeter_pin_cpus=False):
		return shared.xsmall_config(AcmeAir.base_config(), 2, 1, jmeter_pin_cpus)

	@staticmethod
	def small_config(jmeter_pin_cpus=False):
		return shared.small_config(AcmeAir.base_config(), 4, 1, jmeter_pin_cpus)

	@staticmethod
	def medium_config(jmeter_pin_cpus=False):
		return shared.medium_config(AcmeAir.base_config(), 8, 2, jmeter_pin_cpus)

	@staticmethod
	def large_config(jmeter_pin_cpus=False):
		return shared.large_config(AcmeAir.base_config(), 18, 4, jmeter_pin_cpus)
