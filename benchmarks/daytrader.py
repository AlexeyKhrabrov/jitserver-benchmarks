import docker
import liberty
import openj9
import shared


class DayTraderHost(shared.BenchmarkHost):
	def __init__(self, *args, **kwargs):
		super().__init__("daytrader", *args, **kwargs)

	def benchmark_prereqs(self, *, passwd=None):
		super().benchmark_prereqs(passwd=passwd, exclude="docker-copyedit.py")

	def benchmark_setup(self, db2_path=None, *,
	                    build_db2=False, tune=False, **kwargs):
		args = []
		if build_db2:
			args.append("--db2")
		if db2_path is not None:
			self.rsync_put(db2_path, "db2.tar.gz")
			args.append("db2.tar.gz")
		if tune:
			args.append("--tune")

		super().benchmark_setup(args, exclude="docker-copyedit.py", **kwargs)

	def full_cleanup(self, *, passwd=None):
		super().full_cleanup("db2", passwd=passwd)


class DB2Instance(docker.ContainerInstance):
	def __init__(self, config, host, benchmark, config_name, instance_id, *,
	             reserve_cpus=True, collect_stats=False):
		super().__init__(
			host, "db2", benchmark, config_name, instance_id,
			start_log_line="The ACTIVATE DATABASE command completed successfully",
			reserve_cpus=reserve_cpus, collect_stats=collect_stats
		)
		self.config = config
		self.reserved_cpus = self.get_reserved_cpus(config.docker_config)

	def get_port(self):
		return 50000 + self.instance_id

	def start(self, experiment, run_id, attempt_id):
		cmd = [
			"daytrader/run_db2.sh", str(self.instance_id)
		] + self.config.docker_config.docker_args(self.host, self.reserved_cpus)

		super().start(cmd, experiment.name.lower(), run_id, attempt_id,
		              timeout=30.0)


class DayTrader(liberty.Liberty):
	@staticmethod
	def name(): return "daytrader"

	@staticmethod
	def new_host(*args, **kwargs): return DayTraderHost(*args, **kwargs)

	@staticmethod
	def db_name(): return "db2"

	@staticmethod
	def new_db_instance(*args, **kwargs): return DB2Instance(*args, **kwargs)

	@staticmethod
	def full_init_log_line(): return "Settings from daytrader.properties:"

	@staticmethod
	def base_config():
		result = shared.base_config()
		result.application_config.jvm_config = openj9.JVMConfig(
			scc_size="192m",
		)
		result.application_config.start_timeout = 2 * 60.0# seconds
		result.application_config.stop_timeout = 20.0# seconds
		result.application_config.stop_attempts = 6
		return result

	@staticmethod
	def xxsmall_config(jmeter_pin_cpus=False):
		return shared.xxsmall_config(DayTrader.base_config(), 3, 1, jmeter_pin_cpus)

	@staticmethod
	def xsmall_config(jmeter_pin_cpus=False):
		return shared.xsmall_config(DayTrader.base_config(), 4, 1, jmeter_pin_cpus)

	@staticmethod
	def small_config(jmeter_pin_cpus=False):
		return shared.small_config(DayTrader.base_config(), 6, 1, jmeter_pin_cpus)

	@staticmethod
	def medium_config(jmeter_pin_cpus=False):
		return shared.medium_config(DayTrader.base_config(), 12, 2, jmeter_pin_cpus)

	@staticmethod
	def large_config(jmeter_pin_cpus=False):
		return shared.large_config(DayTrader.base_config(), 24, 4, jmeter_pin_cpus)
