import datetime
import signal

import openj9
import shared
import remote


class PetClinicHost(shared.BenchmarkHost):
	def __init__(self, *args, **kwargs):
		super().__init__("petclinic", *args, **kwargs)


class DummyDBInstance(remote.ServerInstance):
	def __init__(self, config, host, *args, **kwargs):
		super().__init__(host, "dummy", *args, **kwargs)
		self.config = config

	def get_port(self): return 0
	def start(self, *args): pass
	def stop(self, *args, **kwargs): pass
	def cleanup(self): pass


class PetClinic:
	@staticmethod
	def name(): return "petclinic"

	@staticmethod
	def new_host(*args, **kwargs): return PetClinicHost(*args, **kwargs)

	@staticmethod
	def db_name(): return None

	@staticmethod
	def new_db_instance(*args, **kwargs): return DummyDBInstance(*args, **kwargs)

	@staticmethod
	def start_log_line(): return "Started PetClinicApplication in"

	@staticmethod
	def error_log_line(): return "ERROR"

	@staticmethod
	def stop_signal(): return signal.SIGTERM

	@staticmethod
	def start_stop_ts_file(): return "petclinic.log"

	@staticmethod
	def stop_log_line(): return "Cache 'vets' removed from EhcacheManager"

	@staticmethod
	def parse_start_stop_ts(line):
		return datetime.datetime.strptime(
			" ".join(line.split(maxsplit=2)[:-1]) + "000", "%Y-%m-%d %H:%M:%S.%f"
		).replace(tzinfo=datetime.timezone.utc)

	@staticmethod
	def base_config():
		result = shared.base_config()
		result.application_config.jvm_config = openj9.JVMConfig(
			scc_size="80m",
		)
		result.application_config.start_timeout = 2 * 60.0# seconds
		result.application_config.stop_timeout = 10.0# seconds
		result.application_config.stop_attempts = 6
		return result

	@staticmethod
	def xxsmall_config(jmeter_pin_cpus=False):
		return shared.xxsmall_config(PetClinic.base_config(), 1, 1, jmeter_pin_cpus)

	@staticmethod
	def xsmall_config(jmeter_pin_cpus=False):
		return shared.xsmall_config(PetClinic.base_config(), 2, 1, jmeter_pin_cpus)

	@staticmethod
	def small_config(jmeter_pin_cpus=False):
		return shared.small_config(PetClinic.base_config(), 3, 1, jmeter_pin_cpus)

	@staticmethod
	def medium_config(jmeter_pin_cpus=False):
		return shared.medium_config(PetClinic.base_config(), 8, 3, jmeter_pin_cpus)

	@staticmethod
	def large_config(jmeter_pin_cpus=False):
		return shared.large_config(PetClinic.base_config(), 16, 6, jmeter_pin_cpus)
