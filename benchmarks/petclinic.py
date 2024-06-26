import datetime
import signal

import openj9
import shared


class PetClinicHost(shared.BenchmarkHost):
	def __init__(self, *args, **kwargs):
		super().__init__("petclinic", *args, **kwargs)


class PetClinic:
	@staticmethod
	def name(): return "petclinic"

	@staticmethod
	def new_host(*args, **kwargs): return PetClinicHost(*args, **kwargs)

	@staticmethod
	def db_name(): return None

	@staticmethod
	def new_db_instance(*args, **kwargs): return shared.DummyDBInstance(*args, **kwargs)

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
		return datetime.datetime.strptime(" ".join(line.split(maxsplit=2)[:-1]) + "000",
		                                  "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc)

	@staticmethod
	def base_config():
		result = shared.base_config()

		result.jitserver_config.comp_stats_on_jitdump = True # workaround for premature JVM shutdown
		result.jitserver_config.exclude_methods = ( # workaround for AOT miscompilation bug
			"org/springframework/beans/AbstractPropertyAccessor.setPropertyValues(Lorg/springframework/beans/PropertyValues;ZZ)V",
		)

		result.application_config.jvm_config = openj9.JVMConfig(
			scc_size="128m",
		)
		result.application_config.start_timeout = 2 * 60.0 # seconds
		result.application_config.stop_timeout = 10.0 # seconds
		result.application_config.stop_attempts = 6
		result.application_config.save_jitdump = True # since stats output at shutdown can be truncated

		return result

	@staticmethod
	def xxsmall_config(jmeter_pin_cpus=False):
		return shared.xxsmall_config(PetClinic.base_config(), 1, 1, jmeter_pin_cpus)

	@staticmethod
	def xsmall_config(jmeter_pin_cpus=False):
		return shared.xsmall_config(PetClinic.base_config(), 1, 1, jmeter_pin_cpus)

	@staticmethod
	def small_config(jmeter_pin_cpus=False):
		return shared.small_config(PetClinic.base_config(), 2, 1, jmeter_pin_cpus)

	@staticmethod
	def medium_config(jmeter_pin_cpus=False):
		return shared.medium_config(PetClinic.base_config(), 4, 1, jmeter_pin_cpus)

	@staticmethod
	def large_config(jmeter_pin_cpus=False):
		return shared.large_config(PetClinic.base_config(), 10, 2, jmeter_pin_cpus)
