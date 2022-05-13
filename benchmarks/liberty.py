import datetime
import signal


class Liberty:
	@staticmethod
	def start_log_line(): return "The defaultServer server started in"

	@staticmethod
	def error_log_line(): return "[ERROR"

	@staticmethod
	def stop_signal(): return signal.SIGINT

	@staticmethod
	def start_stop_ts_file(): return "messages.log"

	@staticmethod
	def stop_log_line(): return "The server defaultServer stopped after"

	@staticmethod
	def parse_start_stop_ts(line):
		#[0/00/00 00:00:00:000 GMT] ...
		return datetime.datetime.strptime(
			" ".join(line[1:].split(maxsplit=2)[:-1]) + "000",
			"%m/%d/%y %H:%M:%S:%f"
		).replace(tzinfo=datetime.timezone.utc)
