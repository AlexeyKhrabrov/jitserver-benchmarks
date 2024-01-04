import os.path
import re
import shlex
import shutil
import signal
import subprocess
import time

import util


#NOTE: Enabling multiplexing of ssh connections on the host running this code
# is highly recommended for good performance. To enable multiplexing, add the
# following to the ssh client configuration file ("~/.ssh/config" by default):
#
# Host * # host name pattern
# ControlMaster auto
# ControlPath ~/.ssh/socket/%r@%h:%p
# ControlPersist 600 # seconds
#
# The "~/.ssh/socket" directory must be created if it does not exist.
#
# It is normally also necessary to increase the maximum number of sessions per
# shared connection on the hosts where remote commands are executed. This limit
# is set by the "MaxSessions" parameter in the sshd configuration file on the
# remote host ("/etc/ssh/sshd_config" by default) and should be increased from
# the default 10 to at least 100.


class RemoteProcess:
	def __init__(self, host, ssh_proc, remote_pid):
		self.host = host
		self.ssh_proc = ssh_proc
		self.remote_pid = remote_pid

	def is_running(self):
		return self.ssh_proc.poll() is None

	# Sends a signal to the process group of the remote shell
	def kill(self, sig):
		if self.is_running():
			self.host.run(["kill", "-" + sig.name, "-{}".format(self.remote_pid)])

	def wait(self, *, check=False, timeout=None, kill_remote_on_timeout=False,
	         try_terminate=False, term_timeout=None, term_attempts=None):
		try:
			return util.wait(self.ssh_proc, check=check, timeout=timeout, kill_on_timeout=False)

		except subprocess.TimeoutExpired:
			if kill_remote_on_timeout:
				if try_terminate:
					term_attempts = term_attempts or 1

					for i in range(term_attempts):
						self.kill(signal.SIGTERM)
						try:
							util.wait(self.ssh_proc, timeout=term_timeout, kill_on_timeout=False)
						except subprocess.TimeoutExpired:
							if i == term_attempts - 1:
								self.kill(signal.SIGKILL)

				else:
					self.kill(signal.SIGKILL)

			else:
				self.ssh_proc.terminate()

			util.wait(self.ssh_proc)
			raise

	def stop(self, sig, *, check=False, timeout=None, attempts=None, kill_remote_on_timeout=False):
		attempts = attempts or 1
		for i in range(attempts):
			self.kill(sig)

			try:
				return util.wait(self.ssh_proc, check=check, expect_ret=128 + sig.value,
				                 timeout=timeout, kill_on_timeout=False)

			except subprocess.TimeoutExpired:
				if i == attempts - 1:
					if kill_remote_on_timeout:
						self.kill(signal.SIGKILL)
					self.ssh_proc.terminate()
					util.wait(self.ssh_proc)
					raise

				print("Re-sending signal {} to pid {} on {} attempt {}/{}".format(
				      sig.name, self.remote_pid, self.host.addr, i + 1, attempts))


class RemoteHost:
	def __init__(self, addr, user=None, internal_addr=None, directory=None, use_storage=False):
		self.addr = addr
		self.user = user
		self.internal_addr = internal_addr or addr
		self.directory = directory
		self.use_storage = use_storage
		self.ncpus = None
		self.memory = None

	def host(self):
		return "{}@{}".format(self.user, self.addr) if self.user else self.addr

	logs_dir = "logs"

	def local_output_path(self, path):
		return os.path.join(self.logs_dir, path)

	def log_path(self, name):
		return self.local_output_path("{}_{}.log".format(name, self.addr))

	def ssh_setup(self, *, passwd=None):
		cmd = []
		if passwd is not None:
			cmd.append("sshpass")
		cmd.extend(("ssh-copy-id", "-o", "StrictHostKeyChecking=accept-new", self.host()))

		input_bytes = passwd.encode() if passwd is not None else None
		util.run(cmd, input=input_bytes, check=True, output=self.log_path("ssh_setup"))

	def get_path(self, path):
		return os.path.join(self.directory or "", path or "")

	shell_operators = ("|", "|&", ";", "&", "!", "&&", "||", "(", ")", ">", ">>", "&>", "&>>")

	@classmethod
	def has_operators(cls, cmd):
		return any(s in cls.shell_operators for s in cmd)

	#NOTE: cwd is relative to self.directory (unless absolute)
	#      output file path is relative to cwd (unless absolute)
	def ssh_cmd(self, cmd, tty, get_pid, cwd, env, globs, output, append):
		c = ["ssh"]
		if tty:
			c.append("-t")
		c.append(self.host())

		if get_pid:
			c.extend(("echo", "$$", "&&"))
		c.extend(("cd", shlex.quote(self.get_path(cwd)), "&&"))

		if env:
			c.append("export")
			c.extend("{}={}".format(k, shlex.quote(str(v))) for k, v in env.items())
			c.append("&&")

		operators = self.has_operators(cmd)
		if operators:
			c.append("(")
		else:
			c.append("exec")

		c.extend((s if (s in self.shell_operators) or (globs and ('*' in s)) else shlex.quote(s)) for s in cmd)

		if operators:
			c.append(")")
		if output is not None:
			c.extend(("&>>" if append else "&>", shlex.quote(output)))

		return c

	def run(self, cmd, *, tty=False, remote_cwd=None, env=None, globs=False,
	        remote_output=None, remote_append=False, **kwargs):
		c = self.ssh_cmd(cmd, tty, False, remote_cwd, env, globs, remote_output, remote_append)
		return util.run(c, **kwargs)

	def start(self, cmd, *, tty=False, remote_cwd=None, env=None, globs=False, remote_output=None,
	          remote_append=False, output=None, append=False, attempts=100, sleep_time=0.001, **kwargs):
		c = self.ssh_cmd(cmd, tty, True, remote_cwd, env, globs, remote_output, remote_append)

		if output is None:
			proc = util.start(c, universal_newlines=True, **kwargs)
			return RemoteProcess(self, proc, int(proc.stdout.readline().strip()))

		proc = util.start(c, output=output, append=append, **kwargs)
		# Wait until the remote shell executing the command outputs its pid
		with open(output, "r") as f:
			line = util.retry_loop(lambda: f.readline or None, attempts, sleep_time)
			if not line:
				raise Exception("Failed to start {} on {}".format(cmd, self.addr))
			return RemoteProcess(self, proc, int(line.strip()))

	#NOTE: cannot use shell operators or env vars
	def run_sudo(self, cmd, *, passwd=None, universal_newlines=False, **kwargs):
		assert not self.has_operators(cmd) and "env" not in kwargs

		c = ["sudo"]
		if passwd is not None:
			c.append("-S")
		c.extend(cmd)

		p = (passwd + "\n\n") if passwd is not None else None
		input_bytes = p if universal_newlines or (p is None) else p.encode()
		return self.run(c, tty=passwd is None, input=input_bytes, universal_newlines=universal_newlines, **kwargs)

	def check_sudo_passwd(self, passwd):
		self.run_sudo(["true"], passwd=passwd, check=True)

	@staticmethod
	def rsync_cmd(exclude=None, delete_excluded=False, keep_perms=False, keep_executable=True):
		cmd = ["rsync", "-lrt", "--delete"]
		if keep_perms:
			cmd.append("-p")
		elif keep_executable:
			cmd.append("-E")

		if exclude is not None:
			if isinstance(exclude, (str, bytes)):
				cmd.append("--exclude={}".format(exclude))
			else:
				cmd.extend("--exclude={}".format(x) for x in exclude)
			if delete_excluded:
				cmd.append("--delete-excluded")

		return cmd

	def rsync_path(self, path):
		return "{}:{}".format(self.host(), self.get_path(path))

	def rsync_paths(self, paths):
		result = [self.rsync_path(paths[0])]
		for p in paths[1:]:
			result.append(":" + self.get_path(p))
		return result

	def rsync_get(self, remote_src, local_dst, *, check=True, **kwargs):
		cmd = self.rsync_cmd(**kwargs)
		if isinstance(remote_src, (str, bytes)):
			cmd.append(self.rsync_path(remote_src))
		else:
			cmd.extend(self.rsync_paths(remote_src))
		cmd.append(local_dst)

		return util.run(cmd, check=check)

	def rsync_put(self, local_src, remote_dst, *, check=True, **kwargs):
		cmd = self.rsync_cmd(**kwargs)
		if isinstance(local_src, (str, bytes)):
			cmd.append(local_src)
		else:
			cmd.extend(local_src)
		cmd.append(self.rsync_path(remote_dst))

		return util.run(cmd, check=check)

	def get_output(self, cmd, *, sudo=False, passwd=None, check=True):
		if sudo:
			result = self.run_sudo(cmd, passwd=passwd, check=check, universal_newlines=True)
		else:
			result = self.run(cmd, check=check, universal_newlines=True)

		return result.stdout if result.returncode == 0 else None

	def abs_path(self, path):
		return self.get_output(["readlink", "-f", path]).strip()

	def get_ncpus(self):
		if self.ncpus is None:
			self.ncpus = int(self.get_output(["nproc"]).strip())
		return self.ncpus

	def get_memory(self):
		if self.memory is None:
			s = self.get_output(["free", "-b"])
			self.memory = int(re.search(r"Mem:\s+(\d+)", s).group(1))
		return self.memory

	def get_host_addr(self, host, use_internal_addr=True):
		if not use_internal_addr or (host.addr == host.internal_addr):
			return host.addr
		cmd = ["ping", "-c", "1", "-W", "1", "-q", host.internal_addr]
		return host.internal_addr if self.run(cmd).returncode == 0 else host.addr

	#NOTE: must be absolute
	storage_root = "/mnt/nfs"

	def storage_setup(self, storage_addr, persist=True, *, passwd=None):
		cmd = ["mount", storage_addr, self.storage_root]
		self.run_sudo(cmd, passwd=passwd, check=True, output=self.log_path("storage_setup"))

		if persist:
			path = "/etc/fstab"
			line = "{} {} nfs defaults,nofail 0 0".format(storage_addr, self.storage_root)

			cmd = ["grep", "-F", "-m", "1", "-q", line, path]
			if self.run(cmd).returncode != 0:
				cmd = ["bash", "-c", "echo {} >> {}".format(line, path)]
				self.run_sudo(cmd, passwd=passwd, check=True, output=self.log_path("storage_setup"), append=True)

	def storage_path(self, path):
		return os.path.join(self.storage_root, path)

	#NOTE: when storing a directory, both src_path and dst_path must end with /
	def store(self, src_path, dst_path, *, delete_orig=False, keep_perms=False, check=True):
		if self.use_storage:
			src = src_path + "." if src_path[-1] == '/' else src_path
			dst = self.storage_path(dst_path)

			cmd = ["mkdir", "-p", os.path.dirname(dst), "&&", "cp", "-a"]
			if not keep_perms:
				cmd.append("--no-preserve=mode")
			cmd.extend((src, dst))
			if delete_orig:
				cmd.extend(("&&", "rm", "-rf", src_path))

			self.run(cmd, check=check)

		else:
			dst = self.local_output_path(dst_path)
			os.makedirs(os.path.dirname(dst), exist_ok=True)
			self.rsync_get(src_path, dst, check=check)
			if delete_orig:
				self.run(["rm", "-rf", src_path])

	def store_rename(self, src_path, dst_path):
		if self.use_storage:
			src = self.storage_path(src_path)
			dst = self.storage_path(dst_path)
			self.run(["mkdir", "-p", os.path.dirname(dst), "&&", "rm", "-rf", dst, "&&", "mv", src, dst], check=True)
		else:
			src = self.local_output_path(src_path)
			dst = self.local_output_path(dst_path)
			os.makedirs(os.path.dirname(dst), exist_ok=True)
			if os.path.isdir(dst):
				shutil.rmtree(dst)
			os.rename(src, dst)

	def cleanup_process(self, name):
		cmd = ["start-stop-daemon", "-K", "-R", "1", "-oq", "-n", name]
		self.run(cmd, check=True)

	# Returns RTT in microseconds
	def get_latency(self, host, *, use_internal_addr=False, count=1000, interval=0.001, passwd=None):
		addr = host.internal_addr if use_internal_addr else host.addr
		cmd = ["ping", "-c", str(count), "-i", str(interval), "-q", addr]
		s = self.get_output(cmd, sudo=(interval < 0.2), passwd=passwd)
		return float(s.splitlines()[-1].split()[3].split("/")[1]) * 1000.0

	# Returns bandwidth in Gbit/s
	def get_tcp_bandwidth(self, host, *, use_internal_addr=False):
		addr = host.internal_addr if use_internal_addr else host.addr
		proc = host.start(["iperf3", "-s"])
		try:
			s = self.get_output(["iperf3", "-c", addr, "-f", "g"])
			return float(s.splitlines()[-3].split()[6])
		finally:
			proc.stop(signal.SIGINT, timeout=1.0, kill_remote_on_timeout=True)

	def set_net_delay(self, dev, delay_us, *, passwd=None):
		cmd = ["tc", "qdisc", "add", "dev", dev, "root", "netem", "delay", "{}us".format(delay_us)]
		self.run_sudo(cmd, passwd=passwd, check=True)

	def reset_net_delay(self, dev, *, check=False, passwd=None):
		cmd = ["tc", "qdisc", "del", "dev", dev, "root", "netem"]
		self.run_sudo(cmd, passwd=passwd, check=check)


# Each line must have the following format: [user@]addr[|internal_addr][:directory]
# Empty lines and comments (starting with #) are ignored
#
# user           specifies the remote ssh user name; default: local user name
# addr           specifies the remote host name or IP address;
# internal_addr  specifies an optional alternative host name or IP address that
#                the remote host can be accessed by from other worker nodes but
#                not the control node;
# directory      working directory for remote commands and file storage;
#                default: remote user home directory
#
# Returns a list of (addr, user or None, internal_addr or None, directory or None) tuples
def load_hosts(path):
	result = []
	regex = re.compile(r"((\S+?)@)?([^\s\|:]+)(\|([^\s:]+))?(:(\S+))?")

	with open(path, "r") as f:
		for line in f:
			s = line[:line.find('#')].strip()
			if s:
				m = regex.match(s)
				result.append((m.group(3), m.group(2), m.group(5), m.group(7)))

	return result


class RemoteCluster:
	def __init__(self, hosts):
		self.hosts = hosts

	def for_each(self, fn, *args, parallel=False, **kwargs):
		return (util.parallelize(fn, self.hosts, *args, **kwargs) if parallel
		        else [fn(h, *args, **kwargs) for h in self.hosts])

	def ssh_setup(self, *, passwd=None):
		self.for_each(RemoteHost.ssh_setup, passwd=passwd, parallel=passwd is not None)

	def check_sudo_passwd(self, passwd):
		self.for_each(RemoteHost.check_sudo_passwd, passwd, parallel=True)

	def storage_setup(self, storage_addr, *, passwd=None):
		self.for_each(RemoteHost.storage_setup, storage_addr, passwd=passwd, parallel=passwd is not None)


class ServerInstance:
	def __init__(self, host, name, benchmark, config_name, instance_id, *, start_log_line=None,
	             error_log_line=None, stop_signal=None, reserve_cpus=True, collect_stats=False):
		self.host = host
		self.name = name
		self.benchmark = benchmark
		self.config_name = config_name
		self.instance_id = instance_id
		self.start_log_line = start_log_line
		self.error_log_line = error_log_line
		self.stop_signal = stop_signal if stop_signal is not None else signal.SIGINT
		self.reserve_cpus = reserve_cpus
		self.collect_stats = collect_stats
		self.experiment = None
		self.run_id = None
		self.attempt_id = None
		self.remote_proc = None
		self.stats_proc = None
		self.start_time = None

	def get_name(self):
		return "{}_{}".format(self.name, self.instance_id)

	def output_dir(self):
		return os.path.join(self.benchmark, self.get_name())

	def output_path(self, path):
		return os.path.join(self.output_dir(), path)

	def log_path(self):
		return self.output_path(self.name + ".log")

	def is_running(self):
		return self.remote_proc.is_running()

	def wait_until_started(self, timeout=None):
		cmd = []
		if timeout is not None:
			cmd.extend(("timeout", str(timeout)))
		cmd.extend(("tail", "-F", "-n", "+1", self.log_path(), "|", "grep", "-F", "-m", "1", "-q", self.start_log_line))

		result = self.host.run(cmd)
		if result.returncode != 0:
			raise Exception("Failed to start {} instance {} on {}:\nstdout: {}\nstderr: {}".format(
			                self.name, self.instance_id, self.host.addr, result.stdout, result.stderr))

		if self.error_log_line is not None:
			cmd = ["grep", "-F", "-m", "1", "-q", self.error_log_line, self.log_path()]
			if self.host.run(cmd).returncode == 0:
				raise Exception("{} instance {} on {} started with errors".format(
				                self.name, self.instance_id, self.host.addr))

	def kill(self, *, sig=None, check=False, timeout=None, attempts=None, kill_remote_on_timeout=False):
		if self.remote_proc is not None:
			self.remote_proc.stop(sig if sig is not None else self.stop_signal, check=check, timeout=timeout,
			                      attempts=attempts, kill_remote_on_timeout=kill_remote_on_timeout)
			self.remote_proc = None

	def cleanup(self):
		try:
			self.kill(timeout=5.0, attempts=1)
		except subprocess.TimeoutExpired:
			self.kill(sig=signal.SIGKILL)

		self.host.run(["rm", "-rf", self.output_dir()])

	def init(self, experiment, run_id, attempt_id):
		self.experiment = experiment
		self.run_id = run_id
		self.attempt_id = attempt_id

		self.cleanup()
		self.host.run(["mkdir", "-p", self.output_dir()], check=True)

	def get_remote_pid(self):
		return self.remote_proc.remote_pid

	def start_stats_proc(self):
		#NOTE: 0.5s update period to match docker stats; 1st update at ~0.5s
		cmd = ["top", "-b", "-p", str(self.get_remote_pid()), "-d", "0.5", "|", "sed", "-n", "-u", "7p;8~9p"]
		self.stats_proc = self.host.start(
			cmd, remote_output=self.output_path("stats.log")
		)

	def stop_stats_proc(self):
		if self.stats_proc is not None:
			self.stats_proc.stop(signal.SIGINT)
			self.stats_proc = None

	# Returns False if the exception needs to be re-raised
	@staticmethod
	def handle_exception(exc, raise_on_failure):
		if (exc is KeyboardInterrupt) or raise_on_failure:
			return False
		util.print_exception()
		return True

	def start(self, cmd, experiment, run_id, attempt_id, *, env=None, timeout=None, raise_on_failure=True):
		self.init(experiment, run_id, attempt_id)
		print("Starting {} instance {} on {}...".format(self.name, self.instance_id, self.host.addr))

		self.remote_proc = self.host.start(cmd, env=env, remote_output=self.log_path())
		if self.collect_stats:
			self.start_stats_proc()

		if self.start_log_line is not None:
			try:
				t0 = time.monotonic()
				self.wait_until_started(timeout)
				t1 = time.monotonic()
				self.start_time = t1 - t0

				print("Started {} instance {} on {} in {:.2f} seconds".format(
				      self.name, self.instance_id, self.host.addr, self.start_time))
				return None

			except Exception as e:
				if not self.handle_exception(e, raise_on_failure):
					raise
				return e

		else:
			print("Started {} instance {} on {}".format(self.name, self.instance_id, self.host.addr))
			return None

	@staticmethod
	def result_dir(benchmark, config_name, experiment, run_id, attempt_id, run_success, prefix=None):
		prefix = (prefix + "_") if prefix else ""
		if run_success:
			run_name = "{}run_{}".format(prefix, run_id)
		else:
			run_name = "failed_{}run_{}_attempt_{}".format(prefix, run_id, attempt_id)
		return os.path.join(benchmark, config_name, experiment, run_name)

	def result_path(self, success, prefix=None, invocation_attempt=None):
		return os.path.join(
			self.result_dir(self.benchmark, self.config_name, self.experiment,
			                self.run_id, self.attempt_id, True, prefix),
			("" if success else "failed_") + self.get_name() +
			("" if success or (invocation_attempt is None) else "_attempt_{}".format(invocation_attempt))
		)

	def store_output(self, success, prefix=None, invocation_attempt=None):
		self.host.store(self.output_dir() + "/", self.result_path(success, prefix, invocation_attempt) + "/",
		                delete_orig=True, check=True)

	@staticmethod
	def rename_failed_run(host, benchmark, config_name, experiment, run_id, attempt_id, prefix=None):
		args = (benchmark, config_name, experiment, run_id, attempt_id)
		src = ServerInstance.result_dir(*args, True, prefix)
		dst = ServerInstance.result_dir(*args, False, prefix)
		host.store_rename(src, dst)

	def wait(self, *, store=True, timeout=None, raise_on_failure=True,
	         kill_remote_on_timeout=False, prefix=None, invocation_attempt=None):
		print("Waiting for {} instance {} on {}...".format(self.name, self.instance_id, self.host.addr))

		try:
			t0 = time.monotonic()
			self.remote_proc.wait(check=True, timeout=timeout, kill_remote_on_timeout=kill_remote_on_timeout)
			t1 = time.monotonic()

			print("Finished {} instance {} on {} in {:.2f} seconds".format(
			      self.name, self.instance_id, self.host.addr, t1 - t0))
			exc = None

		except Exception as e:
			if not self.handle_exception(e, raise_on_failure):
				raise
			exc = e

		finally:
			self.stop_stats_proc()

		if store:
			self.store_output(exc is None, prefix, invocation_attempt)
		return exc

	def stop(self, *, store=True, timeout=None, attempts=None, raise_on_failure=True,
	         kill_remote_on_timeout=False, prefix=None, invocation_attempt=None):
		self.start_time = None
		print("Stopping {} instance {} on {}...".format(self.name, self.instance_id, self.host.addr))

		try:
			t0 = time.monotonic()
			self.kill(check=True, timeout=timeout, attempts=attempts, kill_remote_on_timeout=kill_remote_on_timeout)
			t1 = time.monotonic()

			print("Stopped {} instance {} on {} in {:.2f} seconds".format(
			      self.name, self.instance_id, self.host.addr, t1 - t0))
			exc = None

		except Exception as e:
			if not self.handle_exception(e, raise_on_failure):
				raise
			exc = e

		finally:
			self.stop_stats_proc()

		if store:
			self.store_output(exc is None, prefix, invocation_attempt)
		return exc

	def run(self, cmd, experiment, run_id, attempt_id, *, env=None, store=True,
	        timeout=None, raise_on_failure=True, prefix=None, invocation_attempt=None):
		self.init(experiment, run_id, attempt_id)
		print("Running {} instance {} on {}...".format(self.name, self.instance_id, self.host.addr))

		try:
			self.host.run(cmd, env=env, remote_output=self.log_path(), timeout=timeout, check=True)
			print("Finished {} instance {} on {}".format(self.name, self.instance_id, self.host.addr))
			exc = None

		except Exception as e:
			if not self.handle_exception(e, raise_on_failure):
				raise
			exc = e

		if store:
			self.store_output(exc is None, prefix, invocation_attempt)
		return exc

	def store_crash_files(self, src_path, *, store_patterns, keep_patterns):
		if store_patterns:
			cmd = []
			for p in store_patterns:
				cmd.extend(("mv", os.path.join(src_path, p), self.output_dir(), ";"))
			self.host.run(cmd, globs=True)

		if keep_patterns:
			dst_path = os.path.join(self.benchmark, "crash_{}_{}_run_{}_attempt_{}_{}_{}".format(
			                        self.config_name, self.experiment, self.run_id,
			                        self.attempt_id, self.name, self.instance_id))

			cmd = ["mkdir", "-p", dst_path, "&&", "("]
			for p in keep_patterns:
				cmd.extend(("mv", os.path.join(src_path, p), dst_path, ";"))
			cmd.extend((")", ";", "rmdir", dst_path)) # remove directory if empty
			self.host.run(cmd, globs=True)
