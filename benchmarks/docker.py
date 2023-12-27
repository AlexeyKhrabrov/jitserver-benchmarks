import os.path
import signal

import remote
import util


class DockerHost(remote.RemoteHost):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.n_reserved_cpus = 0

	def clean_images(self):
		self.run(["docker", "system", "prune", "-a", "-f", "--volumes"], check=True)

	def prune_images(self):
		self.run(["docker", "image", "prune", "-f"], check=True)

	def remove_container(self, name, *, force=False):
		cmd = ["docker", "rm"]
		if force:
			cmd.append("-f")
		cmd.append(name)
		self.run(cmd)

	@staticmethod
	def container_path(name, path):
		return "{}:{}".format(name, path)

	def cp_to_container(self, name, host_src, container_dst, *, check=False):
		cmd = ["docker", "cp", host_src, self.container_path(name, container_dst)]
		return self.run(cmd, check=check)

	def cp_from_container(self, name, container_src, host_dst, *,
	                      mkdir_dst=False, check=False):
		cmd = []
		if mkdir_dst:
			cmd.extend(("mkdir", "-p", host_dst, "&&"))
		cmd.extend(("docker", "cp", self.container_path(name, container_src),
		           host_dst))
		return self.run(cmd, check=check)

	def get_host_pid(self, container_name, proc_name=None, *,
	                 cmd_filter=None, check=True):
		if proc_name is None:
			cmd = ["docker", "inspect", "-f", "{{.State.Pid}}", container_name]
			s = self.get_output(cmd, check=check)
			return int(s.strip()) if s is not None else None

		cmd = ["docker", "top", container_name, "-C", proc_name, "-o", "pid,cmd"]
		s = self.get_output(cmd, check=check)
		if s is None:
			return None

		pids = []
		for line in s.splitlines()[1:]:
			pid, cmd = line.split(maxsplit=1)
			if cmd_filter is None or cmd_filter(cmd):
				pids.append(int(pid))
		if len(pids) == 1:
			return pids[0]

		if check:
			raise Exception("No unique {} process in container {} on {}".format(
			                proc_name, container_name, self.addr))
		return None

	def reserve_cpus(self, num_cpus):
		ncpus = self.get_ncpus()
		if self.n_reserved_cpus + num_cpus > ncpus:
			raise Exception("Cannot reserve {} CPUs on {}: {}/{} reserved".format(
			                num_cpus, self.addr, self.n_reserved_cpus, ncpus))

		self.n_reserved_cpus += num_cpus
		return list(range(self.n_reserved_cpus - num_cpus, self.n_reserved_cpus))

	def reset_reserved_cpus(self):
		self.n_reserved_cpus = 0


class DockerConfig:
	def __init__(self, *, ncpus=None, memory=None, pin_cpus=False, network=None):
		self.ncpus = ncpus
		self.memory = memory
		self.pin_cpus = pin_cpus
		self.network = network or "host"

	def docker_args(self, host, reserved_cpus=None):
		args = ["--network={}".format(self.network)]

		if self.ncpus is not None:
			if reserved_cpus is not None:
				args.append("--cpuset-cpus={}".format(",".join(
				            str(c) for c in reserved_cpus)))
			else:
				args.append("--cpus={}".format(self.ncpus))

		if self.memory is not None:
			args.append("--memory={}".format(self.memory))

		return args


class ContainerInstance(remote.ServerInstance):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.docker_stats_proc = None

	def cleanup(self):
		self.host.remove_container(self.get_name(), force=True)
		super().cleanup()

	def get_remote_pid(self):
		return self.host.get_host_pid(self.get_name())

	def start_stats_proc(self):
		wait_cmd = ["scripts/wait_container.sh", self.get_name(), "0.1", "50"]
		if self.host.run(wait_cmd).returncode == 0:
			#NOTE: 0.5s update period; 1st update at ~2s
			cmd = ["docker", "stats", "--format={{.CPUPerc}} {{.MemUsage}}",
			       self.get_name(), "|", "sed", "-r", "-u", "s/^.{7}//"]
			self.docker_stats_proc = self.host.start(
				cmd, remote_output=self.output_path("docker_stats.log")
			)

		super().start_stats_proc()

	def stop_stats_proc(self):
		if self.docker_stats_proc is not None:
			self.docker_stats_proc.stop(signal.SIGINT)
			self.docker_stats_proc = None

		super().stop_stats_proc()

	@staticmethod
	def cgroup_path(cid, cgroup, file):
		return os.path.join("/sys/fs/cgroup", cgroup, "docker", cid, file)

	def stop(self, *args, **kwargs):
		cmd = ["docker", "ps", "-q", "--no-trunc", "-f",
		       "name=^{}$".format(self.get_name())]
		cid = (self.host.get_output(cmd, check=False) or "").strip()

		if cid:
			files = [
				self.cgroup_path(cid, "cpuacct", "cpuacct.stat"),
				self.cgroup_path(cid, "memory", "memory.max_usage_in_bytes"),
			]
			self.host.run(["cat"] + files,
			              remote_output=self.output_path("cgroup_rusage.log"))

		return super().stop(*args, **kwargs)

	def get_reserved_cpus(self, config):
		if self.reserve_cpus and config.pin_cpus and isinstance(config.ncpus, int):
			cpus = self.host.reserve_cpus(config.ncpus)
			if util.verbose:
				print("{} instance {} reserved cpus on {}: {}".format(self.name, self.instance_id, self.host.addr, cpus))
			return cpus
		else:
			return None
