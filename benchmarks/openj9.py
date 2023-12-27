import os.path
import time

import docker
import remote

module_dir = os.path.dirname(__file__)


class OpenJ9Host(remote.RemoteHost):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def update_scripts(self):
		self.rsync_put(os.path.join(module_dir, "../scripts/"), "scripts/",
		               exclude="/vlog_*", delete_excluded=True)

	def openj9_prereqs(self, *, passwd=None):
		self.update_scripts()
		self.run_sudo(["scripts/openj9_prereqs.sh"], passwd=passwd,
		              output=self.log_path("openj9_prereqs"), check=True)

	def openj9_setup(self, jdk_dir, jdk_ver, *, configure=False, debug=False):
		exclude = ("/openj9-openjdk-jdk*/build/")
		self.rsync_put(jdk_dir + "/", "jdk/", exclude=exclude)
		self.update_scripts()

		cmd = ["scripts/openj9_build.sh", "jdk", str(jdk_ver)]
		if configure:
			cmd.append("--configure")
		if debug:
			cmd.append("--debug")

		t0 = time.monotonic()
		self.run(cmd, output=self.log_path("openj9_setup"), check=True)
		t1 = time.monotonic()
		print("OpenJ9 setup on {} took {:.2f} seconds".format(self.addr, t1 - t0))

	def jdk_path(self, jdk_ver, debug=False):
		return "jdk/{}-openjdk-jdk{}/build/{}/images/jdk".format(
			"openj9", jdk_ver, "slowdebug" if debug else "release"
		)


class OpenJ9Cluster(remote.RemoteCluster):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def openj9_prereqs(self, *, passwd=None):
		self.for_each(OpenJ9Host.openj9_prereqs, passwd=passwd,
		              parallel=passwd is not None)

	def openj9_setup(self, jdk_dir, jdk_ver, *, configure=False, debug=False):
		self.for_each(OpenJ9Host.openj9_setup, jdk_dir, jdk_ver,
		              configure=configure, debug=debug, parallel=True)


class JVMConfig:
	def __init__(self, *, heap_size=None, virtualized=False,
	             scc_size=None, nojit=False):
		self.heap_size = heap_size
		self.virtualized = virtualized
		self.scc_size = scc_size
		self.nojit = nojit

	def jvm_args(self):
		args = []

		if self.heap_size is not None:
			args.extend(("-Xms{}".format(self.heap_size),
			             "-Xmx{}".format(self.heap_size)))
		if self.virtualized:
			args.append("-Xtune:virtualized")
		if self.scc_size is not None:
			args.append("-Xscmx{}".format(self.scc_size))
		if self.nojit:
			args.append("-Xint")

		return args


crash_store_patterns = ("javacore.*.txt", "jitdump.*.dmp")
crash_keep_patterns = ("core", "core.*", "Snap.*.trc")

class OpenJ9ContainerInstance(docker.ContainerInstance):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def store_output(self, success, prefix=None, invocation_attempt=None, *,
	                 vlog=None):
		cmd = []
		if vlog is not None:
			cmd.extend(("mv", self.output_path(os.path.join("vlogs", vlog + "*")),
			            self.output_path(vlog + ".log"), "&&"))
		cmd.extend(("rm", "-rf", self.output_path("vlogs")))
		self.host.run(cmd, globs=True)

		super().store_output(success, prefix, invocation_attempt)

	def store_openj9_crash_files(self):
		tmp_path = "_crash_"

		result = self.host.cp_from_container(self.get_name(), "/output/.",
		                                     tmp_path + "/", mkdir_dst=True)
		if result.returncode == 0:
			self.store_crash_files(tmp_path, store_patterns=crash_store_patterns,
			                       keep_patterns=crash_keep_patterns)

		self.host.run(["rm", "-rf", tmp_path])

	def stop(self, *args, **kwargs):
		exc = super().stop(*args, **kwargs)
		self.host.remove_container(self.get_name())
		return exc
