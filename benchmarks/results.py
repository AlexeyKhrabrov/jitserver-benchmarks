import datetime
import csv
import itertools
import math
import os
import os.path
import re

import matplotlib
matplotlib.use("agg")
import matplotlib.lines
import matplotlib.pyplot as plt
import matplotlib.ticker

import numpy as np
import pandas as pd

plt.rcParams.update({
	"figure.constrained_layout.use": True,
	"figure.figsize": (3.0, 2.25),
	"font.size": 9,
	"hatch.linewidth": 0.5,
	"legend.fontsize": 8,
	"legend.framealpha": 0.5,
	"lines.linewidth": 1.0,
	"lines.markersize": 4.0,
	"savefig.dpi": 300,
})

from jitserver import Experiment
import remote
import util

#TODO: factor out common code


def logs_path(benchmark, config, experiment=None, instance_id=None, run_id=None, component=None):
	return os.path.join(
		remote.RemoteHost.logs_dir, benchmark, config.name,
		experiment.name.lower() if experiment is not None else "",
		"run_{}".format(run_id) if run_id is not None else "",
		"{}_{}".format(component, instance_id) if component is not None else ""
	)

results_dir = "results"

def results_path(benchmark, config=None, experiment=None, instance_id=None, run_id=None, component=None):
	return os.path.join(
		results_dir, benchmark or "", config.name if config is not None else "",
		experiment.name.lower() if experiment is not None else "",
		"run_{}".format(run_id) if run_id is not None else "",
		"{}_{}".format(component, instance_id) if component is not None else ""
	)

def save_summary(s, *args, name=None):
	path = os.path.join(results_path(*args), (name or "summary") + ".txt")
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w") as f:
		print(s, end="", file=f)

plot_format = "png"

def save_plot(ax, name, *args, size=None):
	path = os.path.join(results_path(*args), "{}.{}".format(name, plot_format))
	os.makedirs(os.path.dirname(path), exist_ok=True)

	fig = ax.get_figure()
	if size is not None:
		fig.set_size_inches(*size)

	fig.savefig(path, backend="pgf" if plot_format == "pdf" else None)
	plt.close(fig)


class ProcessStats:
	@staticmethod
	def kind(): return "process"

	@staticmethod
	def parse(line):
		# PID USER PR NI VIRT RES SHR S %CPU %MEM TIME+ COMMAND
		s = line.split(maxsplit=9)
		return (float(s[8]), util.size_to_bytes(s[5], 'k', float) / (1024 * 1024))

	def __init__(self, *args, file_name="stats.log", period=0.5, start=1, skip_lines=1):
		self.id = args

		self.period = period
		self.cpu_data = [0.0] * start
		self.mem_data = [0.0] * start

		with open(os.path.join(logs_path(*args), file_name), "r") as f:
			for i in range(skip_lines):
				next(f)

			for line in f:
				try:
					cpu, mem = self.parse(line)
					self.cpu_data.append(cpu)
					self.mem_data.append(mem)
				except:
					pass

		self.peak_cpu_p = max(self.cpu_data)

	def cpu_df(self):
		idx = [self.period * i for i in range(len(self.cpu_data))]
		return pd.DataFrame(self.cpu_data, index=idx, columns=["cpu"])

	def mem_df(self):
		idx = [self.period * i for i in range(len(self.mem_data))]
		return pd.DataFrame(self.mem_data, index=idx, columns=["mem"])

	def save_plot(self, ax, name, label):
		ax.set(xlabel="Time, sec", ylabel=label)
		save_plot(ax, "{}_{}".format(name, self.kind()), *self.id, size=(4.5, 2.25))

	def save_cpu_plot(self):
		self.save_plot(self.cpu_df().plot(legend=False, xlim=(0, None), ylim=(0, None)), "cpu", "CPU usage, %")

	def save_mem_plot(self):
		self.save_plot(self.mem_df().plot(legend=False, xlim=(0, None), ylim=(0, None)), "mem", "Memory usage, MB")

	def save_plots(self):
		self.save_cpu_plot()
		self.save_mem_plot()


class ContainerStats(ProcessStats):
	@staticmethod
	def kind(): return "container"

	@staticmethod
	def parse(line):
		# 0.0% 0.0MiB / 0GiB
		s = line.split(maxsplit=2)
		return (float(s[0][:-1]), util.size_to_bytes(s[1][:-2], 'b', float) / (1024 * 1024))

	def __init__(self, *args):
		super().__init__(*args, file_name="docker_stats.log", period=0.5, start=3, skip_lines=0)


def parse_first_token(val, line, prefix, parse_fn, use_last=False):
	if use_last or (val is None):
		idx = line.find(prefix)
		if idx >= 0:
			return parse_fn(line[idx + len(prefix):].split(maxsplit=1)[0]), True
	return val, False


class ProcessRUsage:
	def is_valid(self):
		return None not in (self.user_time, self.kernel_time, self.peak_mem)

	def __init__(self, file_name, *args, use_last=True):
		self.user_time = None
		self.kernel_time = None
		self.peak_mem = None

		path = os.path.join(logs_path(*args), file_name)
		with open(path, "r") as f:
			for line in f:
				self.user_time, parsed = parse_first_token(self.user_time, line, "User time (seconds):",
				                                           float, use_last)
				if parsed: continue

				self.kernel_time, parsed = parse_first_token(self.kernel_time, line, "System time (seconds):",
				                                             float, use_last)
				if parsed: continue

				self.peak_mem, parsed = parse_first_token(self.peak_mem, line, "Maximum resident set size (kbytes):",
				                                          lambda s: float(s) / 1024, use_last)

		if not self.is_valid():
			raise Exception("Failed to parse rusage in {}".format(path))

	def cpu_time(self):
		return self.user_time + self.kernel_time


class ContainerRUsage(ProcessRUsage):
	def __init__(self, *args):
		path = os.path.join(logs_path(*args), "cgroup_rusage.log")
		with open(path, "r") as f:
			lines = f.readlines()
			self.user_time = float(lines[0].split(maxsplit=1)[1]) * 0.01
			self.kernel_time = float(lines[1].split(maxsplit=1)[1]) * 0.01
			self.peak_mem = float(lines[2].strip()) / (1024 * 1024)


class VLog:
	def parse(self, line, prefix, suffix, first=False):
		self.start_idx = line.index(prefix, 0 if first else self.end_idx) + len(prefix)
		self.end_idx = line.index(suffix, self.start_idx)
		result = line[self.start_idx:self.end_idx]
		self.end_idx += len(suffix)
		return result

	def parse_method(self, line, prefix=" "):
		optlevel = self.parse(line, " (", ")", True)
		return "{} @ {}".format(self.parse(line, prefix, " "), optlevel)

	def __init__(self, file_name, *args, find_dups=False, dups_dlt=False, dups_mht=False):
		self.path = os.path.join(logs_path(*args), file_name)

		self.comp_starts = [] # milliseconds
		self.queue_sizes = []
		self.comp_times = [] # milliseconds
		self.queue_times = [] # milliseconds

		self.n_lambdas = 0
		self.dups = None
		if find_dups:
			methods = set()
			self.dups = set()

		with open(self.path, "r") as f:
			for line in f:
				if line.startswith(" ("):
					method = self.parse_method(line, " Compiling ")
					self.comp_starts.append(int(self.parse(line, " t=", " ")))

				elif line.startswith("+ ("):
					method = self.parse_method(line)
					self.queue_sizes.append(int(self.parse(line, " Q_SZ=", " ")))
					method += " " + self.parse(line, "j9m=", " ")
					self.comp_times.append(float(self.parse(line, " time=", "us")) / 1000.0)
					self.queue_times.append(float(self.parse(line, " queueTime=", "us")) / 1000.0)

					self.n_lambdas += 1 if "$$Lambda$" in method else 0
					if (find_dups and (dups_dlt or " MethodInProgress " not in line) and
					    (dups_mht or "_thunkArchetype_" not in method)
					):
						if method in methods:
							self.dups.add(method)
						else:
							methods.add(method)

	def dups_summary(self):
		if not self.dups:
			return ""

		result = self.path + ":\n"
		for method in self.dups:
			result += "\t{}\n".format(method)
		return result + "\n"


class ApplicationOutput:
	@staticmethod
	def parse_log_ts(s):
		return datetime.datetime.strptime(s[:-3], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc)

	def __init__(self, bench, *args):
		self.benchmark = bench.name()
		self.id = [self.benchmark] + list(args) + [self.benchmark]
		path = logs_path(*self.id)

		self.jvm_ts = None
		self.docker_ts = None
		self.start_ts = None
		self.stop_ts = None
		self.n_comps = None
		self.bytes_recv = None
		self.jit_time = 0.0 # seconds

		with open(os.path.join(path, bench.name() + ".log"), "r") as f:
			for line in f:
				self.jvm_ts, parsed = parse_first_token(self.jvm_ts, line, "JVM start timestamp: ", self.parse_log_ts)
				if parsed: continue

				self.docker_ts, parsed = parse_first_token(self.docker_ts, line,
				                                           "Docker start timestamp: ", self.parse_log_ts)
				if parsed: continue

				self.n_comps, parsed = parse_first_token(self.n_comps, line, "compilationOK", int)
				if parsed: continue

				self.bytes_recv, parsed = parse_first_token(self.bytes_recv, line,
				                                            "Total amount of data received: ", int)
				if parsed: continue

				t, parsed = parse_first_token(None, line, "Time spent in compilation thread =", float)
				self.jit_time += (t or 0.0) / 1000.0
				if parsed: continue

				t, parsed = parse_first_token(None, line, "Time spent in AOT prefetcher thread: ", float)
				self.jit_time += (t or 0.0) / 1000.0

		self.bytes_recv = self.bytes_recv or 0

		start_log_line = bench.start_log_line()
		stop_log_line = bench.stop_log_line()
		with open(os.path.join(path, bench.start_stop_ts_file()), "r") as f:
			for line in f:
				if start_log_line in line:
					self.start_ts = bench.parse_start_stop_ts(line)
					continue

				if stop_log_line in line:
					self.stop_ts = bench.parse_start_stop_ts(line)
					break

	def jvm_start_time(self):
		return (self.start_ts - self.jvm_ts).total_seconds()

	def docker_start_time(self):
		return (self.start_ts - self.docker_ts).total_seconds()

	def process_stats(self):
		return ProcessStats(*self.id)

	def container_stats(self):
		return ContainerStats(*self.id)

	def process_rusage(self):
		return ProcessRUsage(self.benchmark + ".log", *self.id)

	def container_rusage(self):
		try:
			return ContainerRUsage(*self.id)
		except:
			return None

	def vlog(self):
		return VLog("vlog_client.log", *self.id, find_dups=True)


class WarmupData:
	def is_within_margin(self, val, center):
		return (val >= center * (1 - self.margin)) and (val <= center * (1 + self.margin))

	def is_plateau(self, data, idx):
		outliers = 0

		for i in range(idx + 1, len(data)):
			if not self.is_within_margin(data[i], data[idx]):
				outliers += 1
				if outliers > math.ceil((i - idx) * self.outlier_limit):
					return False

		return idx + self.min_plateau < len(data)

	def reached_threshold(self, data, idx):
		return ((data[idx] >= self.threshold * self.peak_throughput) and
		        ((idx + 1 >= len(data)) or (data[idx + 1] >= self.next_threshold * self.peak_throughput)))

	def __init__(self, throughput_data, duration=None, *, keep_throughput_data=True,
	             threshold=None, next_threshold=None, margin=None, outlier_limit=None,
	             min_plateau=None, window=None, exclude_outliers=False):
		self.throughput_data = throughput_data
		self.threshold = threshold or 0.9
		self.next_threshold = next_threshold or 0.8
		self.margin = margin or 0.1
		self.outlier_limit = outlier_limit or 0.1
		self.min_plateau = max(1, min_plateau or 3)

		data = [d[1] for d in self.throughput_data]
		if window is not None:
			data = pd.DataFrame(data).rolling(window, 1, True).mean()[0].to_list()

		self.plateau_start = next((i for i in range(len(data)) if self.is_plateau(data, i)), len(data) - 1)
		self.peak_throughput = mean(d for d in data[self.plateau_start:]
		                            if not exclude_outliers or self.is_within_margin(d, data[self.plateau_start]))

		self.warmup_end = next((i for i in range(len(data)) if self.reached_threshold(data, i)), len(data) - 1)
		self.warmup_time = self.throughput_data[self.warmup_end][0] or self.throughput_data[-1][0]

		if not keep_throughput_data:
			self.throughput_data = None


class JMeterOutput:
	def __init__(self, benchmark, config, *args, tail=None, **kwargs):
		self.id = [benchmark, config] + list(args) + ["jmeter"]

		self.duration = config.jmeter_config.duration
		self.requests = 0

		throughput_data = [(0.0, 0.0)] # time (sec), throughput (req/sec)
		regex = re.compile(r"^summary \+\s+(\d+) in (\d+:\d+:\d+) =\s+(\d+\.\d+)/s")
		t = 0.0

		with open(os.path.join(logs_path(*self.id), "jmeter.log"), "r") as f:
			for line in f:
				m = regex.search(line)
				if m:
					hms = m.group(2).split(":")
					t += 60 * 60 * int(hms[0]) + 60 * int(hms[1]) + int(hms[2])
					if t <= self.duration:
						throughput_data.append((t, float(m.group(3))))
					self.requests += int(m.group(1))

		self.warmup_data = WarmupData(throughput_data, self.duration, **kwargs)

		if config.jmeter_config.latency_data:
			latency_data = [] # time (sec), latency (sec)
			t = 0.0

			with open(os.path.join(logs_path(*self.id), "results.jtl"), "r", newline="") as f:
				#TODO: use faster csv.reader
				reader = csv.DictReader(f)
				start_ts = int(next(reader)["timeStamp"])

				for row in reader:
					if row["label"] not in ("WS2 Open Connection", "WS2 existing"):
						t += (int(row["timeStamp"]) - start_ts) / 1000.0
						if (self.duration is not None) and (t > self.duration):
							break
						latency_data.append((t, int(row["elapsed"]) / 1000.0))

			latencies = [d[1] for d in latency_data]
			tail_percentile = tail or 99.0
			self.tail_latency = np.percentile(latencies, tail_percentile)

	def process_stats(self):
		return ProcessStats(*self.id)

	def container_stats(self):
		return ContainerStats(*self.id)

	def process_rusage(self):
		return ProcessRUsage("jmeter.log", *self.id)


class JITServerOutput:
	def __init__(self, *args):
		self.id = list(args) + ["jitserver"]

		self.bytes_recv = None

		path = os.path.join(logs_path(*self.id), "jitserver.log")
		with open(path, "r") as f:
			for line in f:
				self.bytes_recv, parsed = parse_first_token(self.bytes_recv, line,
				                                            "Total amount of data received: ", int)
				if parsed: break

		self.bytes_recv = self.bytes_recv or 0

	def process_stats(self):
		return ProcessStats(*self.id)

	def container_stats(self):
		try:
			return ContainerStats(*self.id)
		except:
			return None

	def process_rusage(self):
		return ProcessRUsage("jitserver.log", *self.id, use_last=False)

	def container_rusage(self):
		try:
			return ContainerRUsage(*self.id)
		except:
			return None

	def vlog(self):
		return VLog("vlog_server.log", *self.id)


class DBOutput:
	def __init__(self, bench, *args):
		self.id = [bench.name()] + list(args) + [bench.db_name()]

	def container_stats(self):
		return ContainerStats(*self.id)

	def container_rusage(self):
		return ContainerRUsage(*self.id)


# label, unit, detailed
field_descriptors = {
	"start_time":           ("Start time",           "sec",      False),
	"peak_mem":             ("Memory usage",         "MB",       False),
	"n_comps":              ("Compiled methods",     None,       True ),
	"cpu_time":             ("CPU time",             "sec",      True ),
	"jit_time":             ("JIT CPU time",         "sec",      True ),
	"warmup_time":          ("Warm-up time",         "min",      False),
	"warmup_end":           ("Warm-up iterations",   None,       True ),
	"full_warmup_time":     ("Full warm-up time",    "min",      False),
	"peak_throughput":      ("Peak throughput",      "req/sec",  False),
	"requests":             ("Requests served",      None,       True ),
	"jitserver_mem":        ("JITServer memory",     "GB",       False),
	"jitserver_cpu":        ("JITServer CPU",        "min",      False),
	"jitserver_peak_cpu_p": ("JITServer peak CPU",   "%",        False),
	"data_transferred":     ("Data transferred",     "MB",       True ),
	"peak_total_mem":       ("Memory usage",         "GB",       False),
	"cpu_time_per_req":     ("CPU cost",             "msec/req", False),
}

# field, label, log, cut
vlog_cdf_fields = (
	("comp_starts", "Compilation start time, ms", False, None),
	("queue_sizes", "Compilation queue size",     False, None),
	("comp_times",  "Compilation time, ms",       True,  0.99),
	("queue_times", "Total queuing time, ms",     True,  0.99),
)


def base_field(field):
	if field.startswith("overall_"):
		field = field[field.index("_") + 1:]
	if field.startswith("total_"):
		field = field[field.index("_") + 1:]

	if field.endswith("_normalized"):
		field = field[:field.rindex("_")]
	if field.endswith("_perclient"):
		field = field[:field.rindex("_")]

	return field

def field_label(field):
	desc = field_descriptors[base_field(field)]
	normalized = field.endswith("_normalized")
	per_client = "_perclient" in field

	return "".join((
		"Total " if field.startswith("total_") else "",
		"Overall " if field.startswith("overall_") else "",
		desc[0],
		" per JVM" if per_client else "",
		(", " + desc[1] if desc[1] is not None else "") if not normalized else "",
		" (norm.)" if normalized else ""
	))


def mean(data):
	data = [d for d in data if d is not None]
	return np.mean(data) if data else 0.0

def stdev(data):
	data = [d for d in data if d is not None]
	return np.std(data) if len(data) > 1 else 0.0

def rel_change(x, x_ref):
	return ((x - x_ref) / x_ref) if x_ref else 0.0

def rel_change_p(x, x_ref):
	return 100.0 * rel_change(x, x_ref)


max_experiment_name_len = max(len(e.name) for e in Experiment)

def experiment_summary(field, values, experiments, e, rel_e=None, total_e=Experiment.LocalJIT):
	if e not in experiments:
		return ""

	means = values[field + "_means"]
	if means[e] is None:
		return ""

	stdevs = values[field + "_stdevs"]
	s = "\t{}:{} {:.2f} ±{:.2f}".format(e.name, " " * (max_experiment_name_len - len(e.name)), means[e], stdevs[e])

	do_rel = rel_e is not None and rel_e in experiments
	do_total = total_e is not None and total_e in experiments
	if do_rel or do_total:
		s += " ("
		if do_rel:
			s += "{:+2.1f}%{}".format(rel_change_p(means[e], means[rel_e]), ", " if do_total else "")
		if do_total:
			s += "total {:+2.1f}%".format(rel_change_p(means[e], means[total_e]))
		s += ")"

	return s + "\n"

def field_summary(field, values, experiments):
	s = ""

	s += experiment_summary(field, values, experiments, Experiment.LocalJIT, None, None)
	s += experiment_summary(field, values, experiments, Experiment.JITServer, Experiment.LocalJIT, None)
	s += experiment_summary(field, values, experiments, Experiment.AOTCache, Experiment.JITServer)
	s += experiment_summary(field, values, experiments, Experiment.AOTCacheWarm, Experiment.JITServer)
	s += experiment_summary(field, values, experiments, Experiment.ProfileCache, Experiment.JITServer)
	s += experiment_summary(field, values, experiments, Experiment.ProfileCacheWarm, Experiment.JITServer)
	s += experiment_summary(field, values, experiments, Experiment.AOTPrefetcher, Experiment.AOTCache)
	s += experiment_summary(field, values, experiments, Experiment.AOTPrefetcherWarm, Experiment.AOTCacheWarm)
	s += experiment_summary(field, values, experiments, Experiment.FullCache, Experiment.AOTCache)
	s += experiment_summary(field, values, experiments, Experiment.FullCacheWarm, Experiment.AOTCacheWarm)

	return s

def summary(values, fields, experiments, details=False):
	s = ""

	for f in fields:
		if details or f.startswith("overall_") or not field_descriptors[base_field(f)][2]:
			s += "{} ({}):\n{}\n".format(field_label(f), f, field_summary(f, values, experiments))

	return s


def get_values(results, field, in_values=False):
	return [(r.values[field] if in_values else getattr(r, field)) if r is not None else None for r in results]

def add_mean_stdev(result, results_per_run, field, in_values=False):
	values = get_values(results_per_run, field, in_values)
	result.values[field + "_mean"] = mean(values)
	result.values[field + "_stdev"] = stdev(values)


# results: [[RunResult for all runs] for all instances]
def get_totals_per_run(results, field, in_values=False):
	return [sum(get_values((r[run_id] for r in results), field, in_values)) for run_id in range(len(results[0]))]

# results: [[RunResult for all runs] for all instances]
def add_total_mean_stdev(result, results, field, in_values=False):
	values = get_totals_per_run(results, field, in_values)
	result.values["total_" + field + "_mean"]  = mean(values)
	result.values["total_" + field + "_stdev"] = stdev(values)

def add_mean_stdev_lists(result, results_per_e, field, in_values=False):
	result.values[field + "_means"]  = get_values(results_per_e, field + "_mean",  in_values)
	result.values[field + "_stdevs"] = get_values(results_per_e, field + "_stdev", in_values)

def add_normalized_results(result, fields=None):
	for f in fields or result.fields:
		m = result.values[f + "_means"][Experiment.LocalJIT]
		result.values[f + "_normalized_means"]  = [((v / m) if m else 0.0) if v is not None else None
		                                           for v in result.values[f + "_means"]]
		result.values[f + "_normalized_stdevs"] = [((v / m) if m else 0.0) if v is not None else None
		                                           for v in result.values[f + "_stdevs"]]

	result.fields.extend([f + "_normalized" for f in fields or result.fields])


benchmark_full_names = {
	"acmeair":     "AcmeAir",
	"daytrader":   "DayTrader",
	"petclinic":   "PetClinic",
}


experiment_names = (
	"Local JIT",
	"Remote JIT",
	"Cold AOT cache",
	"Warm AOT cache",
	"Cold profile cache",
	"Warm profile cache",
	"Cold AOT prefetcher",
	"Warm AOT prefetcher",
	"Cold full cache",
	"Warm full cache",
)

experiment_names_single = (
	"Local JIT",
	"Remote JIT",
	"Remote JIT",
	"AOT cache",
	"Remote JIT",
	"Profile cache",
	"Remote JIT",
	"AOT prefetcher",
	"Remote JIT",
	"Full cache",
)

experiment_names_multi = (
	"Local JIT",
	"Remote JIT",
	"AOT cache",
	"AOT cache",
	"Profile cache",
	"Profile cache",
	"AOT prefetcher",
	"AOT prefetcher",
	"Full cache",
	"Full cache",
)


experiment_markers = ("o", "s", "x", "+", "D", "*", "v", "^", "<", ">")
assert len(experiment_markers) >= len(Experiment)

throughput_time_index = True
throughput_marker_interval = 5
throughput_alpha = 0.33


class ApplicationRunResult:
	def __init__(self, bench, config, experiment, *args, actual_experiment=None, **kwargs):
		self.config = config
		self.experiment = experiment
		self.actual_experiment = actual_experiment or experiment

		self.application_output = ApplicationOutput(bench, config, experiment, *args)
		rusage = self.application_output.process_rusage()

		self.start_time = self.application_output.jvm_start_time()
		self.n_comps = self.application_output.n_comps
		self.peak_mem = rusage.peak_mem
		self.cpu_time = rusage.cpu_time()
		self.jit_time = self.application_output.jit_time
		self.data_transferred = self.application_output.bytes_recv / (1024 * 1024) # MB

		self.warmup_data = None
		if config.run_jmeter:
			self.jmeter_output = JMeterOutput(bench.name(), config, experiment, *args, **kwargs)
			self.requests = self.jmeter_output.requests
			self.warmup_data = self.jmeter_output.warmup_data

		if self.warmup_data is not None:
			self.warmup_time = self.warmup_data.warmup_time / 60 # minutes
			self.warmup_end = self.warmup_data.warmup_end
			self.full_warmup_time = self.warmup_time + self.start_time / 60 # minutes
			self.peak_throughput = self.warmup_data.peak_throughput

		self.vlog = self.application_output.vlog() if config.jitserver_config.client_vlog else None
		self.n_lambdas = self.vlog.n_lambdas if self.vlog is not None else 0

	def throughput_df(self):
		data = self.warmup_data.throughput_data
		index = [self.start_time + d[0] for d in data] if throughput_time_index else range(len(data))
		return pd.DataFrame([d[1] for d in data], index=index, columns=[experiment_names[self.actual_experiment]])

	def plot_throughput(self, ax):
		self.throughput_df().plot(
			ax=ax, color="C{}".format(self.actual_experiment.value), legend=False,
			marker=experiment_markers[self.actual_experiment] if throughput_marker_interval is not None else None,
			markevery=throughput_marker_interval
		)

	def plot_peak_throughput_warmup_time(self, ax, ymax):
		c = "C{}".format(self.actual_experiment.value)
		ax.hlines(self.peak_throughput, 0, 1, transform=ax.get_yaxis_transform(), colors=c)

		if throughput_time_index:
			ax.vlines(self.warmup_time * 60, 0, ymax, colors=c) # seconds
		else:
			ax.vlines(self.warmup_end, 0, ymax, colors=c)
			ax.vlines(self.warmup_data.plateau_start, 0, ymax, colors=c, linestyles="dashed")

	def save_stats_plots(self):
		if self.config.collect_stats:
			self.application_output.process_stats().save_plots()
			self.application_output.container_stats().save_plots()

			if self.config.run_jmeter:
				self.jmeter_output.process_stats().save_plots()
				self.jmeter_output.container_stats().save_plots()


class JITServerRunResult:
	def __init__(self, benchmark, config, *args):
		self.jitserver_output = JITServerOutput(benchmark, config, *args)
		rusage = self.jitserver_output.process_rusage()

		self.peak_mem = rusage.peak_mem
		self.cpu_time = rusage.cpu_time()
		self.jit_time = self.cpu_time
		self.jitserver_mem = self.peak_mem / 1024 # GB
		self.jitserver_cpu = self.cpu_time / 60 # minutes
		self.jitserver_mem_perclient = self.jitserver_mem / (config.n_instances / config.n_jitservers)
		self.jitserver_cpu_perclient = self.jitserver_cpu / (config.n_instances / config.n_jitservers)
		self.data_transferred = self.jitserver_output.bytes_recv / (1024 * 1024) # MB

		collect_stats = config.collect_stats or config.jitserver_config.server_resource_stats
		self.process_stats = self.jitserver_output.process_stats() if collect_stats else None
		self.jitserver_peak_cpu_p = self.process_stats.peak_cpu_p if collect_stats else 0.0

	def save_stats_plots(self):
		if self.process_stats:
			self.process_stats.save_plots()

			ct_stats = self.jitserver_output.container_stats()
			if ct_stats is not None:
				ct_stats.save_plots()


class DBRunResult:
	def __init__(self, benchmark, config, *args):
		self.config = config

		self.db_output = DBOutput(benchmark, config, *args)
		rusage = self.db_output.container_rusage()

		self.peak_mem = rusage.peak_mem
		self.cpu_time = rusage.cpu_time()

	def save_stats_plots(self):
		if self.config.collect_stats:
			self.db_output.container_stats().save_plots()


def app_result_fields(config, bench):
	fields = ["start_time", "peak_mem", "n_comps", "cpu_time", "jit_time"]

	if config.run_jmeter:
		fields.extend(("warmup_time", "full_warmup_time", "peak_throughput", "requests"))

	return fields

def jitserver_result_fields():
	return ["peak_mem", "cpu_time", "jit_time", "jitserver_mem", "jitserver_cpu", "jitserver_peak_cpu_p"]

def db_result_fields():
	return ["peak_mem", "cpu_time"]


class ApplicationInstanceResult:
	def __init__(self, bench, config, experiment, instance_id, details=False, *, actual_experiment=None, **kwargs):
		self.config = config
		self.experiment = experiment
		self.actual_experiment = actual_experiment or experiment

		self.results = [ApplicationRunResult(bench, config, experiment, instance_id, r,
		                                     actual_experiment=actual_experiment, **kwargs)
		                for r in range(config.n_runs)]

		self.fields = app_result_fields(config, bench)
		self.values = {}
		for f in self.fields:
			add_mean_stdev(self, self.results, f)

		if config.run_jmeter:
			self.interval = mean((r.warmup_data.throughput_data[-1][0] / (len(r.warmup_data.throughput_data) - 1))
			                     if r.warmup_data is not None else 0.0 for r in self.results)

	def aligned_throughput_df(self, run_id):
		data = self.results[run_id].warmup_data.throughput_data
		return pd.DataFrame([d[1] for d in data], index=[self.interval * i for i in range(len(data))],
		                    columns=[experiment_names[self.actual_experiment]])

	def avg_throughput_df_groups(self):
		return pd.concat(self.aligned_throughput_df(r) for r in range(self.config.n_runs)).groupby(level=0)

	def plot_peak_throughput_warmup_time(self, ax, ymax):
		c = "C{}".format(self.actual_experiment.value)

		m = self.values["peak_throughput_mean"]
		s = self.values["peak_throughput_stdev"]
		ax.hlines(m, 0, 1, transform=ax.get_yaxis_transform(), colors=c)
		ax.axhspan(m - s, m + s, alpha=throughput_alpha, color=c)

		m = self.values["warmup_time_mean"]  * 60 if throughput_time_index else self.values["warmup_end_mean"]  # seconds
		s = self.values["warmup_time_stdev"] * 60 if throughput_time_index else self.values["warmup_end_stdev"] # seconds
		ax.vlines(m, 0, ymax, colors=c)
		ax.axvspan(m - s, m + s, alpha=throughput_alpha, color=c)

	def plot_all_throughput(self, ax):
		for r in range(self.config.n_runs):
			self.results[r].throughput_df().plot(
				ax=ax, color="C{}".format(self.actual_experiment.value), alpha=throughput_alpha, legend=False,
				marker=experiment_markers[self.actual_experiment] if throughput_marker_interval is not None else None,
				markevery=throughput_marker_interval
			)

	def plot_avg_throughput(self, ax):
		groups = self.avg_throughput_df_groups()
		x_df = groups.mean()
		yerr_df = groups.std()
		c = "C{}".format(self.actual_experiment.value)

		x_df.plot(
			ax=ax, color=c,
			marker=experiment_markers[self.actual_experiment] if throughput_marker_interval is not None else None,
			markevery=throughput_marker_interval
		)

		name = experiment_names[self.actual_experiment]
		ax.fill_between(x_df.index, (x_df - yerr_df)[name], (x_df + yerr_df)[name], color=c, alpha=throughput_alpha)

	def save_stats_plots(self):
		for r in self.results:
			r.save_stats_plots()

	def plot_cdf(self, ax, field, log=False, cut=None):
		data = list(itertools.chain.from_iterable(getattr(r.vlog, field) for r in self.results))
		s = pd.Series(data).sort_values()
		if cut is not None:
			s = s[:math.floor(len(s) * cut)]
		e = self.actual_experiment.to_non_warm()
		df = pd.DataFrame(np.linspace(0.0, 1.0, len(s)), index=s, columns=[experiment_names[e]])
		df.plot(ax=ax, logx=log, legend=False, color="C{}".format(e.value))


class JITServerInstanceResult:
	def __init__(self, benchmark, config, *args):
		self.results = [JITServerRunResult(benchmark, config, *args, r) for r in range(config.n_runs)]

		self.fields = jitserver_result_fields()
		self.values = {}
		for f in self.fields:
			add_mean_stdev(self, self.results, f)

		self.jitserver_max_peak_cpu_p = max(r.jitserver_peak_cpu_p for r in self.results)

	def save_stats_plots(self):
		for r in self.results:
			r.save_stats_plots()


class DBInstanceResult:
	def __init__(self, bench, config, *args):
		self.results = [DBRunResult(bench, config, *args, r) for r in range(config.n_runs)]

		self.fields = db_result_fields()
		self.values = {}
		for f in self.fields:
			add_mean_stdev(self, self.results, f)

	def save_stats_plots(self):
		for r in self.results:
			r.save_stats_plots()


def bar_plot_df(result, field):
	return pd.DataFrame({experiment_names[e]: [result.values[field][e]] for e in result.experiments}).iloc[0]


min_throughput_ratio = 0.9

class SingleInstanceExperimentResult:
	def __init__(self, experiments, bench, config, details=False, normalized=False, **kwargs):
		self.experiments = experiments
		self.benchmark = bench.name()
		self.config = config
		self.details = details

		self.app_results = [ApplicationInstanceResult(bench, config, *e.to_single_instance(),
		                                              details, actual_experiment=e, **kwargs)
		                    if e in experiments else None for e in Experiment]

		jr = [JITServerInstanceResult(self.benchmark, config, e.to_single_instance()[0], 0)
		      if (e.is_jitserver() and e in experiments) else None for e in Experiment]
		self.jitserver_results = jr

		self.db_results = ([DBInstanceResult(bench, config, e.to_single_instance()[0], 0)
		                    if e in experiments else None for e in Experiment]
		                   if bench.db_name() is not None else [])

		self.fields = app_result_fields(config, bench)
		self.values = {}
		for f in self.fields:
			add_mean_stdev_lists(self, self.app_results, f, True)

		if config.run_jmeter and (Experiment.LocalJIT in experiments):
			lj_throughput = self.values["peak_throughput_means"][Experiment.LocalJIT]

			for e in experiments:
				if e.is_jitserver() and (self.values["peak_throughput_means"][e] < min_throughput_ratio * lj_throughput):
					duration = config.jmeter_config.duration / 60 # minutes
					self.values["warmup_time_means"][e] = duration # minutes
					self.values["warmup_time_stdevs"][e] = 0.0
					self.values["full_warmup_time_means"][e] = duration + self.values["start_time_means"][e] / 60 # minutes
					self.values["full_warmup_time_stdevs"][e] = self.values["start_time_stdevs"][e] / 60 # minutes

		if normalized and (Experiment.LocalJIT in experiments):
			add_normalized_results(self)

		jitserver_fields = ("jitserver_mem", "jitserver_cpu", "jitserver_peak_cpu_p")
		for f in jitserver_fields:
			self.values[f + "_means"]  = [jr[e].values[f + "_mean"]  if jr[e] is not None else 0.0 for e in Experiment]
			self.values[f + "_stdevs"] = [jr[e].values[f + "_stdev"] if jr[e] is not None else 0.0 for e in Experiment]
		self.fields.extend(jitserver_fields)

		self.jitserver_max_peak_cpu_p = max(r.jitserver_max_peak_cpu_p if r is not None else 0.0 for r in jr)

	def summary(self):
		s = summary(self.values, self.fields, self.experiments, self.details)
		if self.jitserver_max_peak_cpu_p:
			s += "JITServer max peak CPU usage: {}%\n".format(self.jitserver_max_peak_cpu_p)
		return s

	def save_bar_plot(self, field, ymax=None):
		ax = bar_plot_df(self, field + "_means").plot.bar(yerr=bar_plot_df(self, field + "_stdevs"),
		                 rot=0, ylim=(0, ymax))
		ax.set(ylabel=field_label(field))
		save_plot(ax, field, self.benchmark, self.config)

	def save_all_bar_plots(self, limits=None):
		for f in self.fields:
			self.save_bar_plot(f, (limits or {}).get(f))

	def save_throughput_plot(self, ax, name):
		ax.set(xlabel="Time, sec" if throughput_time_index else "Iteration", ylabel="Throughput, req/sec")
		ax.set_xlim(0)
		ax.set_ylim(0)
		save_plot(ax, "throughput_{}".format(name), self.benchmark, self.config, size=(4.5, 2.25))

	def save_run_throughput_plots(self, plot_warmup_peak=True):
		for r in range(self.config.n_runs):
			ax = plt.gca()
			for e in self.experiments:
				self.app_results[e].results[r].plot_throughput(ax)

			if plot_warmup_peak:
				ymax = ax.get_ylim()[1]
				for e in self.experiments:
					self.app_results[e].results[r].plot_peak_throughput_warmup_time(ax, ymax)

			self.save_throughput_plot(ax, "run_{}".format(r))

	def save_all_throughput_plot(self, plot_warmup_peak=True):
		ax = plt.gca()
		for e in self.experiments:
			self.app_results[e].plot_all_throughput(ax)

		if plot_warmup_peak:
			ymax = ax.get_ylim()[1]
			for e in self.experiments:
				self.app_results[e].plot_peak_throughput_warmup_time(ax, ymax)

		ax.legend([matplotlib.lines.Line2D([0], [0], color="C{}".format(e.value)) for e in self.experiments],
		          [experiment_names[e] for e in self.experiments])
		self.save_throughput_plot(ax, "all")

	def save_avg_throughput_plot(self, plot_warmup_peak=True):
		ax = plt.gca()
		for e in self.experiments:
			self.app_results[e].plot_avg_throughput(ax)

		if plot_warmup_peak:
			ymax = ax.get_ylim()[1]
			for e in self.experiments:
				self.app_results[e].plot_peak_throughput_warmup_time(ax, ymax)

		self.save_throughput_plot(ax, "avg")

	def save_stats_plots(self):
		for r in (self.app_results + self.jitserver_results + self.db_results):
			if r is not None:
				r.save_stats_plots()

	def save_cdf_plot(self, field, label, log=False, cut=None, legends=None):
		ax = plt.gca()
		for e in self.experiments:
			self.app_results[e].plot_cdf(ax, field, log, cut)
		ax.set(xlabel=label + (" (log scale)" if log else ""), ylabel="CDF", title="")

		ax.minorticks_off()
		if not legends or legends.get(field):
			ax.legend(
				[matplotlib.lines.Line2D([0], [0], color="C{}".format(e.to_non_warm().value)) for e in self.experiments],
				[experiment_names_single[e] for e in self.experiments]
			)

		name = field + ("_log" if log else "") + ("_cut" if cut is not None else "")
		save_plot(ax, name, self.benchmark, self.config)

	def save_results(self, limits=None, legends=None, cdf_plots=False, plot_warmup_peak=True):
		save_summary(self.summary(), self.benchmark, self.config)

		if self.details:
			self.save_all_bar_plots(limits)

			if self.config.run_jmeter:
				self.save_run_throughput_plots(plot_warmup_peak)
				self.save_all_throughput_plot(plot_warmup_peak)
				self.save_avg_throughput_plot(plot_warmup_peak)

			self.save_stats_plots()

		if self.config.jitserver_config.client_vlog:
			if cdf_plots:
				for field, label, log, cut in vlog_cdf_fields:
					self.save_cdf_plot(field, label, legends=legends)
					if log:
						self.save_cdf_plot(field, label, log=True, legends=legends)
					if cut is not None:
						self.save_cdf_plot(field, label, cut=cut, legends=legends)

			s = ""
			for e in self.experiments:
				for r in self.app_results[e].results:
					s += r.vlog.dups_summary()
			if s:
				save_summary(s, self.benchmark, self.config, name="dups")


class SingleInstanceAllExperimentsResult:
	def __init__(self, experiments, bench, mode, configs, config_names, details=False,
	             no_aotcache=False, no_warm_scc=False, kwargs_list=None):
		self.experiments = [e for e in experiments if not e.is_warm_cache()] if no_aotcache else experiments
		self.benchmark = bench.name()
		self.mode = mode
		self.configs = configs
		self.config_names = config_names
		self.warmup = configs[0].run_jmeter
		self.no_aotcache = no_aotcache
		self.no_warm_scc = no_warm_scc

		self.results = [SingleInstanceExperimentResult(experiments, bench, configs[i], details,
		                                               **((kwargs_list[i] or {}) if kwargs_list is not None else {}))
		                for i in range(len(configs))]

	def get_df(self, field, suffix):
		data = {
			experiment_names_single[e]: [self.results[i].values[field + suffix][e] for i in range(len(self.configs))]
			for e in self.experiments
		}
		return pd.DataFrame(data, index=self.config_names)

	def save_bar_plot(self, field, ymax=None, legend=True, dry_run=False):
		ax = self.get_df(field, "_means").plot.bar(yerr=self.get_df(field, "_stdevs"),
		                 rot=0, ylim=(0, ymax), legend=legend)

		xlabel = "{}{}: Container size".format(benchmark_full_names[self.benchmark],
		                                       "" if self.no_warm_scc else (" " + self.mode))
		ax.set(xlabel=xlabel, ylabel=field_label(field))
		result = ax.get_ylim()[1]

		if dry_run:
			plt.close(ax.get_figure())
		else:
			name = "{}single_{}_{}_{}".format("jitserver_" if self.no_aotcache else "",
			                                  "full" if self.warmup else "start", self.mode, field)
			save_plot(ax, name, self.benchmark)

		return result

	def save_all_bar_plots(self, limits=None, legends=None, dry_run=False):
		result = {}
		for f in self.results[0].fields:
			result[f] = self.save_bar_plot(f, (limits or {}).get(f), legends.get(f) if legends else True)
		return result

	def save_results(self, limits=None, legends=None, dry_run=False):
		return self.save_all_bar_plots(limits, legends, dry_run)


class ApplicationAllInstancesResult:
	def peak_total_mem(self, run_id):
		timestamps = []
		buffer = datetime.timedelta(seconds=1.0)

		for i in range(self.n_instances):
			r = self.results[i][run_id]
			start = r.application_output.docker_ts - buffer
			stop  = r.application_output.stop_ts   + buffer
			timestamps.extend(((start, 1, r.peak_mem), (stop, -1, r.peak_mem)))

		timestamps.sort(key=lambda t: t[0])

		total = 0.0
		max_total = 0.0
		for t in timestamps:
			total += t[1] * t[2]
			max_total = max(max_total, total)

		return max_total / 1024 # GB

	def __init__(self, bench, config, experiment, details=False, *, n_instances=None, **kwargs):
		self.config = config
		self.experiment = experiment
		self.n_instances = n_instances or config.n_instances

		self.results = [[ApplicationRunResult(bench, config, experiment, i, r, **kwargs)
		                 for r in range(config.n_runs)] for i in range(self.n_instances)]
		self.all_results = list(itertools.chain.from_iterable(self.results))

		self.fields = app_result_fields(config, bench)
		self.values = {}
		for f in self.fields:
			add_mean_stdev(self, self.all_results, f)

		total_fields = ["peak_mem", "cpu_time", "jit_time"]
		if self.config.run_jmeter:
			total_fields.append("requests")

		for f in total_fields:
			add_total_mean_stdev(self, self.results, f)
		self.fields.extend("total_" + f for f in total_fields)

		if self.config.n_invocations is not None:
			self.fields.append("peak_total_mem")
			peak_total_mem_values = [self.peak_total_mem(r) for r in range(config.n_runs)]
			self.values["peak_total_mem_mean"] = mean(peak_total_mem_values)
			self.values["peak_total_mem_stdev"] = stdev(peak_total_mem_values)


class JITServerAllInstancesResult:
	def __init__(self, benchmark, config, experiment):
		self.results = [[JITServerRunResult(benchmark, config, experiment, i, r)
		                 for r in range(config.n_runs)] for i in range(config.n_jitservers)]
		self.all_results = list(itertools.chain.from_iterable(self.results))

		self.fields = jitserver_result_fields()
		self.values = {}
		for f in self.fields:
			add_mean_stdev(self, self.all_results, f)

		total_fields = ("peak_mem", "cpu_time", "jit_time", "jitserver_mem", "jitserver_cpu")
		for f in total_fields:
			add_total_mean_stdev(self, self.results, f)
		self.fields.extend("total_" + f for f in total_fields)

		per_client_fields = ("jitserver_cpu", "jitserver_mem")
		for f in per_client_fields:
			add_mean_stdev(self, self.all_results, f + "_perclient")
		self.fields.extend(f + "_perclient" for f in per_client_fields)

		if config.n_invocations is not None:
			self.fields.append("peak_total_mem")
			self.values["peak_total_mem_mean"]  = self.values["total_peak_mem_mean"]  / 1024 # GB
			self.values["peak_total_mem_stdev"] = self.values["total_peak_mem_stdev"] / 1024 # GB

		self.jitserver_max_peak_cpu_p = max(r.jitserver_peak_cpu_p for r in itertools.chain.from_iterable(self.results))

	def save_stats_plots(self):
		for r in self.all_results:
			r.save_stats_plots()


class DBAllInstancesResult:
	def __init__(self, bench, config, experiment):
		self.results = [[DBRunResult(bench, config, experiment, i, r)
		                 for r in range(config.n_runs)] for i in range(config.n_dbs)]
		self.all_results = list(itertools.chain.from_iterable(self.results))

		self.fields = db_result_fields()
		self.values = {}
		for f in self.fields:
			add_mean_stdev(self, self.all_results, f)
			add_total_mean_stdev(self, self.results, f)
		self.fields.extend(["total_" + f for f in self.fields])

	def save_stats_plots(self):
		for r in self.all_results:
			r.save_stats_plots()


class MultiInstanceExperimentResult:
	def __init__(self, experiments, bench, config, details=False, normalized=False, *,
	             normalized_fields=None, n_instances=None, **kwargs):
		self.experiments = experiments
		self.benchmark = bench.name()
		self.config = config
		self.details = details

		ar = [ApplicationAllInstancesResult(bench, config, e, details, n_instances=n_instances,
		                                    keep_throughput_data=False, **kwargs)
		      if e in experiments else None for e in Experiment]
		self.app_results = ar

		jr = [JITServerAllInstancesResult(self.benchmark, config, e)
		      if (e.is_jitserver() and e in experiments) else None for e in Experiment]
		self.jitserver_results = jr

		self.db_results = ([DBAllInstancesResult(bench, config, e) if e in experiments else None for e in Experiment]
		                   if bench.db_name() is not None else [])

		self.fields = app_result_fields(config, bench)
		self.values = {}
		for f in self.fields:
			add_mean_stdev_lists(self, self.app_results, f, True)

		overall_fields = ["total_" + f for f in ("peak_mem", "cpu_time", "jit_time")]
		if config.n_invocations is not None:
			overall_fields.append("peak_total_mem")
		self.fields.extend("overall_" + f for f in overall_fields)

		for f in overall_fields:
			self.values["overall_{}_means".format(f)] = [
				(ar[e].values[f + "_mean"] + (jr[e].values[f + "_mean"] if jr[e] is not None else 0))
				if ar[e] is not None else None for e in Experiment
			]
			#TODO: compute more accurately (variables are not independent)
			self.values["overall_{}_stdevs".format(f)] = [
				(ar[e].values[f + "_stdev"] + (jr[e].values[f + "_stdev"] if jr[e] is not None else 0))
				if ar[e] is not None else None for e in Experiment
			]

		if config.run_jmeter and config.n_invocations is not None:
			self.fields.append("cpu_time_per_req")

			reqs = [[
				sum(ar[e].results[i][r].requests for i in range(ar[e].n_instances))
				if ar[e] is not None else None for r in range(config.n_runs)] for e in Experiment
			]
			cpu_times = [[
				(sum(ar[e].results[i][r].cpu_time for i in range(ar[e].n_instances)) +
				(sum(jr[e].results[i][r].cpu_time for i in range(config.n_jitservers)) if jr[e] is not None else 0.0))
				if ar[e] is not None else None for r in range(config.n_runs)] for e in Experiment
			]
			cpu_times_per_req = [[1000 * cpu_times[e][r] / reqs[e][r] if cpu_times[e][r] is not None else None # msec/req
			                      for r in range(config.n_runs)] for e in Experiment]

			self.values["cpu_time_per_req_means"]  = [mean(cpu_times_per_req[e][r] for r in range(config.n_runs))
			                                          if cpu_times_per_req[e] is not None else None for e in Experiment]
			self.values["cpu_time_per_req_stdevs"] = [stdev(cpu_times_per_req[e][r] for r in range(config.n_runs))
			                                          if cpu_times_per_req[e] is not None else None for e in Experiment]

		if Experiment.LocalJIT in experiments:
			if normalized:
				add_normalized_results(self)
			elif normalized_fields is not None and config.run_jmeter:
				add_normalized_results(self, normalized_fields)

		extra_fields = ["jitserver_mem", "jitserver_cpu"]
		jitserver_fields = (extra_fields + ["total_" + f for f in extra_fields] +
		                    [f + "_perclient" for f in extra_fields])
		self.fields.extend(jitserver_fields)

		for f in jitserver_fields:
			self.values[f + "_means"]  = [jr[e].values[f + "_mean"]  if jr[e] is not None else 0.0 for e in Experiment]
			self.values[f + "_stdevs"] = [jr[e].values[f + "_stdev"] if jr[e] is not None else 0.0 for e in Experiment]

		self.jitserver_max_peak_cpu_p = max(r.jitserver_max_peak_cpu_p if r is not None else 0.0
		                                    for r in self.jitserver_results)

	def summary(self):
		s = summary(self.values, self.fields, self.experiments, self.details)
		if self.jitserver_max_peak_cpu_p:
			s += "JITServer max peak CPU usage: {}%\n".format(self.jitserver_max_peak_cpu_p)
		return s

	def save_bar_plot(self, field, ymax=None):
		ax = bar_plot_df(self, field + "_means").plot.bar(yerr=bar_plot_df(self, field + "_stdevs"),
		                                                  rot=0, ylim=(0, ymax))
		ax.set(ylabel=field_label(field))
		save_plot(ax, field, self.benchmark, self.config)

	def save_all_bar_plots(self, limits=None):
		for f in self.fields:
			self.save_bar_plot(f, (limits or {}).get(f))

	def save_stats_plots(self):
		for r in (self.jitserver_results + self.db_results):
			if r is not None:
				r.save_stats_plots()

	def save_results(self, limits=None):
		save_summary(self.summary(), self.benchmark, self.config)

		if self.details:
			self.save_all_bar_plots(limits)
			self.save_stats_plots()


class ScaleExperimentResult(MultiInstanceExperimentResult):
	def __init__(self, experiments, bench, config, details=False, normalized=False, **kwargs):
		super().__init__(experiments, bench, config, details, normalized, normalized_fields=["full_warmup_time"])


class ScaleAllExperimentsResult:
	def __init__(self, experiments, bench, configs, details=False, kwargs_list=None):
		self.experiments = experiments
		self.benchmark = bench.name()
		self.warmup = configs[0].run_jmeter

		self.results = [ScaleExperimentResult(experiments, bench, configs[i], details, keep_throughput_data=False,
		                                      **((kwargs_list[i] or {}) if kwargs_list is not None else {}))
		                for i in range(len(configs))]

	def get_df(self, field, suffix, name_suffix=None):
		return pd.DataFrame({experiment_names_multi[e] + (name_suffix or ""):
		                     [r.values[field + suffix][e] for r in self.results] for e in self.experiments},
		                    index=[r.config.n_instances for r in self.results])

	def save_line_plot(self, field, ymax=None, legend=True):
		name = "scale_{}_{}".format("full" if self.warmup else "start", field)
		ax = self.get_df(field, "_means").plot(yerr=self.get_df(field, "_stdevs"),
		                                       ylim=(0, ymax), xlim=(0, None), legend=False)

		if field.endswith("_normalized"):
			ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
		ax.set(xlabel="{}: Number of instances".format(benchmark_full_names[self.benchmark]), ylabel=field_label(field))

		for i, line in enumerate(ax.get_lines()):
			line.set_marker(experiment_markers[i])
		if legend:
			ax.legend()

		save_plot(ax, name, self.benchmark)

	def save_all_line_plots(self, limits=None, legends=None):
		for f in self.results[0].fields:
			self.save_line_plot(f, (limits or {}).get(f), legends.get(f) if legends else True)

	def save_results(self, limits=None, legends=None):
		self.save_all_line_plots(limits, legends)


class LatencyExperimentResult(SingleInstanceExperimentResult):
	def __init__(self, experiments, bench, config, details=False, **kwargs):
		super().__init__(experiments, bench, config, details, **kwargs)

		with open(os.path.join(logs_path(self.benchmark, config), "latency.log"), "r") as f:
			self.latency = float(f.readlines()[0].strip())

	def summary(self):
		s = "Latency: {} us\n\n".format(self.latency)
		s += super().summary()
		return s


class LatencyAllExperimentsResult:
	def __init__(self, experiments, bench, configs, details=False, kwargs_list=None):
		self.experiments = experiments
		self.benchmark = bench.name()
		self.warmup = configs[0].run_jmeter

		self.results = [LatencyExperimentResult(experiments, bench, configs[i], details,
		                                        **((kwargs_list[i] or {}) if kwargs_list is not None else {}))
		                for i in range(len(configs))]

	def get_df(self, field, suffix, f):
		data = {experiment_names_single[e]: [r.values[field + suffix][e] for r in self.results]
		        for e in self.experiments}
		return pd.DataFrame(data, index=[r.latency for r in self.results])

	def save_line_plot(self, field, ymax=None, legend=True):
		ax = self.get_df(field, "_means", mean).plot(yerr=self.get_df(field, "_stdevs", stdev),
		                                             ylim=(0, ymax), xlim=(0, None), legend=False)
		ax.set(xlabel="{}: Latency, microsec".format(benchmark_full_names[self.benchmark]), ylabel=field_label(field))

		for i, line in enumerate(ax.get_lines()):
			line.set_marker(experiment_markers[i])
		if legend:
			ax.legend()

		name = "latency_{}_{}".format("full" if self.warmup else "start", field)
		save_plot(ax, name, self.benchmark)

	def save_all_line_plots(self, limits=None, legends=None):
		for f in self.results[0].fields:
			self.save_line_plot(f, (limits or {}).get(f), legends.get(f) if legends else True)

	def save_results(self, limits=None, legends=None):
		self.save_all_line_plots(limits, legends)


class DensityExperimentResult(MultiInstanceExperimentResult):
	def __init__(self, experiments, bench, config, details=False, normalized=False, **kwargs):
		super().__init__(experiments, bench, config, details, normalized,
		                 n_instances=config.n_instances * config.n_invocations, **kwargs)


class DensityAllExperimentsResult:
	def __init__(self, experiments, bench, configs, details=False, no_aotcache=False, kwargs_list=None):
		self.experiments = [e for e in experiments if not e.is_cache()] if no_aotcache else experiments
		self.benchmark = bench.name()
		self.configs = configs
		self.no_aotcache = no_aotcache

		self.results = [DensityExperimentResult(experiments, bench, configs[i], details,
		                                        **((kwargs_list[i] or {}) if kwargs_list is not None else {}))
		                for i in range(len(configs))]

	def get_df(self, field, suffix):
		data = {
			experiment_names_multi[e]: [self.results[i].values[field + suffix][e] for i in range(len(self.configs))]
			for e in self.experiments
		}
		index = ["{:g} min".format(c.jmeter_config.duration / 60) for c in self.configs]
		return pd.DataFrame(data, index=index)

	def get_overlay_df(self, field, part_f, total_f, factor):
		v = [self.results[i].values for i in range(len(self.configs))]
		values = [[((factor or 1) * v[i][field + "_means"][e] * v[i][part_f + "_means"][e] / v[i][total_f + "_means"][e])
		           if v[i][part_f + "_means"][e] else 0.0 for i in range(len(self.configs))] for e in Experiment]

		data = {experiment_names_multi[e]: [values[e][i] for i in range(len(self.configs))] for e in self.experiments}
		index = ["{:g} min".format(c.jmeter_config.duration / 60) for c in self.configs]
		return pd.DataFrame(data, index=index)

	def save_bar_plot(self, field, ymax=None, legend=True, dry_run=False,
	                  overlay_label=None, part_f=None, total_f=None, factor=None):
		ax = self.get_df(field, "_means").plot.bar(yerr=self.get_df(field, "_stdevs"), rot=0, ylim=(0, ymax),
		                                           legend=legend and overlay_label is None)

		if overlay_label is not None:
			self.get_overlay_df(field, part_f, total_f, factor).plot.bar(
				ax=ax, rot=0, ylim=(0, ymax), legend=False, edgecolor="black", linewidth=0.5, hatch="xxxxxx", alpha=0.5
			)
			if legend:
				ax.legend(
					[matplotlib.patches.Patch(color="C{}".format(e.value)) for e in self.experiments] +
					[matplotlib.patches.Patch(edgecolor="black", facecolor="none",
					                          linewidth=0.5, hatch="xxxxxx", alpha=0.5)],
					[experiment_names_multi[e] for e in self.experiments] + [overlay_label]
				)

		if (self.benchmark == "acmeair") and (field == "cpu_time_per_req"):
			ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
		ax.set(xlabel="{}: Application lifespan".format(benchmark_full_names[self.benchmark]), ylabel=field_label(field))

		result = ax.get_ylim()[1]
		if dry_run:
			plt.close(ax.get_figure())
		else:
			name = "{}density_{}_{}".format("jitserver_" if self.no_aotcache else "",
			                                "scc" if self.configs[0].application_config.populate_scc else "noscc", field)
			save_plot(ax, name, self.benchmark)

		return result

	def save_all_bar_plots(self, limits=None, legends=None, dry_run=False, overlays=False):
		result = {}

		for f in self.results[0].fields:
			overlay_label = None
			part_f = None
			total_f = None
			factor = None

			if overlays:
				if f == "cpu_time_per_req":
					overlay_label = "JITServer resources" if legends else "JITServer CPU"
					part_f = "total_jitserver_cpu"
					total_f = "overall_total_cpu_time"
					factor = 60
				elif f == "overall_peak_total_mem":
					overlay_label = "JITServer resources" if legends else "JITServer memory"
					part_f = "total_jitserver_mem"
					total_f = "overall_peak_total_mem"

			result[f] = self.save_bar_plot(f, (limits or {}).get(f), legends.get(f) if legends else True,
			                               dry_run, overlay_label, part_f, total_f, factor)

		return result

	def save_results(self, limits=None, legends=None, dry_run=False, overlays=False):
		return self.save_all_bar_plots(limits, legends, dry_run, overlays)


servermem_mode_names = (
	"Baseline",
	"Per-client allocators",
	"Shared ROMClasses",
	"Both optimizations",
)

max_servermem_mode_name_len = max(len(n) for n in servermem_mode_names)


class ServerMemAllExperimentsResult:
	def __init__(self, bench, configs, details=False, kwargs_list=None):
		self.benchmark = bench.name()
		self.warmup = configs[0][0].run_jmeter
		self.experiment = Experiment.JITServer

		self.results = [[
			ScaleExperimentResult([self.experiment], bench, configs[m][i], details, keep_throughput_data=False,
			                      **((kwargs_list[i] or {}) if kwargs_list is not None else {}))
			for i in range(len(configs[m]))] for m in range(len(configs))
		]

		self.fields = ("jitserver_mem", "jitserver_cpu_perclient")

	def get_df(self, field, suffix):
		data = {servermem_mode_names[m]: [r.values[field + suffix][self.experiment] for r in self.results[m]]
		        for m in range(len(self.results))}
		return pd.DataFrame(data, index=[r.config.n_instances for r in self.results[0]])

	def save_line_plot(self, field, legend=True):
		ax = self.get_df(field, "_means").plot(yerr=self.get_df(field, "_stdevs"), xlim=(0, None), legend=False)

		if field == "jitserver_mem":
			ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f"))
		ax.set(xlabel="{}: Number of instances".format(benchmark_full_names[self.benchmark]), ylabel=field_label(field))

		ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
		for i, line in enumerate(ax.get_lines()):
			line.set_marker(experiment_markers[i])
		if legend:
			ax.legend()

		name = "servermem_{}_{}".format("full" if self.warmup else "start", field)
		save_plot(ax, name, self.benchmark)

	def save_all_line_plots(self, legends=None):
		for f in self.fields:
			self.save_line_plot(f, legends.get(f) if legends else True)

	def summary(self):
		s = ""
		e = self.experiment

		for n in range(len(self.results[0])):
			s += "{} instances:\n".format(self.results[0][n].config.n_instances)

			for f in self.fields:
				s += "\t{} ({}):\n".format(field_label(f, True), f)

				for m in range(len(self.results)):
					s += "\t\t{}:{} {:.2f} ±{:.2f}".format(
						servermem_mode_names[m], " " * (max_servermem_mode_name_len - len(servermem_mode_names[m])),
						self.results[m][n].values[f + "_means"][e], self.results[m][n].values[f + "_stdevs"][e]
					)
					if m:
						s += " ({:+2.1f}%)".format(rel_change_p(self.results[m][n].values[f + "_means"][e],
						                                        self.results[0][n].values[f + "_means"][e]))

					s += "\n"
				s += "\n"

		return s

	def save_results(self, legends=None):
		save_summary(self.summary(), self.benchmark, name="servermem_summary")
		self.save_all_line_plots(legends)


class NThreadsAllExperimentsResult:
	def __init__(self, experiments, bench, configs, details=False, kwargs_list=None):
		self.experiments = experiments
		self.benchmark = bench.name()
		self.nthreads = [configs[n][0].jitserver_config.client_threads for n in range(len(configs))]
		self.warmup = configs[0][0].run_jmeter

		self.results = [[LatencyExperimentResult(experiments, bench, configs[n][c], details,
		                                         **((kwargs_list[c] or {}) if kwargs_list is not None else {}))
		                for c in range(len(configs[n]))] for n in range(len(configs))]

	def get_df(self, e, n, field, suffix):
		label = "{} threads".format(self.nthreads[n]) if e.is_jitserver() else experiment_names[e]
		data = {label: [r.values[field + suffix][e] for r in self.results[n]]}
		return pd.DataFrame(data, index=[r.latency for r in self.results[n]])

	def get_agg_mean_df(self, field):
		values = [[mean(r.values[field + "_means"][e] for r in self.results[n]) for n in range(len(self.results))]
		          if e.is_jitserver() and (e in self.experiments) else None for e in Experiment]

		data = {experiment_names[e]: values[e] for e in self.experiments if e.is_jitserver()}
		return pd.DataFrame(data, index=[self.nthreads[n] for n in range(len(self.results))])

	def get_agg_stdev_df(self, field):
		values = [[[r.values[field + "_means"][e] for r in self.results[n]] for n in range(len(self.results))]
		          if e.is_jitserver() and (e in self.experiments) else None for e in Experiment]
		stdevs = [[stdev(values[e][n][i] for i in range(len(self.results[n]))) for n in range(len(self.results))]
		          if e.is_jitserver() and (e in self.experiments) else None for e in Experiment]

		data = {experiment_names[e]: stdevs[e] for e in self.experiments if e.is_jitserver()}
		return pd.DataFrame(data, index=[self.nthreads[n] for n in range(len(self.results))])

	def save_line_plot(self, field, legend=True):
		ax = plt.gca()

		if not field.startswith("jitserver_"):
			self.get_df(Experiment.LocalJIT, 0, field, "_means").plot(
				ax=ax, yerr=self.get_df(Experiment.LocalJIT, 0, field, "_stdevs"),
				color="C0", xlim=(0, None), legend=False
			)

		for n in range(len(self.results)):
			self.get_df(Experiment.JITServer, n, field, "_means").plot(
				ax=ax, yerr=self.get_df(Experiment.JITServer, n, field, "_stdevs"),
				xlim=(0, None), legend=False, color="C{}".format(n + 1)
			)

		ax.set(xlabel="{}: Latency, microsec".format(benchmark_full_names[self.benchmark]), ylabel=field_label(field))
		ax.set_ylim(0)
		for i, line in enumerate(ax.get_lines()):
			line.set_marker(experiment_markers[i + (1 if field.startswith("jitserver_") else 0)])

		if legend:
			handles = [matplotlib.lines.Line2D([0], [0], color="C0", marker=experiment_markers[0])]
			labels = [experiment_names[Experiment.LocalJIT]]

			handles.extend(matplotlib.lines.Line2D([0], [0], color="C{}".format(n + 1), marker=experiment_markers[n + 1])
			               for n in range(len(self.results)))
			labels.extend("{} threads".format(self.nthreads[n]) for n in range(len(self.results)))

			ax.legend(handles, labels)

		name = "nthreads_{}_{}".format("full" if self.warmup else "start", field)
		save_plot(ax, name, self.benchmark)

	def save_agg_line_plot(self, field, legend=True):
		ax = self.get_agg_mean_df(field).plot(
			yerr=self.get_agg_stdev_df(field), logx=True, legend=False,
			color=["C{}".format(e.value) for e in self.experiments if e.is_jitserver()]
		)
		ax.set(xlabel="{}: Client threads".format(benchmark_full_names[self.benchmark]), ylabel=field_label(field))

		ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
		ax.set_xticks(self.nthreads)
		ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
		ax.minorticks_off()

		for i, line in enumerate(ax.get_lines()):
			line.set_marker(experiment_markers[i])
		if legend:
			ax.legend()

		name = "nthreads_{}_{}_agg".format("full" if self.warmup else "start", field)
		save_plot(ax, name, self.benchmark)

	def save_all_line_plots(self, legends=None):
		for f in self.results[0][0].fields:
			self.save_line_plot(f, legends.get(f) if legends else True)
			self.save_agg_line_plot(f, legends.get(f) if legends else True)

	def save_results(self, legends=None):
		self.save_all_line_plots(legends)
