import datetime
import csv
import itertools
import math
import os
import os.path
import re
import statistics

import matplotlib
matplotlib.use("agg")
import matplotlib.lines
import matplotlib.pyplot as plt
import matplotlib.ticker

import numpy as np
import pandas as pd

plt.rcParams.update({
	"axes.labelpad": 3.0,
	"figure.figsize": (1.8, 1.2),
	"font.size": 6,
	"hatch.linewidth": 0.5,
	"legend.fontsize": 5,
	"legend.framealpha": 0.5,
	"lines.linewidth": 1.0,
	"lines.markersize": 3.0,
	"savefig.bbox": "tight",
	"savefig.dpi": 300,
	"savefig.pad_inches": 0.05,
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

def save_plot(ax, name, *args):
	path = os.path.join(results_path(*args), "{}.{}".format(name, plot_format))
	os.makedirs(os.path.dirname(path), exist_ok=True)
	ax.get_figure().savefig(path)
	plt.close(ax.get_figure())


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

		self.max_cpu_p = max(self.cpu_data)

	def cpu_df(self):
		idx = [self.period * i for i in range(len(self.cpu_data))]
		return pd.DataFrame(self.cpu_data, index=idx, columns=["cpu"])

	def mem_df(self):
		idx = [self.period * i for i in range(len(self.mem_data))]
		return pd.DataFrame(self.mem_data, index=idx, columns=["mem"])

	def save_plot(self, ax, name, label):
		ax.set(xlabel="Time, sec", ylabel=label)
		save_plot(ax, "{}_{}".format(name, self.kind()), *self.id)

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
		self.jit_cpu_time = 0.0 # seconds

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
				self.jit_cpu_time += (t or 0.0) / 1000.0
				if parsed: continue

				t, parsed = parse_first_token(None, line, "Time spent in AOT prefetcher thread: ", float)
				self.jit_cpu_time += (t or 0.0) / 1000.0

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

		return idx < len(data) - 1

	def reached_threshold(self, data, idx):
		return ((data[idx] >= self.threshold * self.peak_throughput) and
		        ((idx + 1 >= len(data)) or (data[idx + 1] >= self.next_threshold * self.peak_throughput)))

	def __init__(self, throughput_data, duration=None, *, keep_throughput_data=True, threshold=None,
	             next_threshold=None, margin=None, outlier_limit=None, window=None):
		self.throughput_data = throughput_data
		self.threshold = threshold or 0.9
		self.next_threshold = next_threshold or 0.8
		self.margin = margin or 0.1
		self.outlier_limit = outlier_limit or 0.1

		data = [d[1] for d in self.throughput_data]
		if window is not None:
			data = pd.DataFrame(data).rolling(window, 1, True).mean()[0].to_list()

		plateau_start = next((i for i in range(len(data)) if self.is_plateau(data, i)), len(data) - 1)
		self.peak_throughput = mean(d for d in data[plateau_start:])

		warmup_end = next((i for i in range(len(data)) if self.reached_threshold(data, i)), len(data) - 1)
		self.warmup_time = self.throughput_data[warmup_end][0] or self.throughput_data[-1][0]
		self.warmup_avg_throughput = sum(data[i] for i in range(min(warmup_end + 1, len(data)))) / self.warmup_time

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


def mean(data):
	return statistics.mean(d for d in data if d is not None)

def stdev(data):
	return statistics.stdev(d for d in data if d is not None) if len(data) > 1 else 0.0

def rel_change(x, x_ref):
	return ((x - x_ref) / x_ref) if x_ref else 0.0

def rel_change_p(x, x_ref):
	return 100.0 * rel_change(x, x_ref)


def get_list(results, field, in_values=False):
	return [(r.values[field] if in_values else getattr(r, field)) if r is not None else None for r in results]

def get_totals(results, field, in_values=False):
	return [sum(get_list((r[run_id] for r in results), field, in_values)) for run_id in range(len(results[0]))]


def add_mean_stdev(result, results, field, in_values=False):
	vals = get_list(results, field, in_values)
	result.values[field + "_mean"] = mean(vals)
	result.values[field + "_stdev"] = stdev(vals)
	return vals

def add_min_max(result, results, field, in_values=False):
	vals = get_list(results, field, in_values)
	result.values[field + "_min"] = min(vals)
	result.values[field + "_max"] = max(vals)
	return vals

def add_mean_stdev_lists(result, results, field, in_values=False):
	result.values[field + "_means"] = get_list(results, field + "_mean", in_values)
	result.values[field + "_stdevs"] = get_list(results, field + "_stdev", in_values)

def add_min_max_lists(result, results, field, in_values=False):
	result.values[field + "_mins"] = get_list(results, field + "_min", in_values)
	result.values[field + "_maxs"] = get_list(results, field + "_max", in_values)

# results: [[[RunResult for all runs] for all instances] for all experiments]
def add_total_mean_stdev_lists(result, results, field, in_values=False):
	vals = [get_totals(r, field, in_values) if r is not None else None for r in results]
	result.values["total_{}_means".format(field)] = [mean(vals[i]) if results[i] is not None else None
	                                                 for i in range(len(results))]
	result.values["total_{}_stdevs".format(field)] = [stdev(vals[i]) if results[i] is not None else None
	                                                  for i in range(len(results))]
	return vals


max_experiment_name_len = max(len(e.name) for e in Experiment)

def experiment_summary(means, stdevs, experiments, e, rel_e=None, total_e=Experiment.LocalJIT):
	if e not in experiments:
		return ""

	s = "\t{}:{} {:2.2f} ±{:1.2f}".format(e.name, " " * (max_experiment_name_len - len(e.name)), means[e], stdevs[e])

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

def field_summary(means, stdevs, experiments):
	s = ""

	s += experiment_summary(means, stdevs, experiments, Experiment.LocalJIT, None, None)
	s += experiment_summary(means, stdevs, experiments, Experiment.JITServer, Experiment.LocalJIT, None)
	s += experiment_summary(means, stdevs, experiments, Experiment.AOTCache, Experiment.JITServer)
	s += experiment_summary(means, stdevs, experiments, Experiment.AOTCacheWarm, Experiment.JITServer)

	return s

def summary(values, fields, experiments):
	s = ""
	for f in fields:
		s += "{}:\n{}\n".format(f[1], field_summary(values[f[0] + "_means"], values[f[0] + "_stdevs"], experiments))
	return s


benchmark_full_names = {
	"acmeair": "AcmeAir",
	"daytrader": "DayTrader",
	"petclinic": "PetClinic",
}

experiment_names = (
	"Local JIT",
	"Remote JIT",
	"Remote JIT + cold cache",
	"Remote JIT + warm cache",
)

experiment_names_single = (
	"Local JIT",
	"Remote JIT",
	"Remote JIT",
	"Remote JIT + cache",
)

experiment_names_multi = (
	"Local JIT",
	"Remote JIT",
	"Remote JIT + cache",
	"Remote JIT + cache",
)

experiment_markers = ("o", "s", "x", "+")
assert len(experiment_markers) == len(Experiment)

throughput_marker_interval = 5
throughput_alpha = 0.33
throughput_time_index = True


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
		self.jit_cpu_time = self.application_output.jit_cpu_time
		self.jitserver_mem = 0.0
		self.jitserver_cpu = 0.0
		self.data_transferred = self.application_output.bytes_recv / (1024 * 1024) # MB

		self.warmup_data = None
		if config.run_jmeter:
			self.jmeter_output = JMeterOutput(bench.name(), config, experiment, *args, **kwargs)
			self.requests = self.jmeter_output.requests
			self.warmup_data = self.jmeter_output.warmup_data

		if self.warmup_data is not None:
			self.warmup_time = self.warmup_data.warmup_time
			self.full_warmup_time = self.warmup_time + self.start_time
			self.warmup_avg_throughput = self.warmup_data.warmup_avg_throughput
			self.peak_throughput = self.warmup_data.peak_throughput

		self.vlog = self.application_output.vlog() if config.jitserver_config.client_vlog else None
		self.n_lambdas = self.vlog.n_lambdas if self.vlog is not None else 0

	def throughput_df(self):
		data = self.warmup_data.throughput_data
		index = [d[0] for d in data] if throughput_time_index else range(len(data))
		return pd.DataFrame([d[1] for d in data], index=index, columns=[experiment_names[self.actual_experiment]])

	def plot_run_throughput(self, ax):
		c = "C{}".format(self.actual_experiment.value)

		self.throughput_df().plot(ax=ax, color=c, marker=experiment_markers[self.actual_experiment],
		                          markevery=throughput_marker_interval)

		ax.hlines(self.peak_throughput, 0, 1, transform=ax.get_yaxis_transform(), colors=c)
		ax.vlines(self.warmup_time, 0, 1, transform=ax.get_xaxis_transform(), colors=c)

	def save_stats_plots(self):
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
		self.jit_cpu_time = self.cpu_time
		self.jitserver_mem = self.peak_mem
		self.jitserver_cpu = self.cpu_time
		self.data_transferred = self.jitserver_output.bytes_recv / (1024 * 1024) # MB

		collect_stats = config.collect_stats or config.jitserver_config.server_resource_stats
		self.process_stats = self.jitserver_output.process_stats() if collect_stats else None
		self.max_cpu_p = self.process_stats.max_cpu_p if collect_stats else 0.0

	def save_stats_plots(self):
		self.process_stats.save_plots()

		ct_stats = self.jitserver_output.container_stats()
		if ct_stats is not None:
			ct_stats.save_plots()


class DBRunResult:
	def __init__(self, *args):
		self.db_output = DBOutput(*args)
		rusage = self.db_output.container_rusage()
		self.peak_mem = rusage.peak_mem
		self.cpu_time = rusage.cpu_time()

	def save_stats_plots(self):
		self.db_output.container_stats().save_plots()


def result_fields(config, details=False):
	fields = [
		("start_time", "Start time, sec"),
		("peak_mem", "Memory usage, MB"),
	]

	if details:
		fields.extend((
			("n_comps", "Number of methods compiled"),
			("jit_cpu_time", "JIT CPU time, sec"),
			("cpu_time", "CPU time, sec"),
		))

	if config.run_jmeter:
		fields.extend((
			("requests", "Requests served"),
			("warmup_time", "Warm-up time, sec"),
			("full_warmup_time", "Full warm-up time, sec"),
			("peak_throughput", "Peak throughput, req/sec"),
		))
		if details:
			fields.append(("warmup_avg_throughput", "Warm-up avg throughput, sec"))

	return fields

# field, label, log, cut
vlog_cdf_fields = (
	("comp_starts", "Compilation start time, ms", False, None),
	("queue_sizes", "Compilation queue size", False, None),
	("comp_times", "Compilation time, ms", True, 0.99),
	("queue_times", "Total queuing time, ms", True, 0.99),
)


class ApplicationInstanceResult:
	def __init__(self, bench, config, experiment, instance_id, details=False, *, actual_experiment=None, **kwargs):
		self.config = config
		self.experiment = experiment
		self.actual_experiment = actual_experiment or experiment

		self.results = [ApplicationRunResult(bench, config, experiment, instance_id, r,
		                                     actual_experiment=actual_experiment, **kwargs)
		                for r in range(config.n_runs)]

		self.values = {}
		for f in result_fields(config, details):
			add_mean_stdev(self, self.results, f[0])
		add_min_max(self, self.results, "peak_mem")

		if config.run_jmeter:
			self.interval = mean((r.warmup_data.throughput_data[-1][0] / (len(r.warmup_data.throughput_data) - 1))
			                     if r.warmup_data is not None else 0.0 for r in self.results)

	def aligned_throughput_df(self, run_id):
		data = self.results[run_id].warmup_data.throughput_data
		return pd.DataFrame([d[1] for d in data], index=[self.interval * i for i in range(len(data))],
		                    columns=[experiment_names[self.actual_experiment]])

	def avg_throughput_df_groups(self):
		return pd.concat(self.aligned_throughput_df(r) for r in range(self.config.n_runs)).groupby(level=0)

	def plot_peak_throughput_warmup_time(self, ax):
		c = "C{}".format(self.actual_experiment.value)

		m = self.values["peak_throughput_mean"]
		s = self.values["peak_throughput_stdev"]
		ax.hlines(m, 0, 1, transform=ax.get_yaxis_transform(), colors=c)
		ax.axhspan(m - s, m + s, alpha=throughput_alpha, color=c)

		m = self.values["warmup_time_mean"]
		s = self.values["warmup_time_stdev"]
		ax.vlines(m, 0, 1, transform=ax.get_xaxis_transform(), colors=c)
		ax.axvspan(m - s, m + s, alpha=throughput_alpha, color=c)

	def plot_all_throughput(self, ax):
		for r in range(self.config.n_runs):
			self.results[r].throughput_df().plot(
				ax=ax, color="C{}".format(self.actual_experiment.value), alpha=throughput_alpha, legend=False,
				marker=experiment_markers[self.actual_experiment], markevery=throughput_marker_interval
			)

		self.plot_peak_throughput_warmup_time(ax)

	def plot_avg_throughput(self, ax):
		groups = self.avg_throughput_df_groups()
		x_df = groups.mean()
		yerr_df = groups.std()
		c = "C{}".format(self.actual_experiment.value)

		x_df.plot(ax=ax, color=c, markevery=throughput_marker_interval,
		          marker=experiment_markers[self.actual_experiment])

		name = experiment_names[self.actual_experiment]
		ax.fill_between(x_df.index, (x_df - yerr_df)[name], (x_df + yerr_df)[name], color=c, alpha=throughput_alpha)

		self.plot_peak_throughput_warmup_time(ax)

	def save_stats_plots(self):
		for r in self.results:
			r.save_stats_plots()

	def plot_cdf(self, ax, field, log=False, cut=None):
		data = list(itertools.chain.from_iterable(getattr(r.vlog, field) for r in self.results))
		s = pd.Series(data).sort_values()
		if cut is not None:
			s = s[:math.floor(len(s) * cut)]
		e = self.actual_experiment.cdf_report_experiment()
		df = pd.DataFrame(np.linspace(0.0, 1.0, len(s)), index=s, columns=[experiment_names[e]])
		df.plot(ax=ax, logx=log, legend=False, color="C{}".format(e.value))


class JITServerInstanceResult:
	def __init__(self, benchmark, config, *args):
		self.results = [JITServerRunResult(benchmark, config, *args, r) for r in range(config.n_runs)]

		self.values = {}
		for f in ("peak_mem", "cpu_time", "jit_cpu_time", "jitserver_mem", "jitserver_cpu"):
			add_mean_stdev(self, self.results, f)
		add_min_max(self, self.results, "peak_mem")

		self.max_cpu_p = max(r.max_cpu_p for r in self.results)

	def save_stats_plots(self):
		for r in self.results:
			r.save_stats_plots()


class DBInstanceResult:
	def __init__(self, bench, config, *args):
		self.results = [DBRunResult(bench, config, *args, r) for r in range(config.n_runs)]

		self.values = {}
		for f in ("peak_mem", "cpu_time"):
			add_mean_stdev(self, self.results, f)
		add_min_max(self, self.results, "peak_mem")

	def save_stats_plots(self):
		for r in self.results:
			r.save_stats_plots()


def normalized_field_label(label):
	pos = label.find(", ")
	return (label if pos < 0 else label[:pos]) + " (normalized)"

def bar_plot_df(result, field):
	return pd.DataFrame({experiment_names[e]: [result.values[field][e]] for e in result.experiments}).iloc[0]


class SingleInstanceExperimentResult:
	def __init__(self, experiments, bench, config, details=False, normalized=False, **kwargs):
		self.experiments = experiments
		self.benchmark = bench.name()
		self.config = config
		self.details = details

		self.application_results = [ApplicationInstanceResult(bench, config, *e.to_single_instance(),
		                                                      details, actual_experiment=e, **kwargs)
		                            if e in experiments else None for e in Experiment]

		self.jitserver_results = [JITServerInstanceResult(self.benchmark, config, e.to_single_instance()[0], 0)
		                          if (e.is_jitserver() and e in experiments) else None for e in Experiment]

		self.db_results = ([DBInstanceResult(bench, config, e.to_single_instance()[0], 0)
		                    if e in experiments else None for e in Experiment]
		                   if bench.db_name() is not None else [])

		self.fields = result_fields(config, details)
		self.values = {}
		for f in self.fields:
			add_mean_stdev_lists(self, self.application_results, f[0], True)
		add_min_max_lists(self, self.application_results, "peak_mem", True)

		if normalized and (Experiment.LocalJIT in experiments):
			for f in self.fields:
				m = self.application_results[Experiment.LocalJIT].values[f[0] + "_mean"]
				self.values[f[0] + "_normalized_means"] = [v / m if v is not None else None
				                                           for v in self.values[f[0] + "_means"]]
				self.values[f[0] + "_normalized_stdevs"] = [v / m if v is not None else None
				                                           for v in self.values[f[0] + "_stdevs"]]
			self.fields.extend([(f[0] + "_normalized", normalized_field_label(f[1])) for f in self.fields])

		self.jitserver_max_cpu_p = max(r.max_cpu_p if r is not None else 0.0 for r in self.jitserver_results)

	def summary(self):
		s = summary(self.values, self.fields, self.experiments)
		if self.jitserver_max_cpu_p:
			s += "JITServer max CPU usage: {}%\n".format(self.jitserver_max_cpu_p)
		return s

	def save_bar_plot(self, field, ymax=None):
		ax = bar_plot_df(self, field[0] + "_means").plot.bar(yerr=bar_plot_df(self, field[0] + "_stdevs"),
		                 rot=0, ylim=(0, ymax))
		ax.set(ylabel=field[1])
		save_plot(ax, field[0], self.benchmark, self.config)

	def save_all_bar_plots(self, limits=None):
		for f in self.fields:
			self.save_bar_plot(f, (limits or {}).get(f[0]))

	def save_throughput_plot(self, ax, name, ymax=None):
		ax.set(xlabel="Time, sec", ylabel="Throughput, req/sec")
		ax.set_xlim(0)
		ax.set_ylim(0, ymax)
		save_plot(ax, "throughput_{}".format(name), self.benchmark, self.config)

	def save_run_throughput_plots(self, ymax=None):
		for r in range(self.config.n_runs):
			ax = plt.gca()
			for e in self.experiments:
				self.application_results[e].results[r].plot_run_throughput(ax)
			self.save_throughput_plot(ax, "run_{}".format(r), ymax)

	def save_all_throughput_plot(self, ymax=None):
		ax = plt.gca()
		for e in self.experiments:
			self.application_results[e].plot_all_throughput(ax)

		ax.legend((matplotlib.lines.Line2D([0], [0], color="C{}".format(e.value)) for e in self.experiments),
		          (experiment_names[e] for e in self.experiments))
		self.save_throughput_plot(ax, "all", ymax)

	def save_avg_throughput_plot(self, ymax=None):
		ax = plt.gca()
		for e in self.experiments:
			self.application_results[e].plot_avg_throughput(ax)
		self.save_throughput_plot(ax, "avg", ymax)

	def save_stats_plots(self):
		for r in (self.application_results + self.jitserver_results + self.db_results):
			if r is not None:
				r.save_stats_plots()

	def save_cdf_plot(self, field, label, log=False, cut=None, legends=None):
		ax = plt.gca()
		for e in self.experiments:
			self.application_results[e].plot_cdf(ax, field, log, cut)
		ax.set(xlabel=label + (" (log scale)" if log else ""), ylabel="CDF", title="")

		if (legends or {}).get(field, True):
			ax.legend((matplotlib.lines.Line2D([0], [0], color="C{}".format(e.cdf_report_experiment().value))
			          for e in self.experiments), (experiment_names_single[e] for e in self.experiments))

		name = field + ("_log" if log else "") + ("_cut" if cut is not None else "")
		save_plot(ax, name, self.benchmark, self.config)

	def save_results(self, limits=None, legends=None, cdf_plots=False):
		save_summary(self.summary(), self.benchmark, self.config)

		if self.details:
			self.save_all_bar_plots(limits)

			if self.config.run_jmeter:
				ymax = (limits or {}).get("peak_throughput")
				self.save_run_throughput_plots(ymax)
				self.save_all_throughput_plot(ymax)
				self.save_avg_throughput_plot(ymax)

			if self.config.collect_stats:
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
				for r in self.application_results[e].results:
					s += r.vlog.dups_summary()
			if s:
				save_summary(s, self.benchmark, self.config, name="dups")


class SingleInstanceAllExperimentsResult:
	def __init__(self, experiments, bench, mode, configs, config_names, details=False, kwargs_list=None):
		self.experiments = experiments
		self.benchmark = bench.name()
		self.mode = mode
		self.configs = configs
		self.config_names = config_names
		self.warmup = configs[0].run_jmeter

		self.results = [SingleInstanceExperimentResult(experiments, bench, configs[i], details,
		                                               **((kwargs_list[i] or {}) if kwargs_list is not None else {}))
		                for i in range(len(configs))]

	def get_df(self, field, suffix):
		data = {
			experiment_names_single[e]: [self.results[i].values[field[0] + suffix][e] for i in range(len(self.configs))]
			for e in self.experiments
		}
		return pd.DataFrame(data, index=self.config_names)

	def save_bar_plot(self, field, ymax=None, legend=True, dry_run=False):
		ax = self.get_df(field, "_means").plot.bar(yerr=self.get_df(field, "_stdevs"),
		                 rot=0, ylim=(0, ymax), legend=legend)
		ax.set(xlabel="{} {}: Container size".format(benchmark_full_names[self.benchmark], self.mode), ylabel=field[1])
		result = ax.get_ylim()[1]

		if dry_run:
			plt.close(ax.get_figure())
		else:
			save_plot(ax, "single_{}_{}_{}".format("full" if self.warmup else "start",
			          self.mode, field[0]), self.benchmark)
		return result

	def save_all_bar_plots(self, limits=None, legends=None, dry_run=False):
		result = {}
		for f in self.results[0].fields:
			result[f[0]] = self.save_bar_plot(f, (limits or {}).get(f[0]), (legends or {}).get(f[0], True), dry_run)
		return result

	def save_results(self, limits=None, legends=None, dry_run=False):
		return self.save_all_bar_plots(limits, legends, dry_run)


class ApplicationAllInstancesResult:
	def __init__(self, bench, config, experiment, details=False, *, n_instances=None, **kwargs):
		self.config = config
		self.experiment = experiment

		self.results = [[ApplicationRunResult(bench, config, experiment, i, r, **kwargs)
		                 for r in range(config.n_runs)] for i in range(n_instances or config.n_instances)]
		self.all_results = list(itertools.chain.from_iterable(self.results))

		self.fields = result_fields(config, details)
		self.values = {}
		for f in self.fields:
			add_mean_stdev(self, self.all_results, f[0])
		add_min_max(self, self.all_results, "peak_mem")

		if config.run_jmeter:
			self.interval = mean((r.warmup_data.throughput_data[-1][0] / (len(r.warmup_data.throughput_data) - 1))
			                     if r.warmup_data.throughput_data is not None else 0.0 for r in self.all_results)

	def aligned_throughput_df(self, instance_id, run_id):
		data = self.results[instance_id][run_id].warmup_data.throughput_data
		return pd.DataFrame([d[1] for d in data], index=[self.interval * i for i in range(len(data))],
		                    columns=[experiment_names[self.experiment]])

	def avg_throughput_df_groups(self):
		return pd.concat(self.aligned_throughput_df(i, r) for i in range(self.config.n_instances)
		                 for r in range(self.config.n_runs)).groupby(level=0)

	def plot_peak_throughput_warmup_time(self, ax):
		c = "C{}".format(self.experiment.value)

		m = self.values["peak_throughput_mean"]
		s = self.values["peak_throughput_stdev"]
		ax.hlines(m, 0, 1, transform=ax.get_yaxis_transform(), colors=c)
		ax.axhspan(m - s, m + s, alpha=throughput_alpha, color=c)

		m = self.values["warmup_time_mean"]
		s = self.values["warmup_time_stdev"]
		ax.vlines(m, 0, 1, transform=ax.get_xaxis_transform(), colors=c)
		ax.axvspan(m - s, m + s, alpha=throughput_alpha, color=c)

	def plot_all_throughput(self, ax):
		for i in range(self.config.n_instances):
			for r in range(self.config.n_runs):
				self.results[i][r].throughput_df().plot(
					ax=ax, color="C{}".format(self.experiment.value), alpha=throughput_alpha, legend=False,
					marker=experiment_markers[self.experiment], markevery=throughput_marker_interval
				)

		self.plot_peak_throughput_warmup_time(ax)

	def plot_avg_throughput(self, ax):
		groups = self.avg_throughput_df_groups()
		x_df = groups.mean()
		yerr_df = groups.std()
		c = "C{}".format(self.experiment.value)

		x_df.plot(ax=ax, color=c, markevery=throughput_marker_interval, marker=experiment_markers[self.experiment])
		name = experiment_names[self.experiment]
		ax.fill_between(x_df.index, (x_df - yerr_df)[name], (x_df + yerr_df)[name], color=c, alpha=throughput_alpha)

		self.plot_peak_throughput_warmup_time(ax)

	def plot_cdf(self, ax, field, log=False, cut=None):
		data = list(itertools.chain.from_iterable(getattr(r.vlog, field) for r in self.all_results))
		s = pd.Series(data).sort_values()
		if cut is not None:
			s = s[:math.floor(len(s) * cut)]
		df = pd.DataFrame(np.linspace(0.0, 1.0, len(s)), index=s, columns=[experiment_names[self.experiment]])
		df.plot(ax=ax, logx=log, legend=False, color="C{}".format(self.experiment.value))


class JITServerAllInstancesResult:
	def __init__(self, benchmark, config, experiment):
		self.results = [[JITServerRunResult(benchmark, config, experiment, i, r)
		                 for r in range(config.n_runs)] for i in range(config.n_jitservers)]
		self.all_results = list(itertools.chain.from_iterable(self.results))

		self.values = {}
		for f in ("peak_mem", "cpu_time", "jit_cpu_time", "jitserver_mem", "jitserver_cpu"):
			add_mean_stdev(self, self.all_results, f)
		add_min_max(self, self.all_results, "peak_mem")

		self.max_cpu_p = max(r.max_cpu_p for r in itertools.chain.from_iterable(self.results))

	def save_stats_plots(self):
		for r in self.all_results:
			r.save_stats_plots()


class DBAllInstancesResult:
	def __init__(self, bench, config, experiment):
		self.results = [[DBRunResult(bench, config, experiment, i, r)
		                 for r in range(config.n_runs)] for i in range(config.n_dbs)]
		self.all_results = list(itertools.chain.from_iterable(self.results))

		self.values = {}
		for f in ("peak_mem", "cpu_time"):
			add_mean_stdev(self, self.all_results, f)
		add_min_max(self, self.all_results, "peak_mem")

	def save_stats_plots(self):
		for r in self.all_results:
			r.save_stats_plots()


class ScaleExperimentResult:
	def __init__(self, experiments, bench, config, details=False, **kwargs):
		self.experiments = experiments
		self.benchmark = bench.name()
		self.config = config
		self.details = details

		self.application_results = [ApplicationAllInstancesResult(bench, config, e, details, **kwargs)
		                            if e in experiments else None for e in Experiment]

		self.jitserver_results = [JITServerAllInstancesResult(self.benchmark, config, e)
		                          if (e.is_jitserver() and e in experiments) else None for e in Experiment]

		self.db_results = ([DBAllInstancesResult(bench, config, e) if e in experiments else None for e in Experiment]
		                   if bench.db_name() is not None else [])

		self.fields = result_fields(config, details)
		self.values = {}

		if config.run_jmeter and (Experiment.LocalJIT in experiments):
			self.fields.append(("full_warmup_time_normalized", "Full warm-up time"))
			for e in experiments:
				r = self.application_results
				m = r[Experiment.LocalJIT].values["full_warmup_time_mean"]
				r[e].values["full_warmup_time_normalized_mean"] = r[e].values["full_warmup_time_mean"] / m
				r[e].values["full_warmup_time_normalized_stdev"] = r[e].values["full_warmup_time_stdev"] / m

		for f in self.fields:
			add_mean_stdev_lists(self, self.application_results, f[0], True)
		add_min_max_lists(self, self.application_results, "peak_mem", True)

		if details:
			all_results = [
				(self.application_results[e].results + (self.jitserver_results[e].results if e.is_jitserver() else []))
				if e in experiments else None for e in Experiment
			]

			total_fields = [
				("peak_mem", "Total memory usage, MB"),
				("jit_cpu_time", "Total JIT CPU time, sec"),
				("cpu_time", "Total CPU time, sec"),
				("jitserver_mem", "Total JITServer memory usage, MB"),
				("jitserver_cpu", "Total JITServer CPU time, sec"),
				("data_transferred", "Data transferred, MB")
			]

			self.fields.extend(("total_" + f[0], f[1]) for f in total_fields)
			for f in total_fields:
				add_total_mean_stdev_lists(self, all_results, f[0])

		self.jitserver_max_cpu_p = max(r.max_cpu_p if r is not None else 0.0 for r in self.jitserver_results)

	def summary(self):
		s = summary(self.values, self.fields, self.experiments)
		if self.jitserver_max_cpu_p:
			s += "JITServer max CPU usage: {}%\n".format(self.jitserver_max_cpu_p)
		return s

	def save_bar_plot(self, field, ymax=None):
		ax = bar_plot_df(self, field[0] + "_means").plot.bar(yerr=bar_plot_df(self, field[0] + "_stdevs"),
		                                                     rot=0, ylim=(0, ymax))
		ax.set(ylabel=field[1])
		save_plot(ax, field[0], self.benchmark, self.config)

	def save_all_bar_plots(self, limits=None):
		for f in self.fields:
			self.save_bar_plot(f, (limits or {}).get(f[0]))

	def save_throughput_plot(self, ax, name, ymax=None):
		ax.set(xlabel="Time, sec", ylabel="Throughput, req/sec")
		ax.set_xlim(0, ymax)
		ax.set_ylim(0, ymax)
		save_plot(ax, "throughput_{}".format(name), self.benchmark, self.config)

	def save_all_throughput_plot(self, ymax=None):
		ax = plt.gca()
		for e in self.experiments:
			self.application_results[e].plot_all_throughput(ax)

		ax.legend((matplotlib.lines.Line2D([0], [0], color="C{}".format(e.value)) for e in self.experiments),
		          (experiment_names[e] for e in self.experiments))
		self.save_throughput_plot(ax, "all", ymax)

	def save_avg_throughput_plot(self, ymax=None):
		ax = plt.gca()
		for e in self.experiments:
			self.application_results[e].plot_avg_throughput(ax)
		self.save_throughput_plot(ax, "avg", ymax)

	def save_stats_plots(self):
		for r in (self.jitserver_results + self.db_results):
			if r is not None:
				r.save_stats_plots()

	def save_cdf_plot(self, field, label, log=False, cut=None, legends=None):
		ax = plt.gca()
		for e in self.experiments:
			self.application_results[e].plot_cdf(ax, field, log, cut)
		ax.set(xlabel=label + (" (log scale)" if log else ""), ylabel="CDF", title="")

		if (legends or {}).get(field, True):
			ax.legend((matplotlib.lines.Line2D([0], [0], color="C{}".format(e.value)) for e in self.experiments),
			          (experiment_names_multi[e] for e in self.experiments))

		name = field + ("_log" if log else "") + ("_cut" if cut is not None else "")
		save_plot(ax, name, self.benchmark, self.config)

	def save_results(self, limits=None, legends=None, cdf_plots=False):
		save_summary(self.summary(), self.benchmark, self.config)

		if self.details:
			self.save_all_bar_plots(limits)

			if self.config.run_jmeter:
				ymax = (limits or {}).get("peak_throughput")
				self.save_all_throughput_plot(ymax)
				self.save_avg_throughput_plot(ymax)

			if self.config.collect_stats:
				self.save_stats_plots()

		if self.config.jitserver_config.client_vlog and cdf_plots:
			for field, label, log, cut in vlog_cdf_fields:
				self.save_cdf_plot(field, label, legends=legends)
				if log:
					self.save_cdf_plot(field, label, log=True, legends=legends)
				if cut is not None:
					self.save_cdf_plot(field, label, cut=cut, legends=legends)


class ScaleAllExperimentsResult:
	def __init__(self, experiments, bench, configs, details=False, kwargs_list=None):
		self.experiments = experiments
		self.benchmark = bench.name()
		self.warmup = configs[0].run_jmeter

		self.results = [ScaleExperimentResult(experiments, bench, configs[i], details, keep_throughput_data=False,
		                                      **((kwargs_list[i] or {}) if kwargs_list is not None else {}))
		                for i in range(len(configs))]

		self.fields = result_fields(configs[0], details)
		if self.warmup and (Experiment.LocalJIT in experiments):
			self.fields.append(("full_warmup_time_normalized", "Full warm-up time"))

		if details:
			self.fields.extend((
				("total_peak_mem", "Total memory usage, MB"),
				("total_jit_cpu_time", "Total JIT CPU time, sec"),
				("total_cpu_time", "Total CPU time, sec"),
				("total_jitserver_mem", "Total JITServer memory usage, MB"),
				("total_jitserver_cpu", "Total JITServer CPU time, sec"),
			))

	def get_df(self, results, field, suffix, name_suffix=None):
		return pd.DataFrame({experiment_names_multi[e] + (name_suffix or ""):
		                     [r.values[field[0] + suffix][e] for r in results] for e in self.experiments},
		                    index=[r.config.n_instances for r in results])

	def save_line_plot(self, field, ymax=None, legend=True):
		name = "scale_{}_{}".format("full" if self.warmup else "start", field[0])
		ax = self.get_df(self.results, field, "_means").plot(yerr=self.get_df(self.results, field, "_stdevs"),
		                                                     ylim=(0, ymax), xlim=(0, None), legend=legend)

		if field[0] == "full_warmup_time_normalized":
			ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.25))
		ax.set(xlabel="{}: Number of instances".format(benchmark_full_names[self.benchmark]), ylabel=field[1])
		for i, line in enumerate(ax.get_lines()):
			line.set_marker(experiment_markers[i])
		if legend:
			ax.legend()

		save_plot(ax, name, self.benchmark)

	def save_all_line_plots(self, limits=None, legends=None):
		for f in self.fields:
			self.save_line_plot(f, (limits or {}).get(f[0]), (legends or {}).get(f[0], True))

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
		data = {experiment_names_single[e]: [r.values[field[0] + suffix][e] for r in self.results]
		        for e in self.experiments}

		if Experiment.LocalJIT in self.experiments:
			values = [getattr(r.application_results[Experiment.LocalJIT].results[i], field[0])
			          for r in self.results for i in range(r.config.n_runs)]
			val = f(values)
			data[experiment_names_single[Experiment.LocalJIT]] = [val for r in self.results]

		return pd.DataFrame(data, index=[r.latency for r in self.results])

	def save_line_plot(self, field, ymax=None, legend=True):
		ax = self.get_df(field, "_means", mean).plot(yerr=self.get_df(field, "_stdevs", stdev),
		                                             ylim=(0, ymax), xlim=(0, None), legend=legend)
		ax.set(xlabel="{}: Latency, microsec".format(benchmark_full_names[self.benchmark]), ylabel=field[1])
		for i, line in enumerate(ax.get_lines()):
			line.set_marker(experiment_markers[i])
		if legend:
			ax.legend()
		name = "latency_{}_{}".format("full" if self.warmup else "start", field[0])
		save_plot(ax, name, self.benchmark)

	def save_all_line_plots(self, limits=None, legends=None):
		for f in self.results[0].fields:
			self.save_line_plot(f, (limits or {}).get(f[0]), (legends or {}).get(f[0], True))

	def save_results(self, limits=None, legends=None):
		self.save_all_line_plots(limits, legends)


class DensityExperimentResult:
	def total_jvm_peak_mem(self, experiment, run_id):
		timestamps = []
		buf = datetime.timedelta(seconds=1.0)

		for i in range(self.total_instances):
			r = self.application_results[experiment].results[i][run_id]
			start = r.application_output.docker_ts - buf
			stop = r.application_output.stop_ts + buf
			timestamps.extend(((start, 1, r.peak_mem), (stop, -1, r.peak_mem)))

		timestamps.sort(key=lambda t: t[0])

		total = 0.0
		max_total = 0.0
		for t in timestamps:
			total += t[1] * t[2]
			max_total = max(max_total, total)

		return max_total / 1024.0 # GB

	def jitserver_peak_mem(self, experiment, run_id):
		return (sum(self.jitserver_results[experiment].results[i][run_id].peak_mem
		        for i in range(self.config.n_jitservers)) if experiment.is_jitserver() else 0.0) / 1024.0 # GB

	def __init__(self, experiments, bench, config, details=False, **kwargs):
		self.experiments = experiments
		self.benchmark = bench.name()
		self.config = config
		self.total_instances = config.n_instances * config.n_invocations

		self.application_results = [
			ApplicationAllInstancesResult(bench, config, e, details, n_instances=self.total_instances,
			                              keep_throughput_data=False, **kwargs)
			if e in experiments else None for e in Experiment
		]

		self.jitserver_results = [JITServerAllInstancesResult(self.benchmark, config, e)
		                          if (e.is_jitserver() and e in experiments) else None for e in Experiment]

		self.db_results = ([DBAllInstancesResult(bench, config, e) if e in experiments else None for e in Experiment]
		                   if bench.db_name() is not None else [])

		req_results = [self.application_results[e].results if e in experiments else None for e in Experiment]

		all_results = [
			(self.application_results[e].results + (self.jitserver_results[e].results if e.is_jitserver() else []))
			if e in experiments else None for e in Experiment
		]

		self.fields = result_fields(config, details)
		self.values = {}
		for f in self.fields:
			add_mean_stdev_lists(self, self.application_results, f[0], True)

		self.fields.extend((
			("total_requests", "Total requests served"),
			("total_cpu_time", "Total CPU time, sec"),
			("total_jit_cpu_time", "Total JIT CPU time, sec"),
			("total_jvm_peak_mem", "Total JVM memory usage, GB"),
			("jitserver_peak_mem", "JITServer memory usage, GB"),
			("total_peak_mem", "Total mem. usage, GB"),
			("cpu_time_per_req", "CPU cost, msec/req"),
		))

		req_vals = add_total_mean_stdev_lists(self, req_results, "requests")
		cpu_vals = add_total_mean_stdev_lists(self, all_results, "cpu_time")
		jit_cpu_vals = add_total_mean_stdev_lists(self, all_results, "jit_cpu_time")

		total_jvm_peak_mem_vals = [[self.total_jvm_peak_mem(e, r) for r in range(config.n_runs)]
		                           if e in experiments else None for e in Experiment]
		jitserver_peak_mem_vals = [[self.jitserver_peak_mem(e, r) for r in range(config.n_runs)]
		                           if e in experiments else None for e in Experiment]
		total_peak_mem_vals = [
			[total_jvm_peak_mem_vals[e][r] + jitserver_peak_mem_vals[e][r] for r in range(config.n_runs)]
			if e in experiments else None for e in Experiment
		]

		self.values["total_jvm_peak_mem_means"] = [mean([total_jvm_peak_mem_vals[e][r] for r in range(config.n_runs)])
		                                           if e in experiments else None for e in Experiment]
		self.values["total_jvm_peak_mem_stdevs"] = [stdev([total_jvm_peak_mem_vals[e][r] for r in range(config.n_runs)])
		                                            if e in experiments else None for e in Experiment]
		self.values["jitserver_peak_mem_means"] = [mean([jitserver_peak_mem_vals[e][r] for r in range(config.n_runs)])
		                                           if e in experiments else None for e in Experiment]
		self.values["jitserver_peak_mem_stdevs"] = [stdev([jitserver_peak_mem_vals[e][r] for r in range(config.n_runs)])
		                                            if e in experiments else None for e in Experiment]
		self.values["total_peak_mem_means"] = [mean([total_peak_mem_vals[e][r] for r in range(config.n_runs)])
		                                       if e in experiments else None for e in Experiment]
		self.values["total_peak_mem_stdevs"] = [stdev([total_peak_mem_vals[e][r] for r in range(config.n_runs)])
		                                        if e in experiments else None for e in Experiment]

		cpu_time_per_req_vals = [[1000 * cpu_vals[e][r] / req_vals[e][r] for r in range(config.n_runs)] # msec/req
		                         if e in experiments else None for e in Experiment]
		self.values["cpu_time_per_req_means"] = [mean([cpu_time_per_req_vals[e][r] for r in range(config.n_runs)])
		                                         if e in experiments else None for e in Experiment]
		self.values["cpu_time_per_req_stdevs"] = [stdev([cpu_time_per_req_vals[e][r] for r in range(config.n_runs)])
		                                          if e in experiments else None for e in Experiment]

		jit_cpu_time_per_req_vals = [
			[1000 * jit_cpu_vals[e][r] / req_vals[e][r] for r in range(config.n_runs)] # msec/req
			if e in experiments else None for e in Experiment
		]
		self.values["jit_cpu_time_per_req_means"] = [
			mean([jit_cpu_time_per_req_vals[e][r] for r in range(config.n_runs)])
			if e in experiments else None for e in Experiment
		]
		self.values["jit_cpu_time_per_req_stdevs"] = [
			stdev([jit_cpu_time_per_req_vals[e][r] for r in range(config.n_runs)])
			if e in experiments else None for e in Experiment
		]

		self.jitserver_max_cpu_p = max(r.max_cpu_p if r is not None else 0.0 for r in self.jitserver_results)

	def summary(self):
		s = summary(self.values, self.fields, self.experiments)
		if self.jitserver_max_cpu_p:
			s += "JITServer max CPU usage: {}%\n".format(self.jitserver_max_cpu_p)
		return s

	def save_stats_plots(self):
		for r in (self.jitserver_results + self.db_results):
			if r is not None:
				r.save_stats_plots()

	def save_results(self):
		save_summary(self.summary(), self.benchmark, self.config)
		if self.config.collect_stats:
			self.save_stats_plots()


class DensityAllExperimentsResult:
	def __init__(self, experiments, bench, configs, details=False, kwargs_list=None):
		self.experiments = experiments
		self.benchmark = bench.name()
		self.configs = configs

		self.results = [DensityExperimentResult(experiments, bench, configs[i], details,
		                                        **((kwargs_list[i] or {}) if kwargs_list is not None else {}))
		                for i in range(len(configs))]

	def get_df(self, field, suffix):
		data = {
			experiment_names_multi[e]: [self.results[i].values[field[0] + suffix][e] for i in range(len(self.configs))]
			for e in self.experiments
		}
		index = ["{:g} min".format(c.jmeter_config.duration / 60) for c in self.configs]
		return pd.DataFrame(data, index=index)

	def save_bar_plot(self, field, ymax=None, legend=True, dry_run=False, overlay_field=None):
		ax = self.get_df(field, "_means").plot.bar(yerr=self.get_df(field, "_stdevs"), rot=0, ylim=(0, ymax),
		                                           legend=legend and overlay_field is None)

		if overlay_field is not None:
			self.get_df(overlay_field, "_means").plot.bar(
				ax=ax, yerr=self.get_df(overlay_field, "_stdevs"), rot=0, ylim=(0, ymax),
				legend=False, edgecolor="black", linewidth=0.5, hatch="xxxxxxxx"
			)
			ax.legend(
				[matplotlib.patches.Patch(color="C{}".format(e.value)) for e in self.experiments] +
				[matplotlib.patches.Patch(edgecolor="black", facecolor="none", linewidth=0.5, hatch="xxxxxxxx")],
				[experiment_names_multi[e] for e in self.experiments] + [overlay_field[1]]
			)

		ax.set(xlabel="{}: Application lifespan".format(benchmark_full_names[self.benchmark]), ylabel=field[1])
		result = ax.get_ylim()[1]

		if dry_run:
			plt.close(ax.get_figure())
		else:
			save_plot(ax, "density_{}_{}".format("scc" if self.configs[0].application_config.populate_scc
			          else "noscc", field[0]), self.benchmark)
		return result

	def save_all_bar_plots(self, limits=None, legends=None, dry_run=False, overlays=False):
		result = {}

		for f in self.results[0].fields:
			overlay_field = None
			if f[0] == "total_peak_mem":
				overlay_field = ("jitserver_peak_mem", "JITServer memory")
			elif f[0] == "cpu_time_per_req":
				overlay_field = ("jit_cpu_time_per_req", "JIT CPU time")

			result[f[0]] = self.save_bar_plot(f, (limits or {}).get(f[0]), (legends or {}).get(f[0], True),
			                                  dry_run, overlay_field if overlays else None)

		return result

	def save_results(self, limits=None, legends=None, dry_run=False, overlays=False):
		return self.save_all_bar_plots(limits, legends, dry_run, overlays)
