import concurrent.futures
import itertools
import os
import os.path
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback


verbose = False

#NOTE: stdin is set to DEVNULL by default; use stdin=None to inherit from parent
def run(cmd, *, output=None, append=False, **kwargs):
	if "stdin" not in kwargs and ("input" not in kwargs or kwargs["input"] is None):
		kwargs["stdin"] = subprocess.DEVNULL

	if verbose:
		print("Running {}".format(cmd), flush=True)

	if output is not None:
		os.makedirs(os.path.dirname(output), exist_ok=True)
		with open(output, "a" if append else "w") as f:
			result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
			                        **kwargs)
	else:
		result = subprocess.run(cmd, stdout=subprocess.PIPE,
		                        stderr=subprocess.PIPE, **kwargs)

	if verbose:
		print("Finished {} with {}".format(cmd, result.returncode), flush=True)
	return result

#NOTE: stdin is set to DEVNULL by default; use stdin=None to inherit from parent
def start(cmd, *, output=None, append=False, **kwargs):
	if "stdin" not in kwargs:
		kwargs["stdin"] = subprocess.DEVNULL

	if output is not None:
		os.makedirs(os.path.dirname(output), exist_ok=True)
		with open(output, "a" if append else "w") as f:
			proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
			                        **kwargs)
	else:
		proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
		                        stderr=subprocess.PIPE, **kwargs)

	if verbose:
		print("Started {} pid {}".format(proc.args, proc.pid), flush=True)
	return proc

def wait(proc, *, check=False, expect_ret=0, timeout=None, kill_on_timeout=True,
         try_terminate=False, term_timeout=None):
	try:
		out, err = proc.communicate(timeout=timeout)

	except subprocess.TimeoutExpired:
		if kill_on_timeout:
			if try_terminate:
				proc.terminate()
				try:
					proc.wait(term_timeout)
				except:
					proc.kill()
			else:
				proc.kill()
			proc.wait()
		raise

	#NOTE: this is a workaround for https://bugs.python.org/issue35182
	except ValueError as e:
		proc.kill()
		proc.wait()
		if not str(e).startswith("Invalid file object: "):
			raise
		out, err = None, None

	except:
		proc.kill()
		proc.wait()
		raise

	ret = proc.poll()
	assert ret is not None
	if check and ret and (ret != expect_ret):
		raise subprocess.CalledProcessError(ret, proc.args, out, err)

	if verbose:
		print("Finished {} pid {} with {}".format(
		      proc.args, proc.pid, ret), flush=True)
	return subprocess.CompletedProcess(proc.args, ret, out, err)

def get_output(cmd):
	return run(cmd, check=True, universal_newlines=True).stdout


def print_exception():
	etype, e, tb = sys.exc_info()
	traceback.print_exception(etype, e, tb)

	if etype is subprocess.CalledProcessError:
		print("Command stdout: {}".format(e.stdout))
		print("Command stderr: {}".format(e.stderr))

	sys.stdout.flush()


def sleep(sleep_time):
	if sleep_time is not None:
		time.sleep(sleep_time)

# Returns the first non-None result of fn, or None if attempts were exhausted
def retry_loop(fn, attempts=None, sleep_time=None, pass_i_to_fn=False):
	for i in (range(attempts) if attempts is not None else itertools.count()):
		result = fn(i) if pass_i_to_fn else fn()
		if result is not None:
			return result
		if (attempts is None) or (i < attempts - 1):
			sleep(sleep_time)
	return None


def parallelize(fn, objs, *args, sleep_time=None, result_timeout=None,
                n_workers=None, multiprocess=False, **kwargs):
	if len(objs) == 0:
		return []
	if (len(objs) == 1) or (sleep_time == float("+inf")):
		return [fn(o, *args, **kwargs) for o in objs]

	executor = (concurrent.futures.ProcessPoolExecutor(n_workers) if multiprocess
	            else concurrent.futures.ThreadPoolExecutor(n_workers or len(objs)))
	with executor:
		futures = []
		for i in range(len(objs)):
			if i != 0:
				sleep(sleep_time)
			futures.append(executor.submit(fn, objs[i], *args, **kwargs))
		return [f.result(timeout=result_timeout) for f in futures]


def sigint_handler(signum, frame):
	for t in threading.enumerate():
		print("{} traceback (most recent call last):".format(t))
		traceback.print_stack(sys._current_frames()[t.ident], file=sys.stdout)
	sys.stdout.flush()

	signal.default_int_handler()

def set_sigint_handler():
	signal.signal(signal.SIGINT, sigint_handler)


def args_str(args):
	return " ".join(shlex.quote(s) for s in args)

def size_to_bytes(s, default='b', val_type=int):
	p_map = {'b': 0, 'k': 1, 'm': 2, 'g': 3, 't': 4}
	if s[-1].lower() in p_map:
		return val_type(s[:-1]) * (1024 ** p_map[s[-1].lower()])
	else:
		return val_type(s) * (1024 ** p_map[default])
