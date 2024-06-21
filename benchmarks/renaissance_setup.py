#!/usr/bin/env python3

import argparse
import getpass

import renaissance
import remote
import util


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("hosts_file")
	parser.add_argument("jdk_ver", type=int)

	parser.add_argument("-p", "--prereqs", action="store_true")
	parser.add_argument("-s", "--scripts-only", action="store_true")
	parser.add_argument("-c", "--clean", action="store_true")
	parser.add_argument("-P", "--prune", action="store_true")
	parser.add_argument("-b", "--buildkit", action="store_true")
	parser.add_argument("-S", "--sudo", action="store_true")
	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument("-u", "--update", action="store_true")

	args = parser.parse_args()

	hosts = [renaissance.RenaissanceHost(*h) for h in remote.load_hosts(args.hosts_file)]
	cluster = remote.RemoteCluster(hosts)
	util.verbose = args.verbose

	#NOTE: assuming same credentials for all hosts
	passwd = None
	if args.prereqs or args.sudo:
		passwd = getpass.getpass()
		cluster.check_sudo_passwd(passwd)

	if args.prereqs:
		cluster.for_each(renaissance.RenaissanceHost.benchmark_prereqs, passwd=passwd, parallel=True)

	cluster.for_each(renaissance.RenaissanceHost.benchmark_setup, args.jdk_ver, update=args.update,
	                 scripts_only=args.scripts_only, clean=args.clean, prune=args.prune,
	                 buildkit=args.buildkit, sudo=args.sudo, passwd=passwd, parallel=True)


if __name__ == "__main__":
	main()
