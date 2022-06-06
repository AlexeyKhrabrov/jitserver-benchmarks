#!/usr/bin/env python3

import argparse
import getpass

import acmeair
import remote
import util


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("hosts_file")

	parser.add_argument("-p", "--prereqs", action="store_true")
	parser.add_argument("-s", "--scripts-only", action="store_true")
	parser.add_argument("-c", "--clean", action="store_true")
	parser.add_argument("-S", "--sudo", action="store_true")
	parser.add_argument("-v", "--verbose", action="store_true")

	args = parser.parse_args()

	hosts = [acmeair.AcmeAirHost(*h) for h in remote.load_hosts(args.hosts_file)]
	cluster = remote.RemoteCluster(hosts)
	util.verbose = args.verbose

	#NOTE: assuming same credentials for all hosts
	passwd = None
	if args.prereqs or args.sudo:
		passwd = getpass.getpass()
		cluster.check_sudo_passwd(passwd)

	if args.prereqs:
		cluster.for_each(acmeair.AcmeAirHost.benchmark_prereqs,
		                 passwd=passwd, parallel=True)

	cluster.for_each(acmeair.AcmeAirHost.benchmark_setup,
	                 scripts_only=args.scripts_only, clean=args.clean,
	                 sudo=args.sudo, passwd=passwd, parallel=True)


if __name__ == "__main__":
	main()
