#!/usr/bin/env python3

import argparse
import getpass

import daytrader
import remote
import util


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("hosts_file")
	parser.add_argument("db2_installer_path", nargs="?")

	parser.add_argument("-p", "--prereqs", action="store_true")
	parser.add_argument("-s", "--scripts-only", action="store_true")
	parser.add_argument("-c", "--clean", action="store_true")
	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument("-d", "--db2", action="store_true")
	parser.add_argument("-t", "--tune", action="store_true")

	args = parser.parse_args()

	hosts = [daytrader.DayTraderHost(*h) for h in remote.load_hosts(args.hosts_file)]
	cluster = remote.RemoteCluster(hosts)
	util.verbose = args.verbose

	if args.prereqs:
		#NOTE: assuming same credentials for all hosts
		passwd = getpass.getpass()
		cluster.check_sudo_passwd(passwd)

		cluster.for_each(daytrader.DayTraderHost.benchmark_prereqs,
		                 passwd=passwd, parallel=passwd is not None)

	cluster.for_each(
		daytrader.DayTraderHost.benchmark_setup,
		args.db2_installer_path, scripts_only=args.scripts_only,
		clean=args.clean, build_db2=args.db2, tune=args.tune, parallel=True
	)


if __name__ == "__main__":
	main()
