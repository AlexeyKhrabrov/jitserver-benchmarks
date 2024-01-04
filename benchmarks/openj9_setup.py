#!/usr/bin/env python3

import argparse
import getpass

import openj9
import remote
import util


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("hosts_file")
	parser.add_argument("jdk_dir")
	parser.add_argument("jdk_ver", type=int)

	parser.add_argument("-p", "--prereqs", action="store_true")
	parser.add_argument("-c", "--configure", action="store_true")
	parser.add_argument("-d", "--debug", action="store_true")
	parser.add_argument("-v", "--verbose", action="store_true")

	args = parser.parse_args()

	hosts = [openj9.OpenJ9Host(*h) for h in remote.load_hosts(args.hosts_file)]
	cluster = openj9.OpenJ9Cluster(hosts)
	util.verbose = args.verbose

	if args.prereqs:
		#NOTE: assuming same credentials for all hosts
		passwd = getpass.getpass()
		cluster.check_sudo_passwd(passwd)
		cluster.openj9_prereqs(passwd=passwd)

	cluster.openj9_setup(args.jdk_dir, args.jdk_ver, configure=args.configure, debug=args.debug)


if __name__ == "__main__":
	main()
