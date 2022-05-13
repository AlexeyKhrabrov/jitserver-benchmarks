#!/usr/bin/env python3

import argparse
import getpass

import remote
import util


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("hosts_file")

	parser.add_argument("-s", "--storage-addr")
	parser.add_argument("-v", "--verbose", action="store_true")

	args = parser.parse_args()

	hosts = [remote.RemoteHost(*h) for h in remote.load_hosts(args.hosts_file)]
	cluster = remote.RemoteCluster(hosts)
	util.verbose = args.verbose

	#NOTE: assuming same credentials for all hosts
	passwd = getpass.getpass()

	cluster.ssh_setup(passwd=passwd)

	if args.storage_addr is not None:
		cluster.check_sudo_passwd(passwd)
		cluster.storage_setup(args.storage_addr, passwd=passwd)


if __name__ == "__main__":
	main()
