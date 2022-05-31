#!/usr/bin/env python3

import argparse
import getpass

import acmeair
import daytrader
import petclinic
import remote
import util


bench_cls = {
	"acmeair": acmeair.AcmeAir,
	"daytrader": daytrader.DayTrader,
	"petclinic": petclinic.PetClinic
}


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("benchmark")
	parser.add_argument("hosts_file")

	parser.add_argument("-v", "--verbose", action="store_true")
	parser.add_argument("-s", "--sudo", action="store_true")
	parser.add_argument("-S", "--stdin-passwd", action="store_true")

	args = parser.parse_args()

	bench = bench_cls[args.benchmark]()
	hosts = [bench.new_host(*h) for h in remote.load_hosts(args.hosts_file)]
	cluster = remote.RemoteCluster(hosts)

	#NOTE: assuming same credentials for all hosts
	passwd = None
	if args.sudo:
		if args.stdin_passwd:
			passwd = input("Password: ")
			print()
		else:
			passwd = getpass.getpass()

	util.verbose = args.verbose

	if passwd is not None:
		cluster.check_sudo_passwd(passwd)
	cluster.for_each(lambda h: h.full_cleanup(passwd=passwd), parallel=True)


if __name__ == "__main__":
	main()
