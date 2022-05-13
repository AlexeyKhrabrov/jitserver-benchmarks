#!/bin/bash

set -e -u -o pipefail


usage_str="\
Usage: ${0} name [sleep_time] [attempts]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


if (( $# < 1 )); then usage; fi

name="${1}"
sleep_time="${2:-0}"
attempts="${3:-0}"


for (( i = 0; i < attempts || attempts == 0; i++ )); do
	if [[ "$(docker container inspect -f '{{.State.Running}}' "${name}")" == "true" ]]; then
		break
	fi
	if (( i == attempts - 1 )); then
		echo "Container ${name} is not running" 1>&2
		exit 1
	fi
	sleep "${sleep_time}"
done
