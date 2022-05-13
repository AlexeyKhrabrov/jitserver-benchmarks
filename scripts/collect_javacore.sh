#!/bin/bash

set -e -u -o pipefail


usage_str="\
Usage: ${0} name pid interval path"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


if (( $# < 4 )); then usage; fi

name="${1}"
pid="${2}"
interval="${3}" # seconds
path="${4}"


for (( i = 0; ; i++ )); do
	kill -3 "-${pid}"
	sleep 1
	docker cp "${name}:/output/javacore.txt" "${path}/javacore_${i}.txt"
	docker exec "${name}" rm -f "/output/javacore.txt"
	sleep "$((interval - 1))"
done
