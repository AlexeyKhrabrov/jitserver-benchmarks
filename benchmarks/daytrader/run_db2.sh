#!/bin/bash

set -e -u -o pipefail


usage_str="\
Usage: ${0} instance_id [<docker args>]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


if (( $# < 1 )); then usage; fi

instance_id="${1}"
docker_args=("${@:2}")


name="db2_${instance_id}"
port="$((50000 + instance_id))"

exec docker run --name="${name}" --rm --init --cap-add=IPC_OWNER \
     -p "${port}:${port}" "${docker_args[@]}" "db2-daytrader" "${port}" "tradedb"
