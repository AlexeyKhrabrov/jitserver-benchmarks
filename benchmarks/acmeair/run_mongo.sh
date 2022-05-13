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


name="mongo_${instance_id}"
port="$((27017 + instance_id))"

exec docker run --name="${name}" --rm -p "${port}:${port}" \
     "${docker_args[@]}" "mongo-acmeair" --nojournal --port="${port}"
