#!/bin/bash

set -e -u -o pipefail


usage_str="\
Usage: ${0} instance_id"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


if (( $# < 1 )); then usage; fi

instance_id="${1}"


name="mongo_${instance_id}"
port="$((27017 + instance_id))"

exec docker exec "${name}" mongorestore --port="${port}" --drop "/db_backup"
