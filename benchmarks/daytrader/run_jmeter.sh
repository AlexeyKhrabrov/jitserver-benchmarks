#!/bin/bash

set -e -u -o pipefail

dir="$(dirname "$(readlink -f "${BASH_SOURCE}")")"


usage_str="\
Usage: ${0} instance_id n_instances n_dbs liberty_addr nthreads duration
       interval latency_data report_data scc_path jvm_args [<docker args>]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


if (( $# < 11 )); then usage; fi

instance_id="${1}"
n_instances="${2}"
n_dbs="${3}"
liberty_addr="${4}"
nthreads="${5}"
duration="${6}" # seconds
interval="${7}" # seconds; minimum is 6
latency_data="${8}" # true or false
report_data="${9}" # true or false
scc_path="${10}" # can be "" (scc directory is not mapped outside the container)
jvm_args="${11}"
docker_args=("${@:12}")


name="jmeter_${instance_id}"
docker_args+=(-v "${dir}/${name}:/output")

if [[ "${scc_path}" != "" ]]; then
	scc_path=$(readlink -f "${scc_path}")
	mkdir -p "${scc_path}"
	docker_args+=(-v "${scc_path}:/.classCache")
fi

num_users="$((15000 / ((n_instances - 1) / n_dbs + 1)))"
min_user="$((num_users * ((instance_id % n_instances) / n_dbs)))"
max_user="$((min_user + num_users - 1))"

exec docker run --name="${name}" --rm -e JPORT="$((9080 + instance_id))" \
     -e JTHREADS="${nthreads}" -e JDURATION="${duration}" \
     -e JBOTUID="${min_user}" -e JTOPUID="${max_user}" \
     -e JVM_ARGS="${jvm_args}" "${docker_args[@]}" "jmeter-daytrader" \
     "${liberty_addr}" "${interval}" "${latency_data}" "${report_data}"
