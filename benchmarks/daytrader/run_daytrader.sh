#!/bin/bash

set -e -u -o pipefail

dir="$(dirname "$(readlink -f "${BASH_SOURCE}")")"


usage_str="\
Usage: ${0} instance_id db2_addr db2_port jdk_path scc_path
       jvm_args jvm_env extra_args(unused) [<docker args>]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


if (( $# < 8 )); then usage; fi

instance_id="${1}"
db2_addr="${2}"
db2_port="${3}"
jdk_path=$(readlink -f "${4}")
scc_path="${5}" # can be "" (scc directory is not mapped outside the container)
jvm_args="${6}"
jvm_env=(${7})
extra_args=(${8})
docker_args=("${@:9}")


docker_args+=(-v "${jdk_path}:/opt/ibm/java" -v "${jdk_path}/cert.pem:/cert.pem")

name="daytrader_${instance_id}"
vlogs_path="${dir}/${name}/vlogs"
mkdir -m a=rwx,g+s -p "${vlogs_path}"
docker_args+=(-v "${vlogs_path}:/output/vlogs")

messages_log_path="${dir}/${name}/messages.log"
touch "${messages_log_path}"
chmod a=rw "${messages_log_path}"
docker_args+=(-v "${messages_log_path}:/logs/messages.log")

if [[ "${scc_path}" != "" ]]; then
	scc_path=$(readlink -f "${scc_path}")
	mkdir -p "${scc_path}"
	chmod -R a=rwX "${scc_path}"
	chmod g+s "${scc_path}"
	docker_args+=(-v "${scc_path}:/output/.classCache")
fi

for e in "${jvm_env[@]}"; do
	docker_args+=(-e "${e}")
done


http_port="$((9080 + instance_id))"
jms_port="$((7276 + instance_id))"
iiop_port="$((2809 + instance_id))"

printf "Docker start timestamp: "
date -u "+%FT%T.%N"

#NOTE: the container is not automatically deleted
exec docker run --name="${name}" -p "${http_port}:${http_port}" \
     -p "${jms_port}:${jms_port}" -p "${iiop_port}:${iiop_port}" \
     -e JVM_ARGS="${jvm_args}" "${docker_args[@]}" "liberty-daytrader" \
     "${db2_addr}" "${db2_port}" "${http_port}" "${jms_port}" "${iiop_port}"
