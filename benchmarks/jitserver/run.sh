#!/bin/bash

set -e -u -o pipefail

dir="$(dirname "$(readlink -f "${BASH_SOURCE}")")"


usage_str="\
Usage: ${0} benchmark instance_id jdk_path
       jitserver_args jitserver_env [<docker args>]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


if (( $# < 5 )); then usage; fi

benchmark="${1}"
instance_id="${2}"
jdk_path=$(readlink -f "${3}")
jitserver_args=(${4})
jitserver_env=(${5})
docker_args=("${@:6}")


docker_args+=(
	-w "/output" -v "${jdk_path}:/opt/ibm/java"
	-v "${jdk_path}/cert.pem:/cert.pem" -v "${jdk_path}/key.pem:/key.pem"
)

name="jitserver_${instance_id}"
vlogs_path="$(readlink -f "${dir}/../${benchmark}/${name}/vlogs")"
mkdir -m g+s -p "${vlogs_path}"
docker_args+=(-v "${vlogs_path}:/output/vlogs")

for e in "${jitserver_env[@]}"; do
	docker_args+=(-e "${e}")
done


function unquote() { tmp="${1#\'}"; echo "${tmp%\'}"; }

args=()

for arg in "${jitserver_args[@]}"; do
	arg="$(unquote "${arg}")"
	args+=("${arg}")
done


port="$((38400 + instance_id))"

#NOTE: the container is not automatically deleted
exec docker run --name="${name}" -p "${port}:${port}" \
     "${docker_args[@]}" "jitserver" "${args[@]}"
