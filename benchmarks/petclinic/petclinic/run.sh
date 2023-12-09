#!/bin/bash

set -e -u -o pipefail


http_port="${1}"

args=("-Dserver.port=${http_port}")
for arg in ${JVM_ARGS}; do
	args+=("$(echo "${arg}" | xargs)")
done


cd "/output"

printf "JVM start timestamp: "
date -u "+%FT%T.%N"

exec "/opt/ibm/java/bin/java" "${args[@]}" -jar "/petclinic.jar"
