#!/bin/bash

set -e -u -o pipefail


args=()
for arg in ${JVM_ARGS}; do
	args+=("$(echo "${arg}" | xargs)")
done


cd "/output"

printf "JVM start timestamp: "
date -u "+%FT%T.%N"

exec "/opt/ibm/java/bin/java" "${args[@]}" -jar "/renaissance.jar" \
     --scratch-base "/scratch" --plugin "/JITServerPlugin.jar" "$@"
