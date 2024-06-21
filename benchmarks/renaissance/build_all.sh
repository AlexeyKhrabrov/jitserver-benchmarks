#!/bin/bash

set -e -u -o pipefail

dir="$(dirname "$(readlink -f "${BASH_SOURCE}")")"


usage_str="\
Usage: ${0} jdk_dir [-u|--update]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


if (( $# < 1 )); then usage; fi

jdk_dir=$(readlink -f "${1}")
update=false

for arg in "${@:2}"; do
	case "${arg}" in
		"-u" | "--update" )
			update=true
			;;
		*)
			usage
			;;
	esac
done


renaissance_ver="0.14.2"
renaissance_url="https://github.com/renaissance-benchmarks/renaissance/\
releases/download/v${renaissance_ver}/renaissance-gpl-${renaissance_ver}.jar"

if [[ "${update}" == true || ! -f "${dir}/renaissance.jar" ]]; then
	wget "$renaissance_url" -O "${dir}/renaissance.jar"
fi


jdk_bin="${jdk_dir}/bin"

if [[ "${update}" == true || ! -f "${dir}/JITServerPlugin.jar" ]]; then
	"${jdk_bin}/javac" -cp "${dir}/renaissance.jar" "${dir}/JITServerPlugin.java"
	"${jdk_bin}/jar" cfm "${dir}/JITServerPlugin.jar" "${dir}/JITServerPlugin.mf" \
	                 -C "${dir}" "JITServerPlugin.class"
	rm -f "${dir}/JITServerPlugin.class"
fi


docker build -t "renaissance" "${dir}"
