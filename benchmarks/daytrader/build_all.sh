#!/bin/bash

set -e -u -o pipefail

dir="$(dirname "$(readlink -f "${BASH_SOURCE}")")"


usage_str="\
Usage: ${0} [-d|--db2] [db2_installer_path] [-t|--tune]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


build_db2=false
db2_installer_path=""
tune=false

for arg in "${@:1}"; do
	case "${arg}" in
		"-h" | "--help" )
			usage
			;;
		"-d" | "--db2" )
			build_db2=true
			;;
		"-t" | "--tune" )
			tune=true
			;;
		*)
			if [[ "${db2_installer_path}" == "" ]]; then
				db2_installer_path="${arg}"
			else
				usage
			fi
			;;
	esac
done


"${dir}/build_liberty.sh"

if [[ "${build_db2}" == true ]]; then
	if [[ "${db2_installer_path}" != "" ]]; then
		"${dir}/build_db2_base.sh" "${db2_installer_path}"
	else
		"${dir}/build_db2_base.sh"
	fi

	if [[ "${tune}" != true ]]; then
		"${dir}/build_db2.sh" --tune
	else
		"${dir}/build_db2.sh"
	fi
fi

"${dir}/build_jmeter.sh"
