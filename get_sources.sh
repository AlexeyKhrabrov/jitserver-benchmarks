#!/bin/bash

set -e -u -o pipefail

dir="$(dirname "$(readlink -f "${BASH_SOURCE}")")"


usage_str="\
Usage: ${0} [jdk_ver] [bootjdk_ver]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


for arg in "$@"; do
	if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then usage; fi
done

if (( $# == 2 )); then
	jdk_ver="${1}"
	bootjdk_ver="${2}"
elif (( $# == 1 )); then
	jdk_ver="${1}"
	bootjdk_ver="${1}"
elif (( $# == 0 )); then
	jdk_ver=8
	bootjdk_ver=8
else
	usage
fi


url_base="https://github.com/AlexeyKhrabrov"
branch="atc22ae"

git clone -b "${branch}" "${url_base}/openj9"
git clone -b "${branch}" "${url_base}/omr"

"${dir}/scripts/openj9_setup.sh" "${dir}/jdk/" ${jdk_ver} ${bootjdk_ver} "${dir}/openj9/" "${dir}/omr/" \
                                 "${url_base}/openj9-openjdk-jdk${jdk_ver}" "${branch}"
