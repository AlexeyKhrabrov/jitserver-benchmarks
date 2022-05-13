#!/bin/bash

set -e -u -o pipefail

dir="$(dirname "$(readlink -f "${BASH_SOURCE}")")"


git clone -b "atc22ae" "https://github.com/AlexeyKhrabrov/openj9"
git clone -b "atc22ae" "https://github.com/AlexeyKhrabrov/omr"

"${dir}/scripts/openj9_setup.sh" "jdk/" 8 8 "openj9/" "omr/" \
	"https://github.com/AlexeyKhrabrov/openj9-openjdk-jdk8" "atc22ae"
