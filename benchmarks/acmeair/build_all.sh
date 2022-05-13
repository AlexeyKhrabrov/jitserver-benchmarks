#!/bin/bash

set -e -u -o pipefail

dir="$(dirname "$(readlink -f "${BASH_SOURCE}")")"


"${dir}/build_mongo.sh"
"${dir}/build_liberty.sh"
"${dir}/build_jmeter.sh"
