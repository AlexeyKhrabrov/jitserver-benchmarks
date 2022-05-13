#!/bin/bash

set -e -u -o pipefail

dir="$(dirname "$(readlink -f "${BASH_SOURCE}")")"


"${dir}/build_petclinic.sh"
"${dir}/build_jmeter.sh"
