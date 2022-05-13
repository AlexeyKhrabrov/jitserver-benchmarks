#!/bin/bash

set -e -u -o pipefail


cd "${JMETER_HOME}"

jmeter -g "/output/results.jtl" -o "/output/report"

chmod -R g+w "/output/report"
#TODO: host directory must be created with g+s
