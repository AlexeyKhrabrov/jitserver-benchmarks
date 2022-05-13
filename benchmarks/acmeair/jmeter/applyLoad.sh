#!/bin/bash

set -e -u -o pipefail


liberty_addr="${1}"
interval="${2}"
latency_data="${3}"
report_data="${4}"

cd "${JMETER_HOME}"


sed -i "s/localhost/${liberty_addr}/g" "hosts.csv"
sed -i "s/summariser.interval=.*/summariser.interval=${interval}/g" \
    "bin/jmeter.properties"

extra_args=()
if [[ "${latency_data}" == "true" || "${report_data}" == "true" ]]; then
	extra_args+=(-l "/output/results.jtl")
fi

if [[ "${report_data}" == "true" ]]; then
	prefix="jmeter.save.saveservice"
	properties=("assertion_results_failure_message" "response_code"
	            "response_message" "successful" "thread_name" "latency"
	            "connect_time" "bytes" "sent_bytes" "thread_counts")

	args=()
	for p in "${properties[@]}"; do
		args+=(-e "s/${prefix}.${p}=.*/${prefix}.${p}=true/g")
	done

	sed "${args[@]}" -i "bin/jmeter.properties"
fi


/usr/bin/time -v \
jmeter -n -t "AcmeAir-v3.jmx" "${extra_args[@]}" -DusePureIDs=true \
       -JURL="${JURL}" -JPORT="${JPORT}" -JUSERBOTTOM="${JUSERBOTTOM}" \
       -JUSER="${JUSER}" -JTHREADS="${JTHREADS}" -JDURATION="${JDURATION}"
