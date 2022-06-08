#!/bin/bash

set -e -u -o pipefail


usage_str="\
Usage: ${0} runs density_runs [logs_path] [results_path]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


if (( $# < 2 )); then usage; fi

runs="${1}"
density_runs="${2}"

format="pdf"
args=("--format=${format}")

#NOTE: These options generate the plots as presented in the paper - with a
# single legend per figure (not per plot), and with the same Y axis scale for
# "cold" and "warm" configurations in "single" and "density" experiments.
args+=("--single-legend" "--same-limits")

if (( $# >= 3 )); then
	logs_path="${3}"
	args+=(-L "${logs_path}")
else
	logs_path="logs"
fi

if (( $# >= 4 )); then
	results_path="${4}"
	args+=(-R "${results_path}")
else
	results_path="results"
fi


./run_single.py acmeair -r -n "${runs}" "${args[@]}" &
./run_single.py daytrader -r -n "${runs}" "${args[@]}" &
./run_single.py petclinic -r -n "${runs}" "${args[@]}" &

./run_single.py acmeair -r -j -n "${runs}" "${args[@]}" &
./run_single.py daytrader -r -j -n "${runs}" "${args[@]}" &
./run_single.py petclinic -r -j -n "${runs}" "${args[@]}" &

./run_cdf.py acmeair -r -j -n "${runs}" "${args[@]}" &
./run_cdf.py daytrader -r -j -n "${runs}" "${args[@]}" &
./run_cdf.py petclinic -r -j -n "${runs}" "${args[@]}" &

./run_cdf.py acmeair -r -j -e -n "${runs}" "${args[@]}" &
./run_cdf.py daytrader -r -j -e -n "${runs}" "${args[@]}" &
./run_cdf.py petclinic -r -j -e -n "${runs}" "${args[@]}" &

./run_scale.py acmeair -r -j -n "${runs}" "${args[@]}" &
./run_scale.py daytrader -r -j -n "${runs}" "${args[@]}" &
./run_scale.py petclinic -r -j -n "${runs}" "${args[@]}" &

./run_latency.py acmeair -r -j -n "${runs}" "${args[@]}" &
./run_latency.py daytrader -r -j -n "${runs}" "${args[@]}" &
./run_latency.py petclinic -r -j -n "${runs}" "${args[@]}" &

./run_density.py acmeair -r -n "${density_runs}" "${args[@]}" &
./run_density.py daytrader -r -n "${density_runs}" "${args[@]}" &
./run_density.py petclinic -r -n "${density_runs}" "${args[@]}" &

./run_density.py acmeair -r -s -n "${density_runs}" "${args[@]}" &
./run_density.py daytrader -r -s -n "${density_runs}" "${args[@]}" &
./run_density.py petclinic -r -s -n "${density_runs}" "${args[@]}" &

wait


benchmarks=("acmeair" "daytrader" "petclinic")

plots=(
	"single_start_cold_start_time" "single_full_cold_warmup_time" "single_full_cold_peak_mem"
	"single_start_warm_start_time" "single_full_warm_warmup_time" "single_full_warm_peak_mem"
	"cdf_ne_full/comp_times_log" "cdf_ne_full/queue_times_log"
	"cdf_eq_full/comp_times_log" "cdf_eq_full/queue_times_log"
	"latency_full_full_warmup_time" "scale_full_full_warmup_time_normalized"
	"density_noscc_cpu_time_per_req" "density_noscc_total_peak_mem"
	"density_scc_cpu_time_per_req" "density_scc_total_peak_mem"
)

for b in "${benchmarks[@]}"; do
	dst="${results_path}/plots/${b}"
	mkdir -p "${dst}"
	for p in "${plots[@]}"; do
		cp -a "${results_path}/${b}/${p}.${format}" "${dst}/${p////_}.${format}" || true
	done
done
