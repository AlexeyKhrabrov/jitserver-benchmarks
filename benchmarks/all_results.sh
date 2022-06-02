#!/bin/bash

set -e -u -o pipefail


# Usage: ./all_results.sh runs density_runs [logs_path] [results_path]

runs="${1}"
density_runs="${2}"

args=()

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

./run_cdf.py acmeair -r -j -n "${runs}" &
./run_cdf.py daytrader -r -j -n "${runs}" &
./run_cdf.py petclinic -r -j -n "${runs}" &

./run_cdf.py acmeair -r -j -e -n "${runs}" &
./run_cdf.py daytrader -r -j -e -n "${runs}" &
./run_cdf.py petclinic -r -j -e -n "${runs}" &

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
	mkdir -p "${results_path}/plots/${b}"
	for p in "${plots[@]}"; do
		cp -a "${results_path}/${b}/${p}.png" "${results_path}/plots/${b}/${p////_}.png" || true
	done
done
