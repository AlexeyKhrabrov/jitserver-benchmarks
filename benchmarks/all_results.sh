#!/bin/bash

set -e -u -o pipefail


# Usage: ./all_results.sh [logs_path] [results_path]

args=()

if (( $# >= 1 )); then
	logs_path="${1}"
	args+=(-L "${logs_path}")
else
	logs_path="logs"
fi

if (( $# >= 2 )); then
	results_path="${2}"
	args+=(-R "${results_path}")
else
	results_path="results"
fi


./run_single.py acmeair -r "${args[@]}" &
./run_single.py daytrader -r "${args[@]}" &
./run_single.py petclinic -r "${args[@]}" &

./run_single.py acmeair -r -j "${args[@]}" &
./run_single.py daytrader -r -j "${args[@]}" &
./run_single.py petclinic -r -j "${args[@]}" &

./run_scale.py acmeair -r -j "${args[@]}" &
./run_scale.py daytrader -r -j "${args[@]}" &
./run_scale.py petclinic -r -j "${args[@]}" &

./run_latency.py acmeair -r -j "${args[@]}" &
./run_latency.py daytrader -r -j "${args[@]}" &
./run_latency.py petclinic -r -j "${args[@]}" &

./run_density.py acmeair -r "${args[@]}" &
./run_density.py daytrader -r "${args[@]}" &
./run_density.py petclinic -r "${args[@]}" &

./run_density.py acmeair -r -s "${args[@]}" &
./run_density.py daytrader -r -s "${args[@]}" &
./run_density.py petclinic -r -s "${args[@]}" &

wait


benchmarks=("acmeair" "daytrader" "petclinic")

plots=(
	"density_noscc_cpu_time_per_req" "density_noscc_total_peak_mem"
	"density_scc_cpu_time_per_req" "density_scc_total_peak_mem"
	"latency_full_full_warmup_time" "scale_full_full_warmup_time_normalized"
	"single_start_cold_start_time" "single_full_cold_warmup_time" "single_full_cold_peak_mem"
	"single_start_warm_start_time" "single_full_warm_warmup_time" "single_full_warm_peak_mem"
)

for b in "${benchmarks[@]}"; do
	mkdir -p "${results_path}/plots/${b}"
	for p in "${plots[@]}"; do
		cp -a "${results_path}/${b}/${p}.png" "${results_path}/plots/${b}/" || true
	done
done
