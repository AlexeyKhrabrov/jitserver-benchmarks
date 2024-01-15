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

format="png"
args=(-r "--format=${format}")

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

density_args=("${args[@]}")
args+=(-n "${runs}")
density_args+=(-n "${density_runs}")


benchmarks=("acmeair" "daytrader" "petclinic")

for b in "${benchmarks[@]}"; do
	./run_single.py  "${b}" -j    "${args[@]}"         &
	./run_cdf.py     "${b}" -j    "${args[@]}"         &
	./run_cdf.py     "${b}" -j -e "${args[@]}"         &
	./run_scale.py   "${b}" -j    "${args[@]}"         &
	./run_latency.py "${b}" -j    "${args[@]}"         &
	./run_density.py "${b}"       "${density_args[@]}" &
	./run_density.py "${b}" -s    "${density_args[@]}" &
done

wait


plots=(
	"single_full_cold_start_time" "single_full_cold_warmup_time" "single_full_cold_peak_mem"
	"single_full_warm_start_time" "single_full_warm_warmup_time" "single_full_warm_peak_mem"
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
