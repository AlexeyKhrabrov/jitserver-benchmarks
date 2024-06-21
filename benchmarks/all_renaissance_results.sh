#!/bin/bash

set -e -u -o pipefail


usage_str="\
Usage: ${0} runs multi_runs [logs_path] [results_path]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


if (( $# < 2 )); then usage; fi

runs="${1}"
multi_runs="${2}"

format="png"
args=(-r "--format=${format}")

#NOTE: This option generates the plots with a single legend per figure (not per plot)
args+=("--single-legend")

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

multi_args=("${args[@]}")
args+=(-n "${runs}")
multi_args+=(-n "${multi_runs}" "--overlays")


./run_renaissance_single.py    "${args[@]}" &
./run_renaissance_single.py -F "${args[@]}" &
./run_renaissance_multi.py     "${multi_args[@]}" &
./run_renaissance_multi.py  -F "${multi_args[@]}" &

wait


plots=(
	"single_1st_first_time" "single_1st_peak_mem"
	"single_full_warmup_time" "single_full_peak_throughput" "single_full_peak_mem"
	"multi_1st_overall_total_cpu_time" "multi_1st_overall_total_peak_mem"
	"multi_full_overall_total_cpu_time" "multi_full_overall_total_peak_mem"
)

suffixes=("no_avg" "only_avg" "with_avg")
ngroups=3

dst="${results_path}/plots/renaissance"
mkdir -p "${dst}"

for p in "${plots[@]}"; do
	src="${results_path}/renaissance/${p}_normalized"

	cp -a "${src}_only_avg_all.${format}" "${dst}/${p////_}_only_avg_all.${format}" || true
	for s in "${suffixes[@]}"; do
		for (( g = 0 ; g < ngroups ; ++g)); do
			cp -a "${src}_${s}_${g}.${format}" "${dst}/${p////_}_${s}_${g}.${format}" || true
		done
	done
done
