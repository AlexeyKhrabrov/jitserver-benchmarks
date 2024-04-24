#!/bin/bash

set -e -u -o pipefail


usage_str="\
Usage: ${0} all_hosts_file main_hosts_file runs density_runs [logs_path]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


if (( $# < 4 )); then usage; fi

all_hosts="${1}" # 8 main homogeneous machines
main_hosts="${2}" # total 11 machines
runs="${3}" # number of runs (repetitions) in all experiments except "density"
density_runs="${4}"

#NOTE: This option skips complete experiment runs and only runs the missing ones
args=("--skip-complete")

if (( $# >= 5 )); then
	logs_path="${5}"
	args+=(-L "${logs_path}")
else
	logs_path="logs"
fi

density_args=("${args[@]}")
args+=(-n "${runs}")
density_args+=(-n "${density_runs}")

# Needed by latency experiments
read -s -p "Password: " password
echo

#NOTE: Estimated durations assume 3 runs for "density" and 5 for other experiments


# all single: ~8h
#
# a=acmeair, d=daytrader, p=petclinic
# x=xsmall, s=small, m=medium, l=large
# c=cold, w=warm
#
# 0: dxc
# 1: dsc + dmc
# 2: dxw + dmw
# 3: dsw + dlc + dlw
# 4: axc + asc + amc + alc
# 5: axw + asw + amw + alw
# 6: pxc + psc + pmc + plc
# 7: pxw + psw + pmw + plw

./host_cleanup.py acmeair   "${main_hosts}"
./host_cleanup.py daytrader "${main_hosts}"
./host_cleanup.py petclinic "${main_hosts}"

./run_single.py daytrader "${main_hosts}" 0 0 -j "${args[@]}" &
./run_single.py daytrader "${main_hosts}" 2 1 -j "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 4 1 -j "${args[@]}" &
./run_single.py daytrader "${main_hosts}" 1 2 -j "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 5 2 -j "${args[@]}" &
./run_single.py daytrader "${main_hosts}" 3 3 -j "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 6 3 -j "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 7 3 -j "${args[@]}" &

./run_single.py acmeair   "${main_hosts}" 0 4 -j "${args[@]}" && \
./run_single.py acmeair   "${main_hosts}" 2 4 -j "${args[@]}" && \
./run_single.py acmeair   "${main_hosts}" 4 4 -j "${args[@]}" && \
./run_single.py acmeair   "${main_hosts}" 6 4 -j "${args[@]}" &
./run_single.py acmeair   "${main_hosts}" 1 5 -j "${args[@]}" && \
./run_single.py acmeair   "${main_hosts}" 3 5 -j "${args[@]}" && \
./run_single.py acmeair   "${main_hosts}" 5 5 -j "${args[@]}" && \
./run_single.py acmeair   "${main_hosts}" 7 5 -j "${args[@]}" &

./run_single.py petclinic "${main_hosts}" 0 6 -j "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 2 6 -j "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 4 6 -j "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 6 6 -j "${args[@]}" &
./run_single.py petclinic "${main_hosts}" 1 7 -j "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 3 7 -j "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 5 7 -j "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 7 7 -j "${args[@]}" &

wait

./host_cleanup.py acmeair "${main_hosts}"
./host_cleanup.py daytrader "${main_hosts}"
./host_cleanup.py petclinic "${main_hosts}"


# all cdf: ~4h
#
# a=acmeair, d=daytrader, p=petclinic
# ne="unlimited jitserver cpu", eq="equal jit cpu"
#
# 0: a_ne + p_ne
# 1: a_eq + p_eq
# 2: d_ne
# 3: d_eq

./run_cdf.py acmeair   "${main_hosts}" 0 -j    "${args[@]}" && \
./run_cdf.py petclinic "${main_hosts}" 0 -j    "${args[@]}" &

./run_cdf.py acmeair   "${main_hosts}" 1 -j -e "${args[@]}" && \
./run_cdf.py petclinic "${main_hosts}" 1 -j -e "${args[@]}" &

./run_cdf.py daytrader "${main_hosts}" 2 -j -n "${args[@]}" &

./run_cdf.py daytrader "${main_hosts}" 3 -j -e "${args[@]}" &

wait

./host_cleanup.py acmeair   "${main_hosts}"
./host_cleanup.py daytrader "${main_hosts}"
./host_cleanup.py petclinic "${main_hosts}"


# all latency localjit: ~1.5h
./run_latency.py acmeair   "${main_hosts}" 0 -j -S -l "${args[@]}" <<< "${password}" &
./run_latency.py daytrader "${main_hosts}" 1 -j -S -l "${args[@]}" <<< "${password}" &
./run_latency.py petclinic "${main_hosts}" 2 -j -S -l "${args[@]}" <<< "${password}" &
wait

# acmeair latency: ~4h
./run_latency.py acmeair   "${main_hosts}" 0 -j -S    "${args[@]}" <<< "${password}" &
./run_latency.py acmeair   "${main_hosts}" 1 -j -S    "${args[@]}" <<< "${password}" &
./run_latency.py acmeair   "${main_hosts}" 2 -j -S    "${args[@]}" <<< "${password}" &
./run_latency.py acmeair   "${main_hosts}" 3 -j -S    "${args[@]}" <<< "${password}" &
wait

# daytrader latency: ~9h
./run_latency.py daytrader "${main_hosts}" 0 -j -S    "${args[@]}" <<< "${password}" &
./run_latency.py daytrader "${main_hosts}" 1 -j -S    "${args[@]}" <<< "${password}" &
./run_latency.py daytrader "${main_hosts}" 2 -j -S    "${args[@]}" <<< "${password}" &
./run_latency.py daytrader "${main_hosts}" 3 -j -S    "${args[@]}" <<< "${password}" &
wait

# petclinic latency: ~4h
./run_latency.py petclinic "${main_hosts}" 0 -j -S    "${args[@]}" <<< "${password}" &
./run_latency.py petclinic "${main_hosts}" 1 -j -S    "${args[@]}" <<< "${password}" &
./run_latency.py petclinic "${main_hosts}" 2 -j -S    "${args[@]}" <<< "${password}" &
./run_latency.py petclinic "${main_hosts}" 3 -j -S    "${args[@]}" <<< "${password}" &
wait


# acmeair scale: ~8h
./host_cleanup.py acmeair "${all_hosts}"
./run_scale.py acmeair "${all_hosts}" 0 -j "${args[@]}" &
wait
./host_cleanup.py acmeair "${all_hosts}"
./run_scale.py acmeair "${all_hosts}" 1 -j "${args[@]}" &
./run_scale.py acmeair "${all_hosts}" 2 -j "${args[@]}" &
wait
./host_cleanup.py acmeair "${all_hosts}"
./run_scale.py acmeair "${all_hosts}" 3 -j "${args[@]}" &
./run_scale.py acmeair "${all_hosts}" 4 -j "${args[@]}" &
wait
./host_cleanup.py acmeair "${all_hosts}"
./run_scale.py acmeair "${all_hosts}" 5 -j "${args[@]}" &
./run_scale.py acmeair "${all_hosts}" 6 -j "${args[@]}" &
wait
./host_cleanup.py acmeair "${all_hosts}"
./run_scale.py acmeair "${all_hosts}" 7 -j "${args[@]}" &
./run_scale.py acmeair "${all_hosts}" 8 -j "${args[@]}" &
./run_scale.py acmeair "${all_hosts}" 9 -j "${args[@]}" &
wait
./host_cleanup.py acmeair "${all_hosts}"

# daytrader scale: ~21h
./host_cleanup.py daytrader "${all_hosts}"
./run_scale.py daytrader "${all_hosts}" 0 -j "${args[@]}" &
wait
./host_cleanup.py daytrader "${all_hosts}"
./run_scale.py daytrader "${all_hosts}" 1 -j "${args[@]}" &
wait
./host_cleanup.py daytrader "${all_hosts}"
./run_scale.py daytrader "${all_hosts}" 2 -j "${args[@]}" &
./run_scale.py daytrader "${all_hosts}" 3 -j "${args[@]}" &
wait
./host_cleanup.py daytrader "${all_hosts}"
./run_scale.py daytrader "${all_hosts}" 4 -j "${args[@]}" &
./run_scale.py daytrader "${all_hosts}" 5 -j "${args[@]}" &
wait
./host_cleanup.py daytrader "${all_hosts}"
./run_scale.py daytrader "${all_hosts}" 6 -j "${args[@]}" &
./run_scale.py daytrader "${all_hosts}" 7 -j "${args[@]}" &
./run_scale.py daytrader "${all_hosts}" 8 -j "${args[@]}" &
./run_scale.py daytrader "${all_hosts}" 9 -j "${args[@]}" &
wait
./host_cleanup.py daytrader "${all_hosts}"

# petclinic scale: ~7h
./host_cleanup.py petclinic "${all_hosts}"
./run_scale.py petclinic "${all_hosts}" 0 -j "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 1 -j "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 2 -j "${args[@]}" &
wait
./host_cleanup.py petclinic "${all_hosts}"
./run_scale.py petclinic "${all_hosts}" 3 -j "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 4 -j "${args[@]}" &
wait
./host_cleanup.py petclinic "${all_hosts}"
./run_scale.py petclinic "${all_hosts}" 5 -j "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 6 -j "${args[@]}" &
wait
./host_cleanup.py petclinic "${all_hosts}"
./run_scale.py petclinic "${all_hosts}" 7 -j "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 8 -j "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 9 -j "${args[@]}" &
wait
./host_cleanup.py petclinic "${all_hosts}"


# acmeair density: ~60h
./run_density.py acmeair   "${all_hosts}"    "${args[@]}"
./run_density.py acmeair   "${all_hosts}" -s "${args[@]}"

# daytrader density: ~60h
./run_density.py daytrader "${all_hosts}"    "${args[@]}"
./run_density.py daytrader "${all_hosts}" -s "${args[@]}"

# petclinic density: ~60h
./run_density.py petclinic "${all_hosts}"    "${args[@]}"
./run_density.py petclinic "${all_hosts}" -s "${args[@]}"


# acmeair servermem: ~8h
./host_cleanup.py acmeair "${all_hosts}"
./run_servermem.py acmeair "${all_hosts}" 0 -j "${args[@]}" &
wait
./host_cleanup.py acmeair "${all_hosts}"
./run_servermem.py acmeair "${all_hosts}" 1 -j "${args[@]}" &
./run_servermem.py acmeair "${all_hosts}" 2 -j "${args[@]}" &
wait
./host_cleanup.py acmeair "${all_hosts}"
./run_servermem.py acmeair "${all_hosts}" 3 -j "${args[@]}" &
./run_servermem.py acmeair "${all_hosts}" 4 -j "${args[@]}" &
wait
./host_cleanup.py acmeair "${all_hosts}"
./run_servermem.py acmeair "${all_hosts}" 5 -j "${args[@]}" &
./run_servermem.py acmeair "${all_hosts}" 6 -j "${args[@]}" &
wait
./host_cleanup.py acmeair "${all_hosts}"
./run_servermem.py acmeair "${all_hosts}" 7 -j "${args[@]}" &
./run_servermem.py acmeair "${all_hosts}" 8 -j "${args[@]}" &
./run_servermem.py acmeair "${all_hosts}" 9 -j "${args[@]}" &
wait
./host_cleanup.py acmeair "${all_hosts}"

# daytrader servermem: ~15h
./host_cleanup.py daytrader "${all_hosts}"
./run_servermem.py daytrader "${all_hosts}" 0 -j "${args[@]}" &
wait
./host_cleanup.py daytrader "${all_hosts}"
./run_servermem.py daytrader "${all_hosts}" 1 -j "${args[@]}" &
wait
./host_cleanup.py daytrader "${all_hosts}"
./run_servermem.py daytrader "${all_hosts}" 2 -j "${args[@]}" &
./run_servermem.py daytrader "${all_hosts}" 3 -j "${args[@]}" &
wait
./host_cleanup.py daytrader "${all_hosts}"
./run_servermem.py daytrader "${all_hosts}" 4 -j "${args[@]}" &
./run_servermem.py daytrader "${all_hosts}" 5 -j "${args[@]}" &
wait
./host_cleanup.py daytrader "${all_hosts}"
./run_servermem.py daytrader "${all_hosts}" 6 -j "${args[@]}" &
./run_servermem.py daytrader "${all_hosts}" 7 -j "${args[@]}" &
./run_servermem.py daytrader "${all_hosts}" 8 -j "${args[@]}" &
./run_servermem.py daytrader "${all_hosts}" 9 -j "${args[@]}" &
wait
./host_cleanup.py daytrader "${all_hosts}"

# petclinic servermem: ~8h
./host_cleanup.py petclinic "${all_hosts}"
./run_servermem.py petclinic "${all_hosts}" 0 -j "${args[@]}" &
./run_servermem.py petclinic "${all_hosts}" 1 -j "${args[@]}" &
./run_servermem.py petclinic "${all_hosts}" 2 -j "${args[@]}" &
wait
./host_cleanup.py petclinic "${all_hosts}"
./run_servermem.py petclinic "${all_hosts}" 3 -j "${args[@]}" &
./run_servermem.py petclinic "${all_hosts}" 4 -j "${args[@]}" &
wait
./host_cleanup.py petclinic "${all_hosts}"
./run_servermem.py petclinic "${all_hosts}" 5 -j "${args[@]}" &
./run_servermem.py petclinic "${all_hosts}" 6 -j "${args[@]}" &
wait
./host_cleanup.py petclinic "${all_hosts}"
./run_servermem.py petclinic "${all_hosts}" 7 -j "${args[@]}" &
./run_servermem.py petclinic "${all_hosts}" 8 -j "${args[@]}" &
./run_servermem.py petclinic "${all_hosts}" 9 -j "${args[@]}" &
wait
./host_cleanup.py petclinic "${all_hosts}"
