#!/bin/bash

set -e -u -o pipefail


# Usage: ./all_experiments.sh all_hosts_file main_hosts_file
#        runs density_runs [logs_path]

all_hosts="${1}" # 8 main homogeneous machines
main_hosts="${2}" # total 11 machines
runs="${3}" # number of runs (repetitions) in all experiments except "density"
density_runs="${4}"

args=()
if (( $# >= 5 )); then
	logs_path="${5}"
	args+=(-L "${logs_path}")
else
	logs_path="logs"
fi

# Needed by latency experiments
read -s -p "Password: " password
echo


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

./run_single.py daytrader "${main_hosts}" 0 0 -n "${runs}" "${args[@]}" &
./run_single.py daytrader "${main_hosts}" 2 1 -n "${runs}" "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 4 1 -n "${runs}" "${args[@]}" &
./run_single.py daytrader "${main_hosts}" 1 2 -n "${runs}" "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 5 2 -n "${runs}" "${args[@]}" &
./run_single.py daytrader "${main_hosts}" 3 3 -n "${runs}" "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 6 3 -n "${runs}" "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 7 3 -n "${runs}" "${args[@]}" &

./run_single.py acmeair "${main_hosts}" 0 4 -n "${runs}" "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 2 4 -n "${runs}" "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 4 4 -n "${runs}" "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 6 4 -n "${runs}" "${args[@]}" &
./run_single.py acmeair "${main_hosts}" 1 5 -n "${runs}" "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 3 5 -n "${runs}" "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 5 5 -n "${runs}" "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 7 5 -n "${runs}" "${args[@]}" &

./run_single.py petclinic "${main_hosts}" 0 6 -n "${runs}" "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 2 6 -n "${runs}" "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 4 6 -n "${runs}" "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 6 6 -n "${runs}" "${args[@]}" &
./run_single.py petclinic "${main_hosts}" 1 7 -n "${runs}" "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 3 7 -n "${runs}" "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 5 7 -n "${runs}" "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 7 7 -n "${runs}" "${args[@]}" &

wait

./run_single.py daytrader "${main_hosts}" 0 0 -j -n "${runs}" "${args[@]}" &
./run_single.py daytrader "${main_hosts}" 2 1 -j -n "${runs}" "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 4 1 -j -n "${runs}" "${args[@]}" &
./run_single.py daytrader "${main_hosts}" 1 2 -j -n "${runs}" "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 5 2 -j -n "${runs}" "${args[@]}" &
./run_single.py daytrader "${main_hosts}" 3 3 -j -n "${runs}" "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 6 3 -j -n "${runs}" "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 7 3 -j -n "${runs}" "${args[@]}" &

./run_single.py acmeair "${main_hosts}" 0 4 -j -n "${runs}" "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 2 4 -j -n "${runs}" "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 4 4 -j -n "${runs}" "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 6 4 -j -n "${runs}" "${args[@]}" &
./run_single.py acmeair "${main_hosts}" 1 5 -j -n "${runs}" "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 3 5 -j -n "${runs}" "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 5 5 -j -n "${runs}" "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 7 5 -j -n "${runs}" "${args[@]}" &

./run_single.py petclinic "${main_hosts}" 0 6 -j -n "${runs}" "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 2 6 -j -n "${runs}" "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 4 6 -j -n "${runs}" "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 6 6 -j -n "${runs}" "${args[@]}" &
./run_single.py petclinic "${main_hosts}" 1 7 -j -n "${runs}" "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 3 7 -j -n "${runs}" "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 5 7 -j -n "${runs}" "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 7 7 -j -n "${runs}" "${args[@]}" &

wait


# all latency localjit: ~1.5h
./run_latency.py acmeair "${main_hosts}" 0 -l -j -n "${runs}" -S "${args[@]}" <<< "${password}" &
./run_latency.py daytrader "${main_hosts}" 1 -l -j -n "${runs}" -S "${args[@]}" <<< "${password}" &
./run_latency.py petclinic "${main_hosts}" 2 -l -j -n "${runs}" -S "${args[@]}" <<< "${password}" &
wait

# acmeair latency: ~4h
./run_latency.py acmeair "${main_hosts}" 0 -j -n "${runs}" -S "${args[@]}" <<< "${password}" &
./run_latency.py acmeair "${main_hosts}" 1 -j -n "${runs}" -S "${args[@]}" <<< "${password}" &
./run_latency.py acmeair "${main_hosts}" 2 -j -n "${runs}" -S "${args[@]}" <<< "${password}" &
./run_latency.py acmeair "${main_hosts}" 3 -j -n "${runs}" -S "${args[@]}" <<< "${password}" &
wait

# daytrader latency: ~9h
./run_latency.py daytrader "${main_hosts}" 0 -j -n "${runs}" -S "${args[@]}" <<< "${password}" &
./run_latency.py daytrader "${main_hosts}" 1 -j -n "${runs}" -S "${args[@]}" <<< "${password}" &
./run_latency.py daytrader "${main_hosts}" 2 -j -n "${runs}" -S "${args[@]}" <<< "${password}" &
./run_latency.py daytrader "${main_hosts}" 3 -j -n "${runs}" -S "${args[@]}" <<< "${password}" &
wait

# petclinic latency: ~3h
./run_latency.py petclinic "${main_hosts}" 0 -j -n "${runs}" -S "${args[@]}" <<< "${password}" &
./run_latency.py petclinic "${main_hosts}" 1 -j -n "${runs}" -S "${args[@]}" <<< "${password}" &
./run_latency.py petclinic "${main_hosts}" 2 -j -n "${runs}" -S "${args[@]}" <<< "${password}" &
./run_latency.py petclinic "${main_hosts}" 3 -j -n "${runs}" -S "${args[@]}" <<< "${password}" &
wait


# acmeair scale: ~8h
./run_scale.py acmeair "${all_hosts}" 0 -j -n "${runs}" "${args[@]}" &
wait
./run_scale.py acmeair "${all_hosts}" 1 -j -n "${runs}" "${args[@]}" &
./run_scale.py acmeair "${all_hosts}" 2 -j -n "${runs}" "${args[@]}" &
wait
./run_scale.py acmeair "${all_hosts}" 3 -j -n "${runs}" "${args[@]}" &
./run_scale.py acmeair "${all_hosts}" 4 -j -n "${runs}" "${args[@]}" &
wait
./run_scale.py acmeair "${all_hosts}" 5 -j -n "${runs}" "${args[@]}" &
./run_scale.py acmeair "${all_hosts}" 6 -j -n "${runs}" "${args[@]}" &
wait
./run_scale.py acmeair "${all_hosts}" 7 -j -n "${runs}" "${args[@]}" &
./run_scale.py acmeair "${all_hosts}" 8 -j -n "${runs}" "${args[@]}" &
./run_scale.py acmeair "${all_hosts}" 9 -j -n "${runs}" "${args[@]}" &
wait

# daytrader scale: ~21h
./run_scale.py daytrader "${all_hosts}" 0 -j -n "${runs}" "${args[@]}" &
wait
./run_scale.py daytrader "${all_hosts}" 1 -j -n "${runs}" "${args[@]}" &
wait
./run_scale.py daytrader "${all_hosts}" 2 -j -n "${runs}" "${args[@]}" &
./run_scale.py daytrader "${all_hosts}" 3 -j -n "${runs}" "${args[@]}" &
wait
./run_scale.py daytrader "${all_hosts}" 4 -j -n "${runs}" "${args[@]}" &
./run_scale.py daytrader "${all_hosts}" 5 -j -n "${runs}" "${args[@]}" &
wait
./run_scale.py daytrader "${all_hosts}" 6 -j -n "${runs}" "${args[@]}" &
./run_scale.py daytrader "${all_hosts}" 7 -j -n "${runs}" "${args[@]}" &
./run_scale.py daytrader "${all_hosts}" 8 -j -n "${runs}" "${args[@]}" &
./run_scale.py daytrader "${all_hosts}" 9 -j -n "${runs}" "${args[@]}" &
wait

# petclinic scale: ~5h
./run_scale.py petclinic "${all_hosts}" 0 -j -n "${runs}" "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 1 -j -n "${runs}" "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 2 -j -n "${runs}" "${args[@]}" &
wait
./run_scale.py petclinic "${all_hosts}" 3 -j -n "${runs}" "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 4 -j -n "${runs}" "${args[@]}" &
wait
./run_scale.py petclinic "${all_hosts}" 5 -j -n "${runs}" "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 6 -j -n "${runs}" "${args[@]}" &
wait
./run_scale.py petclinic "${all_hosts}" 7 -j -n "${runs}" "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 8 -j -n "${runs}" "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 9 -j -n "${runs}" "${args[@]}" &
wait
./run_scale.py petclinic "${all_hosts}" 10 -j -n "${runs}" "${args[@]}" &
wait


# acmeair density: ~60h
./run_density.py acmeair "${all_hosts}" -n "${density_runs}" "${args[@]}"
./run_density.py acmeair "${all_hosts}" -s -n "${density_runs}" "${args[@]}"

# daytrader density: ~60h
./run_density.py daytrader "${all_hosts}" -n "${density_runs}" "${args[@]}"
./run_density.py daytrader "${all_hosts}" -s -n "${density_runs}" "${args[@]}"

# petclinic density: ~60h
./run_density.py petclinic "${all_hosts}" -n "${density_runs}" "${args[@]}"
./run_density.py petclinic "${all_hosts}" -s -n "${density_runs}" "${args[@]}"
