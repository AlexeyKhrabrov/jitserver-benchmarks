#!/bin/bash

set -e -u -o pipefail


# Usage: ./all_experiments.sh all_hosts_file main_hosts_file [logs_path]

all_hosts="${1}" # 8 main homogeneous machines
main_hosts="${2}" # total 11 machines

args=()
if (( $# >= 3 )); then
	logs_path="${3}"
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

./run_single.py daytrader "${main_hosts}" 0 0 "${args[@]}" &
./run_single.py daytrader "${main_hosts}" 2 1 "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 4 1 "${args[@]}" &
./run_single.py daytrader "${main_hosts}" 1 2 "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 5 2 "${args[@]}" &
./run_single.py daytrader "${main_hosts}" 3 3 "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 6 3 "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 7 3 "${args[@]}" &

./run_single.py acmeair "${main_hosts}" 0 4 "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 2 4 "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 4 4 "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 6 4 "${args[@]}" &
./run_single.py acmeair "${main_hosts}" 1 5 "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 3 5 "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 5 5 "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 7 5 "${args[@]}" &

./run_single.py petclinic "${main_hosts}" 0 6 "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 2 6 "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 4 6 "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 6 6 "${args[@]}" &
./run_single.py petclinic "${main_hosts}" 1 7 "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 3 7 "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 5 7 "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 7 7 "${args[@]}" &

wait

./run_single.py daytrader "${main_hosts}" 0 0 -j "${args[@]}" &
./run_single.py daytrader "${main_hosts}" 2 1 -j "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 4 1 -j "${args[@]}" &
./run_single.py daytrader "${main_hosts}" 1 2 -j "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 5 2 -j "${args[@]}" &
./run_single.py daytrader "${main_hosts}" 3 3 -j "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 6 3 -j "${args[@]}" && \
./run_single.py daytrader "${main_hosts}" 7 3 -j "${args[@]}" &

./run_single.py acmeair "${main_hosts}" 0 4 -j "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 2 4 -j "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 4 4 -j "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 6 4 -j "${args[@]}" &
./run_single.py acmeair "${main_hosts}" 1 5 -j "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 3 5 -j "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 5 5 -j "${args[@]}" && \
./run_single.py acmeair "${main_hosts}" 7 5 -j "${args[@]}" &

./run_single.py petclinic "${main_hosts}" 0 6 -j "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 2 6 -j "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 4 6 -j "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 6 6 -j "${args[@]}" &
./run_single.py petclinic "${main_hosts}" 1 7 -j "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 3 7 -j "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 5 7 -j "${args[@]}" && \
./run_single.py petclinic "${main_hosts}" 7 7 -j "${args[@]}" &

wait


# all latency localjit: ~1.5h
./run_latency.py acmeair "${main_hosts}" 0 -l -j -S "${args[@]}" <<< "${password}" &
./run_latency.py daytrader "${main_hosts}" 1 -l -j -S "${args[@]}" <<< "${password}" &
./run_latency.py petclinic "${main_hosts}" 2 -l -j -S "${args[@]}" <<< "${password}" &
wait

# acmeair latency: ~4h
./run_latency.py acmeair "${main_hosts}" 0 -j -S "${args[@]}" <<< "${password}" &
./run_latency.py acmeair "${main_hosts}" 1 -j -S "${args[@]}" <<< "${password}" &
./run_latency.py acmeair "${main_hosts}" 2 -j -S "${args[@]}" <<< "${password}" &
./run_latency.py acmeair "${main_hosts}" 3 -j -S "${args[@]}" <<< "${password}" &
wait

# daytrader latency: ~9h
./run_latency.py daytrader "${main_hosts}" 0 -j -S "${args[@]}" <<< "${password}" &
./run_latency.py daytrader "${main_hosts}" 1 -j -S "${args[@]}" <<< "${password}" &
./run_latency.py daytrader "${main_hosts}" 2 -j -S "${args[@]}" <<< "${password}" &
./run_latency.py daytrader "${main_hosts}" 3 -j -S "${args[@]}" <<< "${password}" &
wait

# petclinic latency: ~3h
./run_latency.py petclinic "${main_hosts}" 0 -j -S "${args[@]}" <<< "${password}" &
./run_latency.py petclinic "${main_hosts}" 1 -j -S "${args[@]}" <<< "${password}" &
./run_latency.py petclinic "${main_hosts}" 2 -j -S "${args[@]}" <<< "${password}" &
./run_latency.py petclinic "${main_hosts}" 3 -j -S "${args[@]}" <<< "${password}" &
wait


# acmeair scale: ~8h
./run_scale.py acmeair "${all_hosts}" 0 -j "${args[@]}" &
wait
./run_scale.py acmeair "${all_hosts}" 1 -j "${args[@]}" &
./run_scale.py acmeair "${all_hosts}" 2 -j "${args[@]}" &
./run_scale.py acmeair "${all_hosts}" 3 -j "${args[@]}" &
wait
./run_scale.py acmeair "${all_hosts}" 4 -j "${args[@]}" &
./run_scale.py acmeair "${all_hosts}" 5 -j "${args[@]}" &
wait
./run_scale.py acmeair "${all_hosts}" 6 -j "${args[@]}" &
./run_scale.py acmeair "${all_hosts}" 7 -j "${args[@]}" &
wait
./run_scale.py acmeair "${all_hosts}" 9 -j "${args[@]}" &
./run_scale.py acmeair "${all_hosts}" 8 -j "${args[@]}" &
wait

# daytrader scale: ~21h
./run_scale.py daytrader "${all_hosts}" 0 -j "${args[@]}" &
wait
./run_scale.py daytrader "${all_hosts}" 1 -j "${args[@]}" &
wait
./run_scale.py daytrader "${all_hosts}" 2 -j "${args[@]}" &
./run_scale.py daytrader "${all_hosts}" 3 -j "${args[@]}" &
wait
./run_scale.py daytrader "${all_hosts}" 4 -j "${args[@]}" &
./run_scale.py daytrader "${all_hosts}" 5 -j "${args[@]}" &
wait
./run_scale.py daytrader "${all_hosts}" 6 -j "${args[@]}" &
./run_scale.py daytrader "${all_hosts}" 7 -j "${args[@]}" &
./run_scale.py daytrader "${all_hosts}" 8 -j "${args[@]}" &
./run_scale.py daytrader "${all_hosts}" 9 -j "${args[@]}" &
wait

# petclinic scale: ~5h
./run_scale.py petclinic "${all_hosts}" 0 -j "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 1 -j "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 2 -j "${args[@]}" &
wait
./run_scale.py petclinic "${all_hosts}" 3 -j "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 4 -j "${args[@]}" &
wait
./run_scale.py petclinic "${all_hosts}" 5 -j "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 6 -j "${args[@]}" &
wait
./run_scale.py petclinic "${all_hosts}" 7 -j "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 8 -j "${args[@]}" &
./run_scale.py petclinic "${all_hosts}" 9 -j "${args[@]}" &
wait
./run_scale.py petclinic "${all_hosts}" 10 -j "${args[@]}" &
wait


# acmeair density: ~60h
./run_density.py acmeair "${all_hosts}" "${args[@]}"
./run_density.py acmeair "${all_hosts}" -s "${args[@]}"

# daytrader density: ~60h
./run_density.py daytrader "${all_hosts}" "${args[@]}"
./run_density.py daytrader "${all_hosts}" -s "${args[@]}"

# petclinic density: ~60h
./run_density.py petclinic "${all_hosts}" "${args[@]}"
./run_density.py petclinic "${all_hosts}" -s "${args[@]}"
