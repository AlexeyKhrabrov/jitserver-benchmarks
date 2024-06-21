#!/bin/bash

set -e -u -o pipefail


usage_str="\
Usage: ${0} hosts_file runs multi_runs [logs_path]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


if (( $# < 3 )); then usage; fi

hosts="${1}"
runs="${2}"
multi_runs="${3}"

#NOTE: This option skips complete experiment runs and only runs the missing ones
args=(--skip-complete)

if (( $# >= 4 )); then
	logs_path="${4}"
	args+=(-L "${logs_path}")
else
	logs_path="logs"
fi

multi_args=("${args[@]}")
args+=(-n "${runs}")
multi_args+=(-n "${multi_runs}")


# all single 1st: ~2h
#
# total time (sec) for all jit modes for 1cpu 4g:
#
# 0: naive-bayes       1209                                                               = 1209
# 1: als               467  + finagle-http ! 142 + scrabble         61                    = 670 !
#
# 2: finagle-chirper ! 129  + page-rank      432 + scala-doku       113                   = 674 !
# 3: chi-square        195  + db-shootout    417 + scala-stm-bench7 52                    = 664
#
# 4: dotty !           216  + mnemonics      88  + reactors         367                   = 671 !
# 5: gauss-mix         223  + movie-lens     366 + par-mnemonics    88                    = 677
#
# 6: akka-uct          365  + fj-kmeans      249 + rx-scrabble      39  + scala-kmeans 27 = 680
# 7: dec-tree !        253  + future-genetic 114 + log-regression   276 + philosophers 30 = 673 !

./host_cleanup.py renaissance "${hosts}"

./run_renaissance_single.py "naive-bayes"      "${hosts}" 0 "${args[@]}" &

./run_renaissance_single.py "als"              "${hosts}" 1 "${args[@]}" && \
./run_renaissance_single.py "finagle-http"     "${hosts}" 1 "${args[@]}" && \
./run_renaissance_single.py "scrabble"         "${hosts}" 1 "${args[@]}" &

./run_renaissance_single.py "finagle-chirper"  "${hosts}" 2 "${args[@]}" && \
./run_renaissance_single.py "page-rank"        "${hosts}" 2 "${args[@]}" && \
./run_renaissance_single.py "scala-doku"       "${hosts}" 2 "${args[@]}" &

./run_renaissance_single.py "chi-square"       "${hosts}" 3 "${args[@]}" && \
./run_renaissance_single.py "db-shootout"      "${hosts}" 3 "${args[@]}" && \
./run_renaissance_single.py "scala-stm-bench7" "${hosts}" 3 "${args[@]}" &

./run_renaissance_single.py "dotty"            "${hosts}" 4 "${args[@]}" && \
./run_renaissance_single.py "mnemonics"        "${hosts}" 4 "${args[@]}" && \
./run_renaissance_single.py "reactors"         "${hosts}" 4 "${args[@]}" &

./run_renaissance_single.py "gauss-mix"        "${hosts}" 5 "${args[@]}" && \
./run_renaissance_single.py "movie-lens"       "${hosts}" 5 "${args[@]}" && \
./run_renaissance_single.py "par-mnemonics"    "${hosts}" 5 "${args[@]}" &

./run_renaissance_single.py "akka-uct"         "${hosts}" 6 "${args[@]}" && \
./run_renaissance_single.py "fj-kmeans"        "${hosts}" 6 "${args[@]}" && \
./run_renaissance_single.py "rx-scrabble"      "${hosts}" 6 "${args[@]}" && \
./run_renaissance_single.py "scala-kmeans"     "${hosts}" 6 "${args[@]}" &

./run_renaissance_single.py "dec-tree"         "${hosts}" 7 "${args[@]}" && \
./run_renaissance_single.py "future-genetic"   "${hosts}" 7 "${args[@]}" && \
./run_renaissance_single.py "log-regression"   "${hosts}" 7 "${args[@]}" && \
./run_renaissance_single.py "philosophers"     "${hosts}" 7 "${args[@]}" &

wait


# all single full: ~6h
#
# total time (sec) for all jit modes for 1cpu 4g:
#
# 0: als            3046 + scala-doku        558  + scala-kmeans     122  = 3726
# 1: dotty !        1900 + fj-kmeans         1666 + rx-scrabble !    182  = 3748 !
#
# 2: future-genetic 923  + naive-bayes       2596 + philosophers     185  = 3704
# 3: akka-uct       2284 + finagle-chirper ! 1159 + par-mnemonics    451  = 3894 !
#
# 4: dec-tree !     1237 + reactors          2138 + scala-stm-bench7 454  = 3829 !
# 5: db-shootout    2068 + page-rank         1277 + scrabble         456  = 3801
#
# 6: finagle-http ! 489  + gauss-mix         1379 + log-regression ! 1810 = 3678 !
# 7: chi-square     811  + mnemonics         452  + movie-lens       2552 = 3815

./host_cleanup.py renaissance "${hosts}"

./run_renaissance_single.py "als"              "${hosts}" 0 -F "${args[@]}" && \
./run_renaissance_single.py "scala-doku"       "${hosts}" 0 -F "${args[@]}" && \
./run_renaissance_single.py "scala-kmeans"     "${hosts}" 0 -F "${args[@]}" &

./run_renaissance_single.py "dotty"            "${hosts}" 1 -F "${args[@]}" && \
./run_renaissance_single.py "fj-kmeans"        "${hosts}" 1 -F "${args[@]}" && \
./run_renaissance_single.py "rx-scrabble"      "${hosts}" 1 -F "${args[@]}" &

./run_renaissance_single.py "future-genetic"   "${hosts}" 2 -F "${args[@]}" && \
./run_renaissance_single.py "naive-bayes"      "${hosts}" 2 -F "${args[@]}" && \
./run_renaissance_single.py "philosophers"     "${hosts}" 2 -F "${args[@]}" &

./run_renaissance_single.py "akka-uct"         "${hosts}" 3 -F "${args[@]}" && \
./run_renaissance_single.py "finagle-chirper"  "${hosts}" 3 -F "${args[@]}" && \
./run_renaissance_single.py "par-mnemonics"    "${hosts}" 3 -F "${args[@]}" &

./run_renaissance_single.py "dec-tree"         "${hosts}" 4 -F "${args[@]}" && \
./run_renaissance_single.py "reactors"         "${hosts}" 4 -F "${args[@]}" && \
./run_renaissance_single.py "scala-stm-bench7" "${hosts}" 4 -F "${args[@]}" &

./run_renaissance_single.py "db-shootout"      "${hosts}" 5 -F "${args[@]}" && \
./run_renaissance_single.py "page-rank"        "${hosts}" 5 -F "${args[@]}" && \
./run_renaissance_single.py "scrabble"         "${hosts}" 5 -F "${args[@]}" &

./run_renaissance_single.py "gauss-mix"        "${hosts}" 6 -F "${args[@]}" && \
./run_renaissance_single.py "finagle-http"     "${hosts}" 6 -F "${args[@]}" && \
./run_renaissance_single.py "log-regression"   "${hosts}" 6 -F "${args[@]}" &

./run_renaissance_single.py "chi-square"       "${hosts}" 7 -F "${args[@]}" && \
./run_renaissance_single.py "mnemonics"        "${hosts}" 7 -F "${args[@]}" && \
./run_renaissance_single.py "movie-lens"       "${hosts}" 7 -F "${args[@]}" &

wait


# all multi 1st: ~16h per run

# total time (min) for all jit modes for 1cpu 4g:

# 0: akka-uct       105 + chi-square      75  + db-shootout      110 + dec-tree    110 + 
#    dotty          95  + finagle-http    85  + future-genetic   45  + mnemonics   45  +
#    naive-bayes    180 + par-mnemonics   45  + philosophers     25  + rx-scrabble 35  = 955

# 1: als            135 + finagle-chirper 85  + fj-kmeans        65  + gauss-mix   90  +
#    log-regression 100 + movie-lens      125 + page-rank        110 + reactors    105 +
#    scala-doku     45  + scala-kmeans    25  + scala-stm-bench7 35  + scrabble    35  = 955

./run_renaissance_multi.py "akka-uct"         "${hosts}" 0 "${multi_args[@]}" && \
./run_renaissance_multi.py "chi-square"       "${hosts}" 0 "${multi_args[@]}" && \
./run_renaissance_multi.py "db-shootout"      "${hosts}" 0 "${multi_args[@]}" && \
./run_renaissance_multi.py "dec-tree"         "${hosts}" 0 "${multi_args[@]}" && \
./run_renaissance_multi.py "dotty"            "${hosts}" 0 "${multi_args[@]}" && \
./run_renaissance_multi.py "finagle-http"     "${hosts}" 0 "${multi_args[@]}" && \
./run_renaissance_multi.py "future-genetic"   "${hosts}" 0 "${multi_args[@]}" && \
./run_renaissance_multi.py "mnemonics"        "${hosts}" 0 "${multi_args[@]}" && \
./run_renaissance_multi.py "naive-bayes"      "${hosts}" 0 "${multi_args[@]}" && \
./run_renaissance_multi.py "par-mnemonics"    "${hosts}" 0 "${multi_args[@]}" && \
./run_renaissance_multi.py "philosophers"     "${hosts}" 0 "${multi_args[@]}" && \
./run_renaissance_multi.py "rx-scrabble"      "${hosts}" 0 "${multi_args[@]}" &

./run_renaissance_multi.py "als"              "${hosts}" 1 "${multi_args[@]}" && \
./run_renaissance_multi.py "finagle-chirper"  "${hosts}" 1 "${multi_args[@]}" && \
./run_renaissance_multi.py "fj-kmeans"        "${hosts}" 1 "${multi_args[@]}" && \
./run_renaissance_multi.py "gauss-mix"        "${hosts}" 1 "${multi_args[@]}" && \
./run_renaissance_multi.py "log-regression"   "${hosts}" 1 "${multi_args[@]}" && \
./run_renaissance_multi.py "movie-lens"       "${hosts}" 1 "${multi_args[@]}" && \
./run_renaissance_multi.py "page-rank"        "${hosts}" 1 "${multi_args[@]}" && \
./run_renaissance_multi.py "reactors"         "${hosts}" 1 "${multi_args[@]}" && \
./run_renaissance_multi.py "scala-doku"       "${hosts}" 1 "${multi_args[@]}" && \
./run_renaissance_multi.py "scala-kmeans"     "${hosts}" 1 "${multi_args[@]}" && \
./run_renaissance_multi.py "scala-stm-bench7" "${hosts}" 1 "${multi_args[@]}" && \
./run_renaissance_multi.py "scrabble"         "${hosts}" 1 "${multi_args[@]}" &

wait


# all multi full: ~18h per run

# total time (min) for all jit modes for 1cpu 4g:

# 0: akka-uct       105 + chi-square      75  + db-shootout      110 + dec-tree    110 + 
#    dotty          95  + finagle-http    85  + future-genetic   45  + mnemonics   45  +
#    naive-bayes    180 + par-mnemonics   45  + philosophers     25  + rx-scrabble 35  = 955
#
# 1: als            135 + finagle-chirper 85  + fj-kmeans        65  + gauss-mix   90  +
#    log-regression 100 + movie-lens      125 + page-rank        110 + reactors    105 +
#    scala-doku     45  + scala-kmeans    25  + scala-stm-bench7 35  + scrabble    35  = 955

./run_renaissance_multi.py "akka-uct"         "${hosts}" 0 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "chi-square"       "${hosts}" 0 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "db-shootout"      "${hosts}" 0 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "dec-tree"         "${hosts}" 0 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "dotty"            "${hosts}" 0 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "finagle-http"     "${hosts}" 0 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "future-genetic"   "${hosts}" 0 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "mnemonics"        "${hosts}" 0 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "naive-bayes"      "${hosts}" 0 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "par-mnemonics"    "${hosts}" 0 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "philosophers"     "${hosts}" 0 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "rx-scrabble"      "${hosts}" 0 -F "${multi_args[@]}" &

./run_renaissance_multi.py "als"              "${hosts}" 1 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "finagle-chirper"  "${hosts}" 1 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "fj-kmeans"        "${hosts}" 1 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "gauss-mix"        "${hosts}" 1 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "log-regression"   "${hosts}" 1 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "movie-lens"       "${hosts}" 1 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "page-rank"        "${hosts}" 1 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "reactors"         "${hosts}" 1 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "scala-doku"       "${hosts}" 1 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "scala-kmeans"     "${hosts}" 1 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "scala-stm-bench7" "${hosts}" 1 -F "${multi_args[@]}" && \
./run_renaissance_multi.py "scrabble"         "${hosts}" 1 -F "${multi_args[@]}" &

wait
