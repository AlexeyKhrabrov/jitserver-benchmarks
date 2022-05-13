#!/bin/bash

set -e -u -o pipefail


usage_str="\
Usage: ${0} jdk_dir jdk_ver jitserver_addr [-d|--debug] [-g|--gdb]
       [-b|--break] [-s|--stderr] [-c|--cache] [-p|--purge-scc] [<jvm args>]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


if (( $# < 3 )); then usage; fi

jdk_dir=$(readlink -f "${1}")
jdk_ver="${2}"
jitserver_addr="${3}"
debug=false
use_gdb=false
do_break=false
stderr_vlog=false
purge_scc=false

jvm_args=("-XX:+UseJITServer" "-XX:+RequireJITServer"
          "-XX:JITServerAddress=${jitserver_addr}")
jit_opts=("verbose={failures|compilePerformance|JITServer}")
extra_args=()

for arg in "${@:4}"; do
	case "${arg}" in
		"--help" )
			usage
			;;
		"-d" | "--debug" )
			debug=true
			;;
		"-g" | "--gdb" )
			use_gdb=true
			;;
		"-b" | "--break" )
			do_break=true
			;;
		"-s" | "--stderr" )
			stderr_vlog=true
			;;
		"-c" | "--cache" )
			jvm_args+=("-XX:+JITServerUseAOTCache")
			;;
		"-p" | "--purge-scc" )
			purge_scc=true
			;;
		*)
			extra_args+=("${arg}")
			;;
	esac
done

if [[ "${stderr_vlog}" != true ]]; then
	jit_opts+=("vlog=vlog_client")
fi


function join() { local IFS="${1}"; shift; echo "$*"; }

if [[ ${#jit_opts[@]} > 0 ]]; then
	opts="$(join "," "${jit_opts[@]}")"
	jvm_args+=("-Xjit:${opts}" "-Xaot:${opts}")
fi

jvm_args+=("${extra_args[@]}")


if [[ "${debug}" == true ]]; then
	debug_level="slowdebug"
else
	debug_level="release"
fi

build_dir="${jdk_dir}/openj9-openjdk-jdk${jdk_ver}/build/${debug_level}"
java="${build_dir}/images/jdk/bin/java"


if [[ "${purge_scc}" == true ]]; then
	"${java}" -Xshareclasses:destroyAll || true
fi


export TR_PrintCompStats=1
export TR_PrintCompTime=1
export TR_PrintJITServerMsgStats=1
export TR_PrintJITServerAOTCacheStats=1
export TR_PrintResourceUsageStats=1
export TR_PrintJITServerMallocStats=1

if [[ "${use_gdb}" == true ]]; then
	gdb_args=("-ex" "handle SIGPIPE nostop noprint pass")
	if [[ "${do_break}" != true ]]; then
		gdb_args+=("-ex" "run")
	fi
	gdb "${gdb_args[@]}" --args "${java}" "${jvm_args[@]}"
else
	"${java}" "${jvm_args[@]}"
fi
