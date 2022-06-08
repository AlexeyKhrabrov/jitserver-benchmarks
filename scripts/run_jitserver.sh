#!/bin/bash

set -e -u -o pipefail


usage_str="\
Usage: ${0} jdk_dir jdk_ver [-d|--debug] [-g|--gdb] [-b|--break]
       [-s|--stderr] [-c|--cache] [<jitserver args>]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


if (( $# < 2 )); then usage; fi

jdk_dir=$(readlink -f "${1}")
jdk_ver="${2}"
debug=false
use_gdb=false
do_break=false
stderr_vlog=false
sanity=false

jitserver_args=("-Xshareclasses:none" "-Xdump:jit:events=user")
jit_opts=("verbose={failures|compilePerformance|JITServer}")

for arg in "${@:3}"; do
	case "${arg}" in
		"-h" | "--help" )
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
			jitserver_args+=("-XX:+JITServerUseAOTCache")
			;;
		*)
			jitserver_args+=("${arg}")
			;;
	esac
done


if [[ "${stderr_vlog}" != true ]]; then
	jit_opts+=("vlog=vlog_server")
fi

function join() { local IFS="${1}"; shift; echo "$*"; }

if [[ ${#jit_opts[@]} > 0 ]]; then
	opts="$(join "," "${jit_opts[@]}")"
	jitserver_args+=("-Xjit:${opts}" "-Xaot:${opts}")
fi


if [[ "${debug}" == true ]]; then
	debug_level="slowdebug"
else
	debug_level="release"
fi

build_dir="${jdk_dir}/openj9-openjdk-jdk${jdk_ver}/build/${debug_level}"
jitserver="${build_dir}/images/jdk/bin/jitserver"


export TR_PrintCompStats=1
export TR_PrintCompTime=1
export TR_PrintJITServerMsgStats=1
export TR_PrintJITServerAOTCacheStats=1
export TR_PrintResourceUsageStats=1

if [[ "${use_gdb}" == true ]]; then
	gdb_args=("-ex" "handle SIGPIPE nostop noprint pass")
	if [[ "${do_break}" != true ]]; then
		gdb_args+=("-ex" "run")
	fi
	gdb "${gdb_args[@]}" --args "${jitserver}" "${jitserver_args[@]}"
else
	"${jitserver}" "${jitserver_args[@]}"
fi
