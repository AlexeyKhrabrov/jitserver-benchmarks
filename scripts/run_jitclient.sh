#!/bin/bash

set -e -u -o pipefail


usage_str="\
Usage: ${0} jdk_dir jdk_ver jitserver_addr [-d|--debug] [-g|--gdb]
       [-b|--break] [-v|--valgrind] [-s|--stderr-vlog] [-P|--purge-scc]
       [-a|--aotcache] [-p|--profilecache] [-e|--eager] [<jvm args>]"

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
use_valgrind=false
stderr_vlog=false
purge_scc=false
use_aotcache=false
use_profilecache=false
eager=false

jvm_args=("-XX:+UseJITServer" "-XX:+RequireJITServer" "-XX:JITServerAddress=${jitserver_addr}" "-XX:JITServerTimeout=0")
extra_args=()

for arg in "${@:4}"; do
	case "${arg}" in
		"-h" | "--help" )
			usage
			;;
		"-d" | "--debug" )
			debug=true
			;;
		"-g" | "--gdb" )
			use_gdb=true
			use_valgrind=false
			;;
		"-b" | "--break" )
			do_break=true
			;;
		"-v" | "--valgrind" )
			use_valgrind=true
			use_gdb=false
			;;
		"-s" | "--stderr-vlog" )
			stderr_vlog=true
			;;
		"-P" | "--purge-scc" )
			purge_scc=true
			;;
		"-a" | "--aotcache" )
			use_aotcache=true
			jvm_args+=("-XX:+JITServerUseAOTCache")
			;;
		"-p" | "--profilecache" )
			use_profilecache=true
			jvm_args+=("-XX:+JITServerShareProfilingData")
			;;
		"-e" | "--eager" )
			eager=true
			jvm_args+=("-XX:+JITServerPrefetchAllData" "-Xjit:aotPrefetcherDoNotRequestAtClassLoad")
			;;
		*)
			extra_args+=("${arg}")
			;;
	esac
done

if [[ "${eager}" == true ]]; then
	if [[ "${use_aotcache}" == true ]]; then
		jvm_args+=("-XX:+JITServerUseAOTPrefetcher")
	fi
	if [[ "${use_profilecache}" == true ]]; then
		jvm_args+=("-XX:+JITServerPreCompileProfiledMethods")
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


vlog_opts="verbose={failures|compilePerformance|JITServer}"
if [[ "${stderr_vlog}" != true ]]; then
	vlog_opts+=",vlog=vlog_client"
fi
export TR_Options="${vlog_opts}"

export TR_silentEnv=1
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
	gdb "${gdb_args[@]}" --args "${java}" "${jvm_args[@]}"
elif [[ "${use_valgrind}" == true ]]; then
	valgrind --vgdb=yes --vgdb-error=0 "${java}" "${jvm_args[@]}"
else
	"${java}" "${jvm_args[@]}"
fi
