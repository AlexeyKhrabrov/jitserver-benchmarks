#!/bin/bash

set -e -u -o pipefail


usage_str="\
Usage: ${0} jdk_dir jdk_ver [-d|--debug] [-g|--gdb] [-b|--break]
       [-v|--valgrind] [-s|--stderr-vlog] [-a|--aotcache] [<jitserver args>]"

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
use_valgrind=false
stderr_vlog=false
use_aotcache=false

jitserver_args=("-Xshareclasses:none" "-Xdump:jit:events=user" "-XX:JITServerTimeout=0")

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
		"-a" | "--aotcache" )
			use_aotcache=true
			jitserver_args+=("-XX:+JITServerUseAOTCache")
			;;
		*)
			jitserver_args+=("${arg}")
			;;
	esac
done



if [[ "${debug}" == true ]]; then
	debug_level="slowdebug"
else
	debug_level="release"
fi

build_dir="${jdk_dir}/openj9-openjdk-jdk${jdk_ver}/build/${debug_level}"
jitserver="${build_dir}/images/jdk/bin/jitserver"


vlog_opts="verbose={failures|compilePerformance|JITServer}"
if [[ "${stderr_vlog}" != true ]]; then
	vlog_opts+=",vlog=vlog_server"
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
	gdb "${gdb_args[@]}" --args "${jitserver}" "${jitserver_args[@]}"
elif [[ "${use_valgrind}" == true ]]; then
	valgrind --vgdb=yes --vgdb-error=0 "${jitserver}" "${jitserver_args[@]}"
else
	"${jitserver}" "${jitserver_args[@]}"
fi
