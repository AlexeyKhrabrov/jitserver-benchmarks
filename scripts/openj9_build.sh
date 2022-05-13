#!/bin/bash

set -e -u -o pipefail


usage_str="\
Usage: ${0} jdk_dir jdk_ver [-c|--configure] [-d|--debug]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


if (( $# < 2 )); then usage; fi

jdk_dir=$(readlink -f "${1}")
jdk_ver="${2}"
configure=false
debug=false

for arg in "${@:3}"; do
	case "${arg}" in
		"-c" | "--configure" )
			configure=true
			;;
		"-d" | "--debug" )
			debug=true
			;;
		*)
			usage
			;;
	esac
done


jdk_src_dir="${jdk_dir}/openj9-openjdk-jdk${jdk_ver}"

if [[ "${debug}" == true ]]; then
	debug_level="slowdebug"
else
	debug_level="release"
fi

build_dir="${jdk_src_dir}/build/${debug_level}"


if [[ "${configure}" == true || ! -d "${build_dir}" ]]; then
	rm -rf "${build_dir}"
	pushd "${jdk_src_dir}"

	bash "./configure" --enable-ccache --enable-jitserver --with-cmake \
	                   --with-boot-jdk="${jdk_dir}/bootjdk${jdk_ver}" \
	                   --with-conf-name="${debug_level}" \
	                   --with-debug-level="${debug_level}" \
	                   --with-native-debug-symbols=internal
	rm -f "a.out"

	popd #"${jdk_src_dir}"
fi


extra_cflags=("-ggdb3")
jit_extra_cflags=("-DMESSAGE_SIZE_STATS")

if [[ "${debug}" == true ]]; then
	extra_cflags+=("-Og" "-fno-inline")
	# Do not define DEBUG for the rest of the VM since it results
	# in an enormous amount of debugging output on jdk11+
	jit_extra_cflags+=("-DDEBUG")
fi

cmake_args=(
	"-DCMAKE_VERBOSE_MAKEFILE=ON"
	"-DOMR_PLATFORM_C_COMPILE_OPTIONS=\"${extra_cflags[@]}\""
	"-DOMR_PLATFORM_CXX_COMPILE_OPTIONS=\"${extra_cflags[@]}\""
	"-DJ9JIT_EXTRA_CFLAGS=\"${jit_extra_cflags[@]}\""
	"-DJ9JIT_EXTRA_CXXFLAGS=\"${jit_extra_cflags[@]}\""
	"-DOMR_SEPARATE_DEBUG_INFO=OFF"
)

export EXTRA_CMAKE_ARGS="${cmake_args[@]}"

make -C "${jdk_src_dir}" CONF="${debug_level}" all


img_dir="${build_dir}/images"

if [[ "${jdk_ver}" == 8 ]]; then
	# For compatibility with later jdk versions
	rsync -a --delete "${img_dir}/j2sdk-image/" "${img_dir}/jdk/"
fi

"${img_dir}/jdk/bin/java" -Xshareclasses:destroyAll || true

# Copy the key pair and certificate for testing encryption (if present)
if [[ -f "${jdk_dir}/cert.pem" && -f "${jdk_dir}/key.pem" ]]; then
	cp -a "${jdk_dir}/cert.pem" "${jdk_dir}/key.pem" "${img_dir}/jdk/"
fi
