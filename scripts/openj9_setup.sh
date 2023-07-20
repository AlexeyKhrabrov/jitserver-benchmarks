#!/bin/bash

set -e -u -o pipefail


usage_str="\
Usage: ${0} jdk_dir jdk_ver bootjdk_ver openj9_dir omr_dir jdk_repo_url
       jdk_repo_branch [-c|--clean] [-u|--update] [-g|--generate-keys]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


if (( $# < 7 )); then usage; fi

jdk_dir=$(readlink -f "${1}")
jdk_ver="${2}"
bootjdk_ver="${3}"
openj9_dir=$(readlink -f "${4}")
omr_dir=$(readlink -f "${5}")
jdk_repo_url="${6}"
jdk_repo_branch="${7}"
clean=false
update=false
generate_keys=false

for arg in "${@:8}"; do
	case "${arg}" in
		"-c" | "--clean" )
			clean=true
			;;
		"-u" | "--update" )
			update=true
			clean=true
			;;
		"-g" | "--generate_keys" )
			generate_keys=true
			;;
		*)
			usage
			;;
	esac
done


mkdir -p "${jdk_dir}"

bootjdk_url="https://api.adoptopenjdk.net/v3/binary/latest/${bootjdk_ver}/\
ga/linux/x64/jdk/openj9/normal/adoptopenjdk?project=jdk"

bootjdk_dir="${jdk_dir}/bootjdk${jdk_ver}"

if [[ ! -d "${bootjdk_dir}" ]]; then
	archive="${jdk_dir}/bootjdk${jdk_ver}.tar.gz"
	wget "${bootjdk_url}" -O "${archive}"
	mkdir -p "${bootjdk_dir}"
	tar -xzf "${archive}" -C "${bootjdk_dir}" --strip-components=1
	rm -f "${archive}" ~/".wget-hsts"
fi


if [[ "${jdk_repo_url}" == "" ]]; then
	jdk_repo_url="https://github.com/ibmruntimes/openj9-openjdk-jdk${jdk_ver}"
	jdk_repo_branch="openj9"
fi

jdk_src_dir="${jdk_dir}/openj9-openjdk-jdk${jdk_ver}"

if [[ ! -d "${jdk_src_dir}" ]]; then
	git clone -b "${jdk_repo_branch}" "${jdk_repo_url}" "${jdk_src_dir}"
	pushd "${jdk_src_dir}"
	bash "./get_source.sh"
	popd #"${jdk_src_dir}"
fi


if [[ "${clean}" == true ]]; then
	git -C "${jdk_src_dir}" reset --hard
	git -C "${jdk_src_dir}" clean -dfx

	git -C "${jdk_src_dir}/openj9" reset --hard
	git -C "${jdk_src_dir}/openj9" clean -dfx
	rm -rf "${jdk_src_dir}/openj9/test/TKG"

	git -C "${jdk_src_dir}/omr" reset --hard
	git -C "${jdk_src_dir}/omr" clean -dfx
fi

if [[ "${update}" == true ]]; then
	git -C "${jdk_src_dir}" pull
	git -C "${jdk_src_dir}/openj9" pull
	git -C "${jdk_src_dir}/omr" pull
fi

rsync -a --delete --exclude=".git/" "${omr_dir}/" "${jdk_src_dir}/omr/"
rsync -a --delete --exclude=".git/" "${openj9_dir}/" "${jdk_src_dir}/openj9/"


if [[ "${generate_keys}" == true ]]; then
	# Generate a key pair and a self-signed certificate for testing encryption
	openssl req -x509 -sha256 -newkey rsa:2048 -keyout "${jdk_dir}/key.pem" \
	            -out "${jdk_dir}/cert.pem" -days 365 -nodes -subj "/CN=localhost"
fi
