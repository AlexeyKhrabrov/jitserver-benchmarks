#!/bin/bash

set -e -u -o pipefail

dir="$(dirname "$(readlink -f "${BASH_SOURCE}")")"


usage_str="\
Usage: ${0} [db2_installer_path]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


for arg in "$@"; do
	if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then usage; fi
done

if (( $# == 1 )); then
	db2_installer_path="${1}"
elif (( $# == 0 )); then
	db2_installer_path=""
else
	usage
fi


# Cleanup before exit in case of failures
function cleanup()
{
	if [[ "${db2_installer_path}" != "" ]]; then
		rm -f "${dir}/db2_base_installer/db2.tar.gz"
	else
		docker rm -f "db2-tmp" &> "/dev/null" || true
	fi
}

trap cleanup EXIT


if [[ "${db2_installer_path}" != "" ]]; then
	# Create link to installer inside docker build context
	link "${db2_installer_path}" "${dir}/db2_base_installer/db2.tar.gz"

	# Do not use buildkit since it doesn't support host network build
	DOCKER_BUILDKIT=0 \
	docker build --network=host -t "db2-base" "${dir}/db2_base_installer"

else
	docker pull "ibmcom/db2"

	# Modify original db2 container image to remove volumes
	args=(FROM "ibmcom/db2" INTO "db2-no-volumes" REMOVE ALL VOLUMES)
	docker-copyedit.py "${args[@]}" || "${dir}/docker-copyedit.py" "${args[@]}"
	rm -rf "load.tmp"

	# Build next intermediate image with modified entrypoint script
	docker build -t "db2-fixed-entrypoint" "${dir}/db2_fixed_entrypoint"

	# Start temporary db2 container and wait until it's ready and setup is complete
	docker run --name="db2-tmp" -d --net=host --privileged \
	       -e LICENSE="accept" -e DB2INST1_PASSWORD="p@ssw0rd" "db2-no-volumes"
	while [[ "$(docker container inspect -f '{{.State.Running}}' db2-tmp)" != "true" ]]; do
		sleep 0.1
	done
	(docker logs -f "db2-tmp" &) | timeout 180 grep -q "Setup has completed."
	sleep 1

	# Commit next intermediate image
	docker stop "db2-tmp"
	docker commit "db2-tmp" "db2-setup"

	# Build final image
	docker build -t "db2-base" "${dir}/db2_base_container"
fi
