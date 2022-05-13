#!/bin/bash

set -e -u -o pipefail

dir="$(dirname "$(readlink -f "${BASH_SOURCE}")")"


usage_str="\
Usage: ${0} [-t|--tune]"

function usage()
{
	echo "${usage_str}" 1>&2
	exit 1
}


tune=false

for arg in "${@:1}"; do
	case "${arg}" in
		"-t" | "--tune" )
			tune=true
			;;
		*)
			usage
			;;
	esac
done


# Cleanup before exit in case of failures
function cleanup()
{
	docker rm -f "liberty-tmp" "db2-tmp" &> "/dev/null" || true
}

trap cleanup EXIT


# Start temporary db2 container and wait until it's ready
docker run --name="db2-tmp" --net=host --cap-add=IPC_OWNER -d "db2-base" 50000
while [[ "$(docker container inspect -f '{{.State.Running}}' db2-tmp)" != "true" ]]; do
	sleep 0.1
done
(docker logs -f "db2-tmp" &) | timeout 10 grep -q \
	"DB2START processing was successful"

# Copy required files into container
docker cp "${dir}/tradedb/TradeDB.ddl" "db2-tmp":"/"
docker cp "${dir}/tradedb/createTradeDB.sh" "db2-tmp":"/"
docker cp "${dir}/tradedb/tuneTradeDB.sh" "db2-tmp":"/"

# Create database
docker exec "db2-tmp" "/createTradeDB.sh"

# Start temporary liberty container and wait until it's ready
docker run --name="liberty-tmp" --net=host -d --rm \
       "liberty-daytrader" "localhost" 50000 9080 7276 2809
while [[ "$(docker container inspect -f '{{.State.Running}}' liberty-tmp)" != "true" ]]; do
	sleep 0.1
done
(docker logs -f "liberty-tmp" &) | timeout 60 grep -q \
	"The defaultServer server started in"

# Populate database
wget -O- --post-data='action=updateConfig&RunTimeMode=1' \
     "http://localhost:9080/daytrader/config"
wget -O- "http://localhost:9080/daytrader/config?action=buildDB"

# Tune database if requested
if [[ "${tune}" == true ]]; then
	docker exec "db2-tmp" "/tuneTradeDB.sh"
fi

# Delete unnecessary files
docker exec -u root "db2-tmp" rm -f \
       "/TradeDB.ddl" "/createTradeDB.sh" "/tuneTradeDB.sh"

# Commit new image
docker stop "liberty-tmp" "db2-tmp"
docker commit "db2-tmp" "db2-daytrader"
