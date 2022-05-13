#!/bin/bash

set -e -u -o pipefail


port="${1}"

db_name=""
if (( $# > 1 )); then
	db_name="${2}"
fi


sed -i "s/50000/${port}/g" "/etc/services"

. ~/"sqllib/db2profile"


trap '{ db2stop force; exit; }' SIGINT SIGTERM
db2start

if [[ "${db_name}" != "" ]]; then
	db2 activate db "${db_name}"
fi

sleep infinity &
wait $!
