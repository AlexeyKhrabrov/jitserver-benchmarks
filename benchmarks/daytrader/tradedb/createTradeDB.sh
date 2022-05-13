#!/bin/bash

set -e -u -o pipefail

. ~/"sqllib/db2profile"


db="tradedb"

if (( $# >= 1 )); then
	data_dir="${1}"
else
	data_dir=~/"TradeDB"
fi

if (( $# >= 2 )); then
	metadata_dir="${2}"
else
	metadata_dir=~/"TradeDB"
fi


mkdir -p "${data_dir}" "${metadata_dir}"

db2 create db "${db}" automatic storage yes on "${data_dir}" dbpath on \
    "${metadata_dir}" autoconfigure using admin_priority performance apply db only

db2 connect to "${db}"
db2 -tvf "/TradeDB.ddl"
db2 disconnect all
