#!/bin/bash

set -e -u -o pipefail


db2_addr="${1}"
db2_port="${2}"
http_port="${3}"
jms_port="${4}"
iiop_port="${5}"


sed -i -e "s/db2addr=.*/db2addr=${db2_addr}/g" \
       -e "s/db2port=.*/db2port=${db2_port}/g" \
    "config/bootstrap.properties"

sed -i -e "s/httpPort=\".*\"/httpPort=\"${http_port}\"/g" \
       -e "s/wasJmsPort=\".*\"/wasJmsPort=\"${jms_port}\"/g" \
       -e "s/iiopPort=\".*\"/iiopPort=\"${iiop_port}\"/g" \
    "config/server.xml"


printf "JVM start timestamp: "
date -u "+%FT%T.%N"

exec "/opt/ibm/helpers/runtime/docker-server.sh" \
     "/opt/ibm/wlp/bin/server" run defaultServer
