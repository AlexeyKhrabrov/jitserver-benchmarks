#!/bin/bash

set -e -u -o pipefail


mongo_addr="${1}"
mongo_port="${2}"
http_port="${3}"


sed -i -e "s/hostname=.*/hostname=${mongo_addr}/g" \
       -e "s/port=.*/port=${mongo_port}/g" \
    "config/mongo.properties"

sed -i "s/httpPort=\".*\"/httpPort=\"${http_port}\"/g" "config/server.xml"


printf "JVM start timestamp: "
date -u "+%FT%T.%N"

exec "/opt/ibm/helpers/runtime/docker-server.sh" \
     "/opt/ibm/wlp/bin/server" run defaultServer
