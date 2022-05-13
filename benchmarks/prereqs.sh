#!/bin/bash

set -e -u -o pipefail


packages=("openssh-client" "python3-pip" "rsync" "sshpass")

apt-get update
apt-get install -y --no-install-recommends "${packages[@]}"
rm -rf "/var/lib/apt/lists/"*
