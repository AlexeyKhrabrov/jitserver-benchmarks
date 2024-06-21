#!/bin/bash

set -e -u -o pipefail


packages=("docker.io")
other_packages=("ca-certificates" "wget")

apt-get update
apt-get install -y "${packages[@]}"
apt-get install -y --no-install-recommends "${other_packages[@]}"
rm -rf "/var/lib/apt/lists/"*


# Allow docker without sudo
groupadd "docker" || true
usermod -aG "docker" "${SUDO_USER}"
