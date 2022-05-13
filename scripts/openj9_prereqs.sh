#!/bin/bash

set -e -u -o pipefail


#NOTE: the list is for Ubuntu 18.04 and is not guaranteed to be complete
packages=(
	"autoconf" "automake" "build-essential" "ccache" "cmake" "cpio"
	"curl" "git" "libasound2-dev" "libcups2-dev" "libdwarf-dev"
	"libelf-dev" "libfontconfig1-dev" "libnuma-dev" "libssl-dev" "libtool"
	"libxrandr-dev" "libxt-dev" "libxtst-dev" "nasm" "pkg-config"
	"software-properties-common" "ssh" "unzip" "wget" "zip"
)

apt-get update
apt-get install -y --no-install-recommends "${packages[@]}"
rm -rf "/var/lib/apt/lists/"*
