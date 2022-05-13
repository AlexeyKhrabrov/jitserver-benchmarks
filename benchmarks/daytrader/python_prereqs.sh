#!/bin/bash

set -e -u -o pipefail

dir="$(dirname "$(readlink -f "${BASH_SOURCE}")")"


python_packages=("pip" "setuptools" "docker-copyedit")

if ! pip3 install "${python_packages[@]}"; then
	# Workaround in case installation using pip fails
	package="0c/2a/d8ac9f0a2d38273cef2330c1d3aff454de2e201594cdb196366ccd3d67e4"
	version="1.4.5016"
	wget "https://files.pythonhosted.org/packages/${package}/docker-copyedit-${version}.tar.gz"
	tar -xzf "docker-copyedit-${version}.tar.gz"
	mv "docker-copyedit-${version}/docker-copyedit.py" "${dir}/"
	rm -rf "docker-copyedit-${version}.tar.gz" "docker-copyedit-${version}/"
fi
