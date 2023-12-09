#!/bin/bash

set -e -u -o pipefail

dir="$(dirname "$(readlink -f "${BASH_SOURCE}")")"


docker_package="docker.io"
other_packages=("ca-certificates" "python3-pip" "wget")

apt-get update
apt-get install -y "${docker_package}"
apt-get install -y --no-install-recommends "${other_packages[@]}"
rm -rf "/var/lib/apt/lists/"*


# Allow docker without sudo
groupadd "docker" || true
usermod -aG "docker" "${SUDO_USER}"


python_packages=("pip" "setuptools" "docker-copyedit")

pip3 install --upgrade "${python_packages[@]}" || true

# Workaround in case installation using pip fails - download package directly
package="0c/2a/d8ac9f0a2d38273cef2330c1d3aff454de2e201594cdb196366ccd3d67e4"
version="1.4.5016"
archive="docker-copyedit-${version}.tar.gz"
url="https://files.pythonhosted.org/packages/${package}/${archive}"

wget "${url}" && tar -xzf "${archive}" \
	&& mv "docker-copyedit-${version}/docker-copyedit.py" "${dir}/" \
	&& chown "${SUDO_USER}:$(id -gn ${SUDO_USER})" "${dir}/docker-copyedit.py" \
	|| true
rm -rf "${archive}" "docker-copyedit-${version}/" ~/".wget-hsts"

which docker-copyedit.py || test -f "${dir}/docker-copyedit.py" \
	|| { echo "Failed to install docker-copyedit" 1>&2; exit 1; }
