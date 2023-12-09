#!/bin/bash

set -e -u -o pipefail


python_packages=("pip" "setuptools" "matplotlib" "pandas")

pip3 install --upgrade "${python_packages[@]}"
