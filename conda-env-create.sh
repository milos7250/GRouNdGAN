#!/bin/bash

source ~/.bashrc # to ensure conda is available

set -euo pipefail

conda env create -f ./environment.yml
conda activate groundgan

# Dependencies need to be installed in two phases, as sparselinear needs to be installed after torch 
# pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu torch==2.9.1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130 torch==2.9.1
pip install -r requirements2.txt --no-build-isolation

patch "$(dirname "$(which python)")/../lib/python3.11/site-packages/arboreto/core.py" ./arboreto.patch

# Ask user if they want to install developer dependencies, default to no
read -p "Do you want to install developer dependencies? (y/N) " -n 1 -r
echo    # move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -r requirements-dev.txt
fi