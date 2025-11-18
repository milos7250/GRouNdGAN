#!/bin/bash
set -euo pipefail

conda env create -f ./environment.yml
conda activate groundgan
patch "$(dirname "$(which python)")/../lib/python3.11/site-packages/arboreto/core.py" ./arboreto.patch