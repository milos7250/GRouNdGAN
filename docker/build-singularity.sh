#!/bin/bash

set -euo pipefail

# Build the Singularity image from the definition file
singularity build docker/groundgan.sif docker/singularity.def
