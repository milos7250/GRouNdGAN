#!/bin/bash

set -euo pipefail

# Build the Apptainer image from the definition file
apptainer build docker/groundscale.sif docker/apptainer.def
