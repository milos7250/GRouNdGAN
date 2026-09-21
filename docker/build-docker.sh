#!/bin/bash

set -euo pipefail

# Build the Docker image from the definition file
docker build --target runtime -t groundgan -f docker/Dockerfile .
