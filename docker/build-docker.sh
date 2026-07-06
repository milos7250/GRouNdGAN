#!/bin/bash

set -euo pipefail

# Build the Docker image from the definition file
docker build -t groundgan -f docker/Dockerfile .
