#!/bin/bash

echo "Running GRouNdGAN on $HOSTNAME"
apptainer exec --nv --env "OMP_NUM_THREADS=4" ~/apps/apptainer/groundgan.sif \
    python3.9 \
    -m torch.distributed.run \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    --nproc_per_node=2 \
    simulation/GRouNdGAN/src/main.py \
    --config simulation/causal_gan.cfg \
    --train~