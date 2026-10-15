#!/usr/bin/env bash
# Keep an interactive allocation for 30 minutes; failed commands do not exit it.
set -euo pipefail
cd /scratch/l/luli/src/Lightning-OPD
exec salloc --job-name=lightning-verl-interactive --account=aip-plbacon \
    --time=00:30:00 --nodes=1 \
    --ntasks-per-node=1 --gpus-per-node=h100:4 --cpus-per-task=48 \
    --mem=0 --exclusive \
    /bin/bash --noprofile --rcfile configs/environment/verl_interactive.bash -i
