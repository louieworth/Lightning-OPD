#!/usr/bin/env bash
#SBATCH --job-name=qwen3-container
#SBATCH --account=aip-plbacon
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --time=08:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=tmp/slurm-%x-%j.log

set -euo pipefail
TASK_REPO="${LIGHTNING_OPD_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
cd "$TASK_REPO"
exec "${LIGHTNING_OPD_HOST_PYTHON:-/usr/bin/python3}" -I -S -B \
    containers/launch.py batch "$@"
