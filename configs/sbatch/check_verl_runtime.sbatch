#!/usr/bin/env bash
#SBATCH --job-name=lightning-runtime-check
#SBATCH --account=aip-plbacon
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --chdir=/scratch/l/luli/src/Lightning-OPD
#SBATCH --output=/scratch/l/luli/src/Lightning-OPD/tmp/slurm-%x-%j.log

# On a fresh checkout, create tmp/ before submitting: mkdir -p tmp
# Short GPU environment validation; production training uses its own 2-node sbatch.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/lightning-opd-pycache-${SLURM_JOB_ID}"
# Match the worker preload using packages from the selected environment.
srun --mpi=none --ntasks=1 --cpu-bind=none --gpu-bind=none --kill-on-bad-exit=1 --unbuffered \
    bash configs/environment/run_runtime_check.sh "$@"
