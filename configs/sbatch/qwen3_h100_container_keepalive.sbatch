#!/usr/bin/env bash
#SBATCH --job-name=opd-container-debug
#SBATCH --account=aip-plbacon
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=tmp/slurm-%x-%j.log

# Independent srun steps may fail without releasing this allocation.
set -u
TASK_REPO="${LIGHTNING_OPD_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
cd "$TASK_REPO" || exit 1
trap 'exit 0' TERM INT
printf 'Debug allocation %s: %s; remains available until the one-hour limit or scancel.\n' \
    "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST"
scontrol show hostnames "$SLURM_JOB_NODELIST"
while :; do
    sleep 30 &
    wait "$!" || true
done
