#!/usr/bin/env bash
#SBATCH --job-name=qwen3-1.7b-lightning-opd-k2
#SBATCH --account=aip-plbacon
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --time=06:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --chdir=/scratch/l/luli/src/Lightning-OPD
#SBATCH --output=/scratch/l/luli/src/Lightning-OPD/tmp/slurm-%x-%j.log

# Qwen3-1.7B-Base student, Qwen3-8B teacher, K=2; 150 total steps.
# Allocation: 2 nodes x 4 H100s, 6 hours for the shared Lightning OPD pipeline.
# Submit: sbatch configs/sbatch/qwen3_1_7b_lightning_opd_k2.sbatch
# Optional: alternate YAML path, --resume, --check or --dry-run.
set -euo pipefail
TASK_CONFIG=configs/config/qwen3_1_7b_lightning_opd_k2.yaml
if [[ $# -gt 0 && "$1" != --* ]]; then TASK_CONFIG="$1"; shift; fi
exec bash /scratch/l/luli/src/Lightning-OPD/configs/sbatch/qwen3_4b_lightning_opd.sbatch "$TASK_CONFIG" "$@"
