#!/usr/bin/env bash
#SBATCH --job-name=qwen3-opd
#SBATCH --account=aip-plbacon
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --time=10:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --chdir=/scratch/l/luli/src/Lightning-OPD
#SBATCH --output=/scratch/l/luli/src/Lightning-OPD/tmp/slurm-%x-%j.log

# From the repo root, after setting python_executable in YAML:
#   mkdir -p tmp
#   sbatch configs/sbatch/qwen3_4b_opd.sbatch
# One job prepares input parquet, checks all nodes, then trains and evaluates.
# Standard OPD computes teacher logprobs online; it skips offline precomputation.
# Student initialization uses model_path in YAML.
# Optional: --resume, --check, --dry-run.
# Evaluate every 25 steps; save at the final step or 20 minutes before walltime.
# Keep only the latest full training checkpoint after its successful evaluation.
# The teacher runs on the last two GPUs of the node with the greatest IP address;
# Ray receives the other six GPUs: 2 actor + 4 rollout GPUs.

set -euo pipefail
TASK_CONFIG=configs/config/qwen3_4b_opd.yaml
if [[ $# -gt 0 && "$1" != --* ]]; then TASK_CONFIG="$1"; shift; fi
exec bash /scratch/l/luli/src/Lightning-OPD/configs/sbatch/qwen3_4b_lightning_opd.sbatch "$TASK_CONFIG" "$@"
