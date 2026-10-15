#!/usr/bin/env bash
#SBATCH --job-name=qwen3-4b-sft-h200-resume
#SBATCH --account=aip-plbacon
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --chdir=/scratch/l/luli/src/Lightning-OPD
#SBATCH --output=/scratch/l/luli/src/Lightning-OPD/tmp/slurm-%x-%j.log

# Same eight training ranks and batch 256 as the two-node H100 allocation.
# Reuse the complete optimizer/scheduler/RNG checkpoint in the same experiment.
# sbatch configs/sbatch/qwen3_4b_sft_downloaded_h200.sbatch
set -euo pipefail
TASK_SFT_H200_ROOT=/scratch/l/luli/src/Lightning-OPD
TASK_SFT_H200_CONFIG="${1:-configs/config/qwen3_4b_sft_downloaded_h200.yaml}"
cd "$TASK_SFT_H200_ROOT"
exec bash configs/sbatch/qwen3_4b_sft_downloaded.sbatch "$TASK_SFT_H200_CONFIG"
