#!/usr/bin/env bash
#SBATCH --job-name=qwen3-1.7b-lightning-opd
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

# Qwen3-1.7B (instruct) student and Qwen3-4B-Instruct-2507 teacher; 2 nodes x 4 H100s, 10 hours.
# Reuse the shared Lightning OPD pipeline with an independent experiment.
# Submit: sbatch configs/sbatch/qwen3_1_7b_nt_teacher_4b_instruct_2507_k40.sbatch
# Optional: alternate YAML path, --resume, --check or --dry-run.
set -euo pipefail
TASK_CONFIG=configs/config/qwen3_1_7b_nt_teacher_4b_instruct_2507_k40.yaml
if [[ $# -gt 0 && "$1" != --* ]]; then TASK_CONFIG="$1"; shift; fi
exec bash /scratch/l/luli/src/Lightning-OPD/configs/sbatch/qwen3_4b_base_nt_teacher_8b.sbatch "$TASK_CONFIG" "$@"
