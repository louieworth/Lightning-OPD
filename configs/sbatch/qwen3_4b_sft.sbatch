#!/usr/bin/env bash
#SBATCH --job-name=qwen3-4b-sft
#SBATCH --account=aip-plbacon
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --time=08:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --chdir=/scratch/l/luli/src/Lightning-OPD
#SBATCH --output=/scratch/l/luli/src/Lightning-OPD/tmp/slurm-%x-%j.log

# On a fresh checkout, create tmp/ before submitting: mkdir -p tmp
# sbatch configs/sbatch/qwen3_4b_sft.sbatch [configs/config/qwen3_4b_sft.yaml]
# OpenThoughts3 Parquet prompts -> Qwen3-8B responses -> full Qwen3-4B SFT.
# Existing completed teacher batches and training checkpoints are resumed.
# If the full recipe exceeds eight hours, resubmit this same script.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
TASK_REPO_ROOT=/scratch/l/luli/src/Lightning-OPD
cd "$TASK_REPO_ROOT"
TASK_CONFIG="${1:-configs/config/qwen3_4b_sft.yaml}"
TASK_BASE_PYTHON=/scratch/l/luli/conda/envs/verl/bin/python
# Disable the cluster's inherited Python sitecustomize; use this prepared stack.
unset PYTHONPATH
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 SLURM_EXPORT_ENV=ALL
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/lightning-sft-pycache-${SLURM_JOB_ID}"
readarray -t TASK_SETTINGS < <("$TASK_BASE_PYTHON" -B -c \
    'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print("\n".join(str(c[k]) for k in ["python_executable","generation_python","cuda_home","nnodes","n_gpus_per_node","master_port","walltime_checkpoint_margin_seconds","sft_dependency_archive"]))' "$TASK_CONFIG")
TASK_SFT_PYTHON="${TASK_SETTINGS[0]}"
TASK_GENERATION_PYTHON="${TASK_SETTINGS[1]}"
export CUDA_HOME="${TASK_SETTINGS[2]}"
TASK_NODES="${TASK_SETTINGS[3]}"
TASK_GPUS="${TASK_SETTINGS[4]}"
export MASTER_PORT="${TASK_SETTINGS[5]}"
TASK_MARGIN="${TASK_SETTINGS[6]}"
TASK_SFT_ARCHIVE="${TASK_SETTINGS[7]}"
TASK_SFT_DEPENDENCIES="/tmp/lightning-sft-deps-${SLURM_JOB_ID}-${USER}"
[[ -f "$TASK_SFT_ARCHIVE" ]] || { echo "Missing SFT dependency archive: $TASK_SFT_ARCHIVE" >&2; exit 1; }
[[ "$SLURM_JOB_NUM_NODES" == "$TASK_NODES" ]] || { echo "Allocation/YAML node count mismatch" >&2; exit 1; }
export PATH="$(dirname "$TASK_SFT_PYTHON"):$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/scratch/l/luli/conda/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME=/scratch/l/luli/hf
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR:-/tmp}/lightning-sft-extensions-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/lightning-sft-triton-${SLURM_JOB_ID}"
export VLLM_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/lightning-sft-vllm-${SLURM_JOB_ID}"
TASK_END_TIME=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n' | sed -n 's/^EndTime=//p')
export SFT_CHECKPOINT_DEADLINE="$(( $(date -d "$TASK_END_TIME" +%s) - TASK_MARGIN ))"
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

run_stage() {
    srun --mpi=none --nodes="$1" --ntasks="$1" --ntasks-per-node=1 \
        --cpu-bind=none --gpu-bind=none --kill-on-bad-exit=1 --unbuffered "${@:2}"
}
echo "[1/5] Prepare/reuse 300K OpenThoughts3 Parquet prompts"
run_stage 1 "$TASK_GENERATION_PYTHON" -B scripts/prepare_sft_parquet.py --config "$TASK_CONFIG"
echo "[2/5] Record immutable recipe and dataset provenance"
run_stage 1 "$TASK_GENERATION_PYTHON" -B scripts/qwen3_sft.py --config "$TASK_CONFIG" --stage configure
echo "[3/5] Generate/resume SAME Qwen3-8B teacher responses on all eight GPUs"
set +e
run_stage "$TASK_NODES" "$TASK_GENERATION_PYTHON" -B scripts/qwen3_sft.py --config "$TASK_CONFIG" --stage generate-node
TASK_STATUS=$?
set -e
[[ "$TASK_STATUS" == 0 ]] || exit "$TASK_STATUS"
if (( $(date +%s) >= SFT_CHECKPOINT_DEADLINE )); then
    run_stage 1 "$TASK_GENERATION_PYTHON" -B scripts/qwen3_sft.py --config "$TASK_CONFIG" --stage pause-generation
    echo "Teacher generation paused before wall time. Resubmit this sbatch to resume saved Parquet batches."
    exit 0
fi
echo "[4/5] Merge and verify all 300K teacher responses as Parquet"
run_stage 1 "$TASK_GENERATION_PYTHON" -B scripts/qwen3_sft.py --config "$TASK_CONFIG" --stage merge
echo "[5/5] LlamaFactory SFT: 3000 steps, global batch 256, ZeRO-0"
# Static rendezvous: all nodes share MASTER_ADDR/PORT; Slurm supplies node rank.
run_stage "$TASK_NODES" /bin/bash -c '
    mkdir -p "$5"
    tar -xzf "$6" -C "$5"
    export PYTHONPATH="$5:$PWD"
    exec "$1" -B -m torch.distributed.run --nnodes="$3" --nproc-per-node="$4" \
        --node-rank="$SLURM_PROCID" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
        scripts/qwen3_sft.py --config "$2" --stage train
' _ "$TASK_SFT_PYTHON" "$TASK_CONFIG" "$TASK_NODES" "$TASK_GPUS" "$TASK_SFT_DEPENDENCIES" "$TASK_SFT_ARCHIVE"
run_stage 1 "$TASK_GENERATION_PYTHON" -B scripts/qwen3_sft.py --config "$TASK_CONFIG" --stage finish
