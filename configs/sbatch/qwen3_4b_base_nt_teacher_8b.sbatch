#!/usr/bin/env bash
#SBATCH --job-name=qwen3-lightning-opd
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

# From the repo root, with a prepared Python environment active:
#   mkdir -p tmp
#   sbatch configs/sbatch/qwen3_4b_base_nt_teacher_8b.sbatch
# Allocation: 2 nodes x 4 H100s (8 total), 8 hours for the entire pipeline.
# One allocation runs the complete pipeline, in order:
#   1. Read train_files/val_files from YAML and check models, data, dependencies
#      and allocation on every node. Input parquet must already be prepared.
#   2. Build the fixed prompt sequence and K-block plan on one node.
#   3. Convert/load the student, then repeat K times without resetting the trainer:
#      current-policy rollout -> offload -> teacher logprobs -> stop teacher -> train block.
#      Evaluate before training and every global 25 steps, independently of K.
# The 8-hour wall time covers all stages; override it with sbatch --time=... .
# Optional single-stage modes: --convert-datasets, --prepare-data, --preflight,
# --check, --dry-run. For GPU preparation alone, sbatch --nodes=1 ... --prepare-data.
# Dataset format conversion is explicit; --prepare-prompts is its legacy alias.
# Student initialization uses model_path in YAML.
# Common flags: --resume, --env-file /path.
# Both methods save at evaluation steps and retain the latest evaluated checkpoint.
# --resume also recovers an evaluation interrupted after checkpoint saving.
# A new experiment_name is required when switching student/teacher models.
# Results: <output_dir>/<experiment_name>/{checkpoints,data,eval_metrics,config.yaml}.
# W&B logs: <repo>/wandb/<experiment_name>/ (or the YAML wandb_dir override).
# The default validation sets follow opd-baselines; add the paper's own benchmarks
# separately for a direct comparison with the paper tables.

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
TASK_REPO_ROOT=/scratch/l/luli/src/Lightning-OPD
cd "$TASK_REPO_ROOT"
# Keep Python bytecode on each node's local disk rather than the scratch quota.
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${SLURM_TMPDIR:-/tmp}/lightning-opd-pycache-${SLURM_JOB_ID}}"
TASK_CONFIG=configs/config/qwen3_4b_base_nt_teacher_8b.yaml
if [[ $# -gt 0 && "$1" != --* ]]; then TASK_CONFIG="$1"; shift; fi
# Route to the container before importing anything from host Python.
if [[ -n "${LIGHTNING_OPD_CONTAINER:-}" ]]; then
    exec /usr/bin/python3 -I -S -B containers/launch.py batch \
        --config "$TASK_CONFIG" --image "$LIGHTNING_OPD_CONTAINER" \
        --runtime "${LIGHTNING_OPD_CONTAINER_RUNTIME:-auto}" -- "$@"
fi
# LIGHTNING_OPD_PYTHON overrides YAML. Otherwise bootstrap with the active
# Python (with PyYAML installed), and default to that same interpreter.
TASK_PYTHON="${LIGHTNING_OPD_PYTHON:-$(python -B -c \
    'import sys,yaml; print(yaml.safe_load(open(sys.argv[1])).get("python_executable") or sys.executable)' "$TASK_CONFIG")}"
if ! TASK_PYTHON="$(command -v -- "$TASK_PYTHON")"; then
    echo "Python not found. Activate the training environment or set LIGHTNING_OPD_PYTHON." >&2
    exit 1
fi
TASK_METHOD="$("$TASK_PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["method"])' "$TASK_CONFIG")"
case "$TASK_METHOD" in
    lightning_opd) TASK_ENTRY=configs/lightning_opd/qwen3-4b-lightning-opd.py ;;
    opd) TASK_ENTRY=configs/opd/qwen3-4b-opd.py ;;
    *) echo "Unsupported method: $TASK_METHOD" >&2; exit 1 ;;
esac
export PYTHONUNBUFFERED=1
export SLURM_EXPORT_ENV=ALL
export PATH="$(dirname "$TASK_PYTHON"):$PATH"
export PYTHONPATH="$TASK_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
TASK_COUNT="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
TASK_SINGLE_STAGE=false
# Explicit modes keep their original meaning and do not run the full pipeline.
for TASK_ARG in "$@"; do
    case "$TASK_ARG" in
        --prepare-data|--convert-datasets|--prepare-prompts|--check|--dry-run)
            TASK_SINGLE_STAGE=true
            TASK_COUNT=1
            ;;
        --preflight) TASK_SINGLE_STAGE=true ;;
    esac
done

run_stage() {
    local task_nodes="$1"
    shift
    srun --mpi=none --nodes="$task_nodes" --ntasks="$task_nodes" --ntasks-per-node=1 \
        --cpu-bind=none --gpu-bind=none --kill-on-bad-exit=1 --unbuffered \
        "$TASK_PYTHON" "$TASK_ENTRY" --config "$TASK_CONFIG" "$@"
}

if [[ "$TASK_SINGLE_STAGE" == true ]]; then
    run_stage "$TASK_COUNT" "$@"
    exit 0
fi

echo "[1/3] Validate configured train/eval parquet, training environment and allocation on all nodes"
run_stage "$TASK_COUNT" --preflight "$@"
if [[ "$TASK_METHOD" == lightning_opd ]]; then
    echo "[2/3] Build the fixed prompt sequence from train_files for K-block training"
    run_stage 1 --prepare-data "$@"
else
    echo "[2/3] Standard OPD uses a live teacher during training"
fi
echo "[3/3] Train and evaluate on all nodes"
run_stage "$TASK_COUNT" "$@"
echo "End-to-end job completed successfully."
