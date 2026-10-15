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

# From the repo root, after setting python_executable in YAML:
#   mkdir -p tmp
#   sbatch configs/sbatch/qwen3_4b_lightning_opd.sbatch
# Allocation: 2 nodes x 4 H100s (8 total), 8 hours for the entire pipeline.
# One allocation runs the complete pipeline, in order:
#   1. Prepare/reuse DAPO and eval parquet on one node.
#   2. Check models, data, training dependencies and allocation on every node.
#   3. Build the fixed 38,400-row DAPO prompt sequence and K-block plan on one node.
#   4. Convert/load the student, then repeat K times without resetting the trainer:
#      current-policy rollout -> offload -> teacher logprobs -> stop teacher -> train block.
#      Evaluate before training and every global 25 steps, independently of K.
# The 8-hour wall time covers all stages; override it with sbatch --time=... .
# Optional single-stage modes: --prepare-prompts, --prepare-data, --preflight,
# --check, --dry-run. For GPU preparation alone, sbatch --nodes=1 ... --prepare-data.
# Student initialization uses model_path in YAML.
# Common flags: --resume, --env-file /path.
# Both methods save only at the final step or 20 minutes before walltime.
# --resume also recovers an evaluation interrupted after checkpoint saving.
# A new experiment_name is required when switching student/teacher models.
# Results: <output_dir>/<experiment_name>/{checkpoints,data,eval_metrics,wandb,config.yaml}.
# The default validation sets follow opd-baselines; add the paper's own benchmarks
# separately for a direct comparison with the paper tables.

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
TASK_REPO_ROOT=/scratch/l/luli/src/Lightning-OPD
cd "$TASK_REPO_ROOT"
# Keep Python bytecode on each node's local disk rather than the scratch quota.
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${SLURM_TMPDIR:-/tmp}/lightning-opd-pycache-${SLURM_JOB_ID}}"
TASK_CONFIG=configs/config/qwen3_4b_lightning_opd.yaml
if [[ $# -gt 0 && "$1" != --* ]]; then TASK_CONFIG="$1"; shift; fi
# Resolve the YAML interpreter using the baseline environment's PyYAML.
# LIGHTNING_OPD_PYTHON is an optional submission-time interpreter override.
TASK_PYTHON="${LIGHTNING_OPD_PYTHON:-$(/scratch/l/luli/conda/envs/verl/bin/python -c \
    'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["python_executable"])' "$TASK_CONFIG")}"
if [[ ! -x "$TASK_PYTHON" ]]; then
    echo "Prepare the README training environment, then set python_executable in $TASK_CONFIG or LIGHTNING_OPD_PYTHON." >&2
    exit 1
fi
TASK_METHOD="$($TASK_PYTHON -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["method"])' "$TASK_CONFIG")"
TASK_MARGIN="$($TASK_PYTHON -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1])).get("walltime_checkpoint_margin_seconds", 1200))' "$TASK_CONFIG")"
TASK_END_TIME=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n' | sed -n 's/^EndTime=//p')
export OPD_CHECKPOINT_DEADLINE="$(( $(date -d "$TASK_END_TIME" +%s) - TASK_MARGIN ))"
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
        --prepare-data|--prepare-prompts|--check|--dry-run)
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

echo "[1/4] Prepare DAPO-Math-17k and evaluation parquet"
run_stage 1 --prepare-prompts "$@"
echo "[2/4] Validate training environment and allocation on all nodes"
run_stage "$TASK_COUNT" --preflight "$@"
if [[ "$TASK_METHOD" == lightning_opd ]]; then
    echo "[3/4] Prepare Lightning data (fixed prompt sequence for K-block training)"
    run_stage 1 --prepare-data "$@"
else
    echo "[3/4] Standard OPD uses a live teacher during training"
fi
echo "[4/4] Train and evaluate on all nodes"
run_stage "$TASK_COUNT" "$@"
echo "End-to-end job completed successfully."
