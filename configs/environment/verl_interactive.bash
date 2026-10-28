# Source this file inside the 30-minute salloc shell. A failing check returns
# to the interactive prompt; it does not release the allocation.
set +e
set -o pipefail
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/lightning-opd-pycache-${SLURM_JOB_ID:?Run inside salloc}"
TASK_CHECK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TASK_CHECK_ROOT" || return
mkdir -p tmp/runtime-checks

check_verl() {
    srun --mpi=none --nodes=1 --ntasks=1 --cpus-per-task=48 \
        --cpu-bind=none --gpu-bind=none --unbuffered \
        bash configs/environment/run_runtime_check.sh "$@" \
        2>&1 | /usr/bin/tee -a "tmp/runtime-checks/interactive-${SLURM_JOB_ID}.log"
    local task_status=${PIPESTATUS[0]}
    printf 'Check exit status: %s. Allocation remains available until Slurm time limit.\n' "$task_status"
    return "$task_status"
}
printf 'Allocation %s: run check_verl; failures return to this shell.\n' "$SLURM_JOB_ID"
