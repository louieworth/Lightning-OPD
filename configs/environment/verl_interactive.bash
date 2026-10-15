# Source this file inside the 30-minute salloc shell. A failing check returns
# to the interactive prompt; it does not release the allocation.
set +e
set -o pipefail
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/lightning-opd-pycache-${SLURM_JOB_ID:?Run inside salloc}"
export PYTHONPATH=/scratch/l/luli/src/Lightning-OPD:/scratch/l/luli/src/Lightning-OPD/.runtime/verl/src/Megatron-LM:/scratch/l/luli/src/opd-baselines
export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.6.2
export PATH="/scratch/l/luli/conda/envs/verl/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/scratch/l/luli/conda/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd /scratch/l/luli/src/Lightning-OPD || return

check_verl() {
    srun --mpi=none --nodes=1 --ntasks=1 --cpus-per-task=48 \
        --cpu-bind=none --gpu-bind=none --unbuffered \
        /usr/bin/env LD_PRELOAD=/scratch/l/luli/conda/envs/verl/lib/python3.12/site-packages/torch_memory_saver_hook_mode_preload.abi3.so \
        /scratch/l/luli/conda/envs/verl/bin/python configs/environment/check_verl_runtime.py "$@" \
        2>&1 | /usr/bin/tee -a ".runtime/verl/interactive-${SLURM_JOB_ID}.log"
    local task_status=${PIPESTATUS[0]}
    printf 'Check exit status: %s. Allocation remains available until Slurm time limit.\n' "$task_status"
    return "$task_status"
}
printf 'Allocation %s: run check_verl; failures return to this shell.\n' "$SLURM_JOB_ID"
