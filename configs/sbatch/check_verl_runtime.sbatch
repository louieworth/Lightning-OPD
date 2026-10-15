#!/usr/bin/env bash
#SBATCH --job-name=lightning-verl-check
#SBATCH --account=aip-plbacon
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --chdir=/scratch/l/luli/src/Lightning-OPD
#SBATCH --output=/scratch/l/luli/src/Lightning-OPD/tmp/slurm-%x-%j.log

# On a fresh checkout, create tmp/ before submitting: mkdir -p tmp
# Short GPU environment validation; production training uses its own 2-node sbatch.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/lightning-opd-pycache-${SLURM_JOB_ID}"
export PYTHONPATH=/scratch/l/luli/src/Lightning-OPD:/scratch/l/luli/src/Lightning-OPD/.runtime/verl/src/Megatron-LM:/scratch/l/luli/src/opd-baselines
export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.6.2
export PATH="/scratch/l/luli/conda/envs/verl/bin:$CUDA_HOME/bin:$PATH"
# Match the preload used by slime/ray/actor_group.py for training workers.
# Apply it only to Python, so Slurm itself does not load CUDA hooks.
srun --mpi=none --ntasks=1 --cpu-bind=none --gpu-bind=none --kill-on-bad-exit=1 --unbuffered \
    /usr/bin/env LD_PRELOAD="/scratch/l/luli/conda/envs/verl/lib/python3.12/site-packages/torch_memory_saver_hook_mode_preload.abi3.so${LD_PRELOAD:+:$LD_PRELOAD}" \
    LD_LIBRARY_PATH="/scratch/l/luli/conda/envs/verl/lib/python3.12/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    /scratch/l/luli/conda/envs/verl/bin/python configs/environment/check_verl_runtime.py "$@"
