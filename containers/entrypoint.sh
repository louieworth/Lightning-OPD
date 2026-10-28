#!/usr/bin/env bash
set -euo pipefail
TASK_REPO="${LIGHTNING_OPD_REPO:-/opt/Lightning-OPD}"
test -f "$TASK_REPO/train.py" || { echo "Repository mount is missing: $TASK_REPO" >&2; exit 1; }
export MEGATRON_PATH=/opt/Megatron-LM
export PYTHONPATH="$TASK_REPO:$MEGATRON_PATH"
export CUDA_HOME=/usr/local/cuda
export PATH="/opt/opd-venv/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONHOME CONDA_PREFIX VIRTUAL_ENV LIGHTNING_OPD_PYTHON LD_PRELOAD
export HOME="${TMPDIR:-/tmp}/lightning-container-home-$(id -u)"
mkdir -p "$HOME"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/torch-extensions}"
TASK_CUDA_RUNTIME="$(python -c 'import sys; from pathlib import Path; print(next(str(Path(p) / "nvidia/cuda_runtime/lib") for p in sys.path if (Path(p) / "nvidia/cuda_runtime/lib").is_dir()))')"
if [[ -d "$TASK_CUDA_RUNTIME" ]]; then
    export LD_LIBRARY_PATH="$TASK_CUDA_RUNTIME${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
cd "$TASK_REPO"
if [[ $# == 0 ]]; then set -- bash; fi
exec "$@"
