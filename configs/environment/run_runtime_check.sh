#!/usr/bin/env bash
# Run the existing GPU check with the selected Python and its own CUDA hooks.
set -euo pipefail
TASK_CHECK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_CHECK_PYTHON="${LIGHTNING_OPD_PYTHON:-python}"
TASK_CHECK_SITE="$("$TASK_CHECK_PYTHON" -B -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export PYTHONPATH="$TASK_CHECK_ROOT${MEGATRON_PATH:+:$MEGATRON_PATH}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "$TASK_CHECK_SITE/torch_memory_saver_hook_mode_preload.abi3.so" ]]; then
    export LD_PRELOAD="$TASK_CHECK_SITE/torch_memory_saver_hook_mode_preload.abi3.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi
if [[ -d "$TASK_CHECK_SITE/nvidia/cuda_runtime/lib" ]]; then
    export LD_LIBRARY_PATH="$TASK_CHECK_SITE/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
cd "$TASK_CHECK_ROOT"
exec "$TASK_CHECK_PYTHON" -B configs/environment/check_verl_runtime.py "$@"
