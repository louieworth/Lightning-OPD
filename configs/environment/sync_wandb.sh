#!/usr/bin/env bash
# Run on a login node with internet access after the training job finishes.
set -euo pipefail
TASK_SYNC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TASK_SYNC_ROOT"
TASK_SYNC_PYTHON="${LIGHTNING_OPD_PYTHON:-python}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/lightning-opd-wandb-sync-${UID}}"
exec "$TASK_SYNC_PYTHON" configs/environment/sync_wandb.py "$@"
