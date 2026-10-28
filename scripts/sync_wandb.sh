#!/usr/bin/env bash
# Run on a login node with internet access after training finishes.
set -euo pipefail
TASK_SYNC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_SYNC_PYTHON="${LIGHTNING_OPD_PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
exec "$TASK_SYNC_PYTHON" "$TASK_SYNC_ROOT/scripts/sync_wandb.py" "$@"
