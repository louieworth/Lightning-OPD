#!/usr/bin/env bash
# Use the existing verl interpreter. The temporary venv lets pip resolve only
# missing/incompatible SFT packages against verl, without editing verl itself.
set -euo pipefail
TASK_REPO_ROOT=/scratch/l/luli/src/Lightning-OPD
TASK_BASE_PYTHON=/scratch/l/luli/conda/envs/verl/bin/python
TASK_RUNTIME="$TASK_REPO_ROOT/.runtime/sft"
TASK_BUILD_ENV="/tmp/lightning-sft-bootstrap-${UID}"
TASK_ARCHIVE="$TASK_RUNTIME/packages.tar.gz"
if [[ -f "$TASK_ARCHIVE" ]]; then
    echo "SFT packages already prepared: $TASK_ARCHIVE"
    exit 0
fi
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null DS_BUILD_OPS=0
export PIP_CACHE_DIR=/tmp/lightning-sft-pip-cache
"$TASK_BASE_PYTHON" -S -B -m venv --system-site-packages --without-pip "$TASK_BUILD_ENV"
TASK_SITE="$TASK_BUILD_ENV/lib/python3.12/site-packages"
# verl has a tokenizer .pth hook that eagerly imports torch on every Python
# start. Resolve packages without that hook, while retaining the temporary
# prefix so pip cannot uninstall anything from the shared verl environment.
PYTHONPATH="$TASK_SITE:/scratch/l/luli/conda/envs/verl/lib/python3.12/site-packages" \
    "$TASK_BUILD_ENV/bin/python" -S -B - "$TASK_BUILD_ENV" \
    "$TASK_REPO_ROOT/configs/environment/sft_requirements.txt" <<'PY'
import sys
prefix, requirements = sys.argv[1:]
sys.prefix = sys.exec_prefix = prefix
from pip._internal.cli.main import main
raise SystemExit(main(["install", "--no-build-isolation", "--no-compile", "-r", requirements]))
PY
mkdir -p "$TASK_RUNTIME"
tar -czf "$TASK_ARCHIVE.tmp" -C "$TASK_SITE" .
mv "$TASK_ARCHIVE.tmp" "$TASK_ARCHIVE"
echo "Prepared. Training uses $TASK_BASE_PYTHON and extracts $TASK_ARCHIVE to node-local /tmp."
