#!/usr/bin/env bash
# Optional archive overlay. The default SFT workflow uses installed packages
# directly and does not need this script or an archive.
set -euo pipefail
TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_BASE_PYTHON="${LIGHTNING_OPD_PYTHON:-python}"
TASK_ARCHIVE="${SFT_DEPENDENCY_ARCHIVE:-$TASK_REPO_ROOT/tmp/sft/packages.tar.gz}"
if [[ -f "$TASK_ARCHIVE" ]]; then
    echo "SFT packages already prepared: $TASK_ARCHIVE"
    exit 0
fi
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null DS_BUILD_OPS=0
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/lightning-sft-pip-cache}"
TASK_BUILD_ENV="$(mktemp -d "${TMPDIR:-/tmp}/lightning-sft-bootstrap.XXXXXX")"
trap 'rm -rf -- "$TASK_BUILD_ENV"' EXIT
"$TASK_BASE_PYTHON" -B -m venv --system-site-packages "$TASK_BUILD_ENV"
TASK_SITE="$("$TASK_BUILD_ENV/bin/python" -B -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
"$TASK_BUILD_ENV/bin/python" -B -m pip install --no-build-isolation --no-compile \
    -r "$TASK_REPO_ROOT/configs/environment/sft_requirements.txt"
mkdir -p "$(dirname "$TASK_ARCHIVE")"
tar -czf "$TASK_ARCHIVE.tmp" -C "$TASK_SITE" .
mv "$TASK_ARCHIVE.tmp" "$TASK_ARCHIVE"
echo "Prepared. Training uses $TASK_BASE_PYTHON and extracts $TASK_ARCHIVE to node-local /tmp."
