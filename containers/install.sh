#!/usr/bin/env bash
# Build-time installation only; runtime never invokes pip or host environments.
set -euo pipefail
umask 022
cd /opt/Lightning-OPD
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0
python - <<'PY'
from importlib import metadata
from pathlib import Path
native = ['torch', 'triton', 'transformer-engine', 'transformer-engine-torch',
          'flash-attn', 'sglang', 'sgl-kernel', 'torch-memory-saver', 'sglang-router']
constraints = []
for name in native:
    try:
        constraints.append(f'{name}=={metadata.version(name)}')
    except metadata.PackageNotFoundError:
        if name in ('torch', 'transformer-engine', 'flash-attn', 'sglang'):
            raise
Path('/opt/lightning-native-constraints.txt').write_text('\n'.join(constraints) + '\n')
PY
# Inherit the immutable CUDA stack and install Python additions separately.
# This also avoids pip traversing the entire base filesystem to uninstall it.
python -m venv --without-pip --system-site-packages /opt/opd-venv
export PATH="/opt/opd-venv/bin:$PATH"
# TE locates NVIDIA wheels relative to this interpreter's purelib. Expose
# the inherited wheels there too, without copying or reinstalling CUDA.
python - <<'PY'
from pathlib import Path
import sys
import sysconfig
target = Path(sysconfig.get_path('purelib')) / 'nvidia'
if not target.exists():
    native = next(Path(path) / 'nvidia' for path in sys.path
                  if (Path(path) / 'nvidia/cudnn/lib').is_dir())
    target.symlink_to(native, target_is_directory=True)
PY
# Megatron's editable build explicitly invokes python3 -m pybind11.
python -m pip install 'setuptools==79.0.1' 'pybind11==2.13.6' wheel
# Install full training/core source in a readable path. Test datasets and
# repository history are unnecessary at runtime and costly on a FUSE overlay.
if [[ ! -d /opt/Megatron-LM ]]; then
    test -d /root/Megatron-LM/megatron/training
    mkdir -p /opt/Megatron-LM
    tar -C /root/Megatron-LM --exclude=.git --exclude=tests --exclude=docs --exclude=examples -cf - . \
        | tar -C /opt/Megatron-LM -xf -
fi
chmod -R a+rX /opt/Megatron-LM /opt/Lightning-OPD
python -m pip install --no-build-isolation --no-deps -e /opt/Megatron-LM
python -m pip install -c /opt/lightning-native-constraints.txt \
    -r /opt/lightning-native-constraints.txt -r containers/requirements-opd.txt
python -m pip install --no-build-isolation --no-deps -e /opt/Lightning-OPD
# Backport correct freeing of TMS-owned pointers outside an active region.
python containers/patch_memory_saver.py
# This runs on CPU builders too and catches CUDA wheel lookup/ABI failures.
python -c 'import transformer_engine.pytorch; import mbridge'
python -m pip freeze --all > /opt/lightning-pip-freeze.txt
python - <<'PY'
from importlib import metadata
from pathlib import Path
import json
names = ['torch', 'sglang', 'sglang-router', 'sgl-kernel', 'torch-memory-saver',
         'transformer-engine', 'flash-attn', 'megatron-core', 'mbridge', 'transformers',
         'ray', 'pyarrow', 'math-verify']
source = json.loads(Path('/opt/Lightning-OPD/containers/source.json').read_text())
manifest = {'schema': 1, **source, 'versions': {name: metadata.version(name) for name in names},
            'tms_native_patch': json.loads(Path('/opt/lightning-tms-build.json').read_text())}
Path('/opt/lightning-container.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
