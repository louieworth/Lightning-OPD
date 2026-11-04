#!/usr/bin/env bash
# Set up the Lightning-OPD runtime on a Byte Arnold H100 node (system Python 3.11, torch 2.8.0).
# Idempotent; run on every Ray node. Keeps the node's Ray (bytedray 2.10) and CUDA stack untouched.
set -euo pipefail
REPO=${REPO:-/opt/tiger/Lightning-OPD}
MEGATRON_DIR=${MEGATRON_DIR:-/opt/tiger/Megatron-LM}
MEGATRON_COMMIT=23e00ed0963c35382dfe8a5a94fb3cda4d21e133   # core_v0.14.0, see configs/environment/README.md

source ~/.bashrc >/dev/null 2>&1 || true
CERT=$HOME/.claude/certs/bytedance-ca-bundle.pem
if [[ -f $CERT ]]; then export SSL_CERT_FILE=$CERT CURL_CA_BUNDLE=$CERT GIT_SSL_CAINFO=$CERT REQUESTS_CA_BUNDLE=$CERT; fi
# The internal index must bypass the outbound proxy (it returns 502 through it).
export no_proxy="${no_proxy:-},bytedpypi.byted.org,.byted.org" NO_PROXY="${NO_PROXY:-},bytedpypi.byted.org,.byted.org"
export PIP_DISABLE_PIP_VERSION_CHECK=1 MAX_JOBS=16 TORCH_CUDA_ARCH_LIST=9.0

# Pin what the node already provides so pip never replaces it.
python3 - <<'PY' > /tmp/lightning-constraints.txt
from importlib import metadata
for n in ['torch', 'torchaudio', 'torchvision', 'triton', 'transformer-engine', 'transformer_engine_torch',
          'flash-attn', 'ray', 'transformers', 'apex', 'megatron-core']:
    try: print(f'{n}=={metadata.version(n)}')
    except metadata.PackageNotFoundError: pass
PY
cat /tmp/lightning-constraints.txt

python3 -m pip install -c /tmp/lightning-constraints.txt \
    'sglang==0.5.5.post1' 'sglang-router>=0.2.3,<0.3' 'torch_memory_saver==0.0.9' \
    'mbridge==0.13.1' 'math-verify==0.9.0' 'mathruler==0.1.0' 'liger-kernel==0.6.4' \
    'tensorstore==0.1.85' 'wandb==0.27.0' 'pylatexenc==2.10' 'numpy==1.26.4' \
    ring_flash_attn omegaconf blobfile einops sentencepiece pyyaml 'setuptools==79.0.1' 'pybind11==2.13.6' wheel
# sglang 0.5.5 pulls flashinfer 0.5.0; the image's flashinfer-jit-cache 0.4.1 then makes `import flashinfer` fail.
python3 -m pip install --no-deps 'flashinfer-jit-cache==0.5.0' --index-url https://flashinfer.ai/whl/cu128

# Full Megatron-LM (megatron.training) at the commit the repo was validated with, plus its no-MTP patch.
if [[ ! -d $MEGATRON_DIR/megatron/training ]]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git "$MEGATRON_DIR"
fi
cd "$MEGATRON_DIR"
git fetch -q origin "$MEGATRON_COMMIT" 2>/dev/null || true
git checkout -q "$MEGATRON_COMMIT"
git apply --check "$REPO/configs/environment/megatron-core-v0.14-no-mtp.patch" 2>/dev/null \
    && git apply "$REPO/configs/environment/megatron-core-v0.14-no-mtp.patch" || echo "no-mtp patch already applied"

cd "$REPO"
python3 -m pip install --no-build-isolation --no-deps -e "$REPO"
# Backport TorchMemorySaver's cudaFree fix (needs a writable /opt marker).
python3 containers/patch_memory_saver.py

PYTHONPATH=$MEGATRON_DIR python3 - <<'PY'
import torch, sglang, sgl_kernel, torch_memory_saver, mbridge, megatron.core, megatron.training, ray, transformers, slime
print('torch', torch.__version__, 'sglang', sglang.__version__, 'ray', ray.__version__, 'transformers', transformers.__version__)
print('megatron.training', megatron.training.__file__)
PY
echo INSTALL_OK
