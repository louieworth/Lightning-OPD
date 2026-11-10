#!/usr/bin/env bash
# Launch one byte16 preset on the pre-started Ray cluster, with the DCGM GPU monitor.
#   bash tmp/runs/launch.sh <preset yaml> <run name>
set -euo pipefail
cd /opt/tiger/Lightning-OPD
CONFIG=$1 NAME=$2
source ~/.bashrc >/dev/null 2>&1 || true
export MEGATRON_PATH=/opt/tiger/Megatron-LM
export SSL_CERT_FILE=$HOME/.claude/certs/bytedance-ca-bundle.pem REQUESTS_CA_BUNDLE=$HOME/.claude/certs/bytedance-ca-bundle.pem
python3 configs/lightning_opd/qwen3-4b-lightning-opd.py --config "$CONFIG" --prepare-data > "tmp/runs/$NAME.prepare.log" 2>&1
nohup python3 configs/lightning_opd/qwen3-4b-lightning-opd.py --config "$CONFIG" > "tmp/runs/$NAME.log" 2>&1 &
echo $! > "tmp/runs/$NAME.pid"
nohup python3 tools/gpu_monitor.py --out "tmp/gpu_monitor/$NAME" --log "tmp/runs/$NAME.log" --interval 10 \
    > "tmp/runs/$NAME.monitor.log" 2>&1 &
echo $! > "tmp/runs/$NAME.monitor.pid"
echo "launched $NAME pid=$(cat tmp/runs/$NAME.pid)"
