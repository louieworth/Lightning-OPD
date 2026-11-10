#!/usr/bin/env bash
# Upload A k40's offline W&B logs once its training exits (queue logs "END a_k40").
cd /opt/tiger/Lightning-OPD
until grep -q "END a_k40" tmp/runs/queue_k40.log; do sleep 30; done
sleep 30
source ~/.bashrc >/dev/null 2>&1 || true
export SSL_CERT_FILE=$HOME/.claude/certs/bytedance-ca-bundle.pem REQUESTS_CA_BUNDLE=$HOME/.claude/certs/bytedance-ca-bundle.pem
echo "$(date '+%F %T') SYNC_START a_k40"
LIGHTNING_OPD_PYTHON=python3 bash scripts/sync_wandb.sh qwen3_1_7b_nt_teacher_4b_instruct_2507_k40_byte16
echo "$(date '+%F %T') SYNC_END exit=$?"
