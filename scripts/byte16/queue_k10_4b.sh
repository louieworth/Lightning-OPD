# 11/06: base students <- Qwen3-4B-Instruct-2507: 0.6B and 1.7B K=10, then 4B K=1/10/20/40/120 (_v2 settings).
#!/usr/bin/env bash
# Run Qwen3-4B-Base <- Qwen3-4B-Instruct-2507 at K=1, K=40 and K=120 one after another (16 GPUs).
#   nohup bash tmp/runs/queue_base4b_212.sh > tmp/runs/queue_base4b_212.log 2>&1 &
# K=1/K=40 use the Lightning launcher (with --prepare-data); K=120 is standard OPD.
# Each run: --prepare-data, then training with its own DCGM monitor (tmp/gpu_monitor/<name>).
# A failed run is logged and the queue moves on to the next preset.
cd /opt/tiger/Lightning-OPD
source ~/.bashrc >/dev/null 2>&1 || true
export MEGATRON_PATH=/opt/tiger/Megatron-LM
export SSL_CERT_FILE=$HOME/.claude/certs/bytedance-ca-bundle.pem REQUESTS_CA_BUNDLE=$HOME/.claude/certs/bytedance-ca-bundle.pem
LIGHTNING=configs/lightning_opd/qwen3-4b-lightning-opd.py
OPD=configs/opd/qwen3-4b-opd.py
RUNS=(
  "b06_k10 lightning configs/config/byte16_qwen3_0_6b_base_nt_teacher_4b_instruct_2507_k10.yaml"
  "b17_k10 lightning configs/config/byte16_qwen3_1_7b_base_nt_teacher_4b_instruct_2507_k10.yaml"
  "b4_k1 lightning configs/config/byte16_qwen3_4b_base_nt_teacher_4b_instruct_2507_k1.yaml"
  "b4_k10 lightning configs/config/byte16_qwen3_4b_base_nt_teacher_4b_instruct_2507_k10.yaml"
  "b4_k20 lightning configs/config/byte16_qwen3_4b_base_nt_teacher_4b_instruct_2507_k20.yaml"
  "b4_k40 lightning configs/config/byte16_qwen3_4b_base_nt_teacher_4b_instruct_2507_k40.yaml"
  "b4_k120 opd configs/config/byte16_qwen3_4b_base_nt_teacher_4b_instruct_2507_k120.yaml"
)

gpus_free() {  # every GPU of both nodes has no process holding more than 2 GB (the keep-alive holds ~0.7 GB)
  python3 - <<'PY' 2>/dev/null | tail -1
import ray
ray.init(address="auto", logging_level="ERROR", log_to_driver=False)
free = ray._private.state.available_resources_per_node()
print("yes" if all(v.get("GPU", 0) >= 8 for v in free.values() if "GPU" in v) else "no")
PY
}

for entry in "${RUNS[@]}"; do
  read -r NAME METHOD CONFIG <<<"$entry"
  if [ "$METHOD" = opd ]; then LAUNCHER=$OPD; else LAUNCHER=$LIGHTNING; fi
  for i in $(seq 1 60); do [ "$(gpus_free)" = yes ] && break; echo "$(date +%T) waiting for free GPUs before $NAME"; sleep 10; done
  echo "$(date '+%F %T') START $NAME $CONFIG"
  if [ "$METHOD" = lightning ]; then
    python3 $LAUNCHER --config "$CONFIG" --prepare-data > "tmp/runs/$NAME.prepare.log" 2>&1 || { echo "$(date '+%F %T') PREPARE_FAILED $NAME"; continue; }
  fi
  python3 tools/gpu_monitor.py --out "tmp/gpu_monitor/$NAME" --log "tmp/runs/$NAME.log" --interval 10 \
      > "tmp/runs/$NAME.monitor.log" 2>&1 &
  MONITOR=$!
  python3 $LAUNCHER --config "$CONFIG" >> "tmp/runs/$NAME.log" 2>&1
  CODE=$?
  kill $MONITOR 2>/dev/null
  echo "$(date '+%F %T') END $NAME exit=$CODE"
  sleep 60  # let Ray release the job's actors and GPUs
done
echo "$(date '+%F %T') QUEUE_DONE"
