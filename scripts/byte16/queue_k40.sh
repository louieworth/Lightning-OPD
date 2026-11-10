#!/usr/bin/env bash
# Run the four K=40 byte16 presets one after another on the 16-GPU Ray cluster (resident + streamed blocks).
#   nohup bash tmp/runs/queue_k40.sh > tmp/runs/queue_k40.log 2>&1 &
# Each run: --prepare-data, then training with its own DCGM monitor (tmp/gpu_monitor/<name>).
# A failed run is logged and the queue moves on to the next preset.
cd /opt/tiger/Lightning-OPD
source ~/.bashrc >/dev/null 2>&1 || true
export MEGATRON_PATH=/opt/tiger/Megatron-LM
export SSL_CERT_FILE=$HOME/.claude/certs/bytedance-ca-bundle.pem REQUESTS_CA_BUNDLE=$HOME/.claude/certs/bytedance-ca-bundle.pem
LAUNCHER=configs/lightning_opd/qwen3-4b-lightning-opd.py
RUNS=(
  "a_k40 configs/config/byte16_qwen3_1_7b_nt_teacher_4b_instruct_2507_k40.yaml"
  "b_k40 configs/config/byte16_qwen3_1_7b_nt_teacher_4b_k40.yaml"
  "c_k40 configs/config/byte16_r1_distill_qwen_1_5b_teacher_justrl_1_5b_k40.yaml"
  "d_k40 configs/config/byte16_r1_distill_qwen_7b_teacher_light_r1_14b_k40.yaml"
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
  read -r NAME CONFIG <<<"$entry"
  for i in $(seq 1 60); do [ "$(gpus_free)" = yes ] && break; echo "$(date +%T) waiting for free GPUs before $NAME"; sleep 10; done
  echo "$(date '+%F %T') START $NAME $CONFIG"
  python3 $LAUNCHER --config "$CONFIG" --prepare-data > "tmp/runs/$NAME.prepare.log" 2>&1 || { echo "$(date '+%F %T') PREPARE_FAILED $NAME"; continue; }
  python3 tools/gpu_monitor.py --out "tmp/gpu_monitor/$NAME" --log "tmp/runs/$NAME.log" --interval 10 \
      > "tmp/runs/$NAME.monitor.log" 2>&1 &
  MONITOR=$!
  python3 $LAUNCHER --config "$CONFIG" > "tmp/runs/$NAME.log" 2>&1
  CODE=$?
  kill $MONITOR 2>/dev/null
  echo "$(date '+%F %T') END $NAME exit=$CODE"
  sleep 60  # let Ray release the job's actors and GPUs
done
echo "$(date '+%F %T') QUEUE_DONE"
