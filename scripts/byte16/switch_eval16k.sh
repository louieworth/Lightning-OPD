#!/usr/bin/env bash
# 11/05: once 4B base K=40 has saved its step-100 checkpoint (its 32k step-100 eval is then starting),
# stop it and restart the queue so K=40 resumes with 16k evaluations, then K=120 runs.
cd /opt/tiger/Lightning-OPD
D=outputs/qwen3_4b_base_nt_teacher_4b_instruct_2507_k40_byte16
until [ "$(cat $D/checkpoints/latest_checkpointed_iteration.txt 2>/dev/null)" = 99 ]; do sleep 20; done
echo "$(date '+%F %T') step-100 checkpoint saved; stopping K=40"
kill "$(cat tmp/runs/queue_base4b_212.pid)" 2>/dev/null
for p in $(pgrep -f "lightning-opd.py --config configs/config/byte16_qwen3_4b_base_nt_teacher_4b_instruct_2507_k40"); do kill -TERM "$p"; done
for p in $(pgrep -f "gpu_monitor.py --out tmp/gpu_monitor/base4b_k40"); do kill "$p"; done
for i in $(seq 1 60); do
  sleep 10
  [ -z "$(pgrep -f 'lightning-opd.py --config configs/config/byte16_qwen3_4b_base')" ] && \
  [ "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | awk -F, '$2+0>2000' | wc -l)" = 0 ] && break
done
sleep 60
echo "$(date '+%F %T') restarting queue with 32k avg@4 evaluations"
nohup bash tmp/runs/queue_base4b_212.sh >> tmp/runs/queue_base4b_212.log 2>&1 &
echo $! > tmp/runs/queue_base4b_212.pid
echo "$(date '+%F %T') SWITCHED"
