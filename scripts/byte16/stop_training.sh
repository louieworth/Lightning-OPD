#!/usr/bin/env bash
# Stop every queue, launcher and per-run GPU monitor (not the HF exporter), then wait until the GPUs are free.
# Kept in a file so the match patterns never appear in the calling shell's own command line.
cd /opt/tiger/Lightning-OPD
for f in tmp/runs/queue_*.pid; do kill "$(cat "$f")" 2>/dev/null; done
for p in $(pgrep -f "bash tmp/runs/queue_"); do kill "$p"; done
for p in $(pgrep -f "lightning-opd.py --config|qwen3-4b-opd.py --config"); do kill -TERM "$p"; done
for p in $(pgrep -f "tools/gpu_monitor.py"); do kill "$p"; done
for i in $(seq 1 60); do
  sleep 5
  [ -z "$(pgrep -f 'lightning-opd.py --config|qwen3-4b-opd.py --config')" ] && \
  [ "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | awk -F, '$2+0>2000' | wc -l)" = 0 ] && break
done
echo "launchers left: $(pgrep -fc 'lightning-opd.py --config|qwen3-4b-opd.py --config')"
echo "head GPU procs >2GB: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | awk -F, '$2+0>2000' | wc -l)"
