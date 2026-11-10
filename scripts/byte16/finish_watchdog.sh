#!/usr/bin/env bash
# End train.py when it hangs after finishing: training_status.json says "complete" (written after the final
# evaluation and checkpoint) but the process is still alive 10 minutes later. Seen 11/05: 0.6B base K=1 hung
# 40+ minutes in the rollout manager's final W&B upload while its results were complete, blocking the queue.
cd /opt/tiger/Lightning-OPD
while true; do
  for status in outputs/*/training_status.json; do
    [ -f "$status" ] || continue
    age=$(( $(date +%s) - $(stat -c %Y "$status") ))
    [ "$age" -gt 600 ] && [ "$age" -lt 86400 ] || continue
    run=$(dirname "$status")
    for p in $(pgrep -f "train.py .*--save $(pwd)/$run/checkpoints"); do
      echo "$(date '+%F %T') $run finished ${age}s ago but train.py $p is still alive; stopping it"
      kill -TERM "$p"; sleep 60; kill -0 "$p" 2>/dev/null && kill -KILL "$p"
    done
  done
  sleep 60
done
