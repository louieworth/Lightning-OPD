#!/usr/bin/env bash
# Start the K=20 queue once the base v2 queue (0.6B/1.7B K=1/40/120) has finished.
cd /opt/tiger/Lightning-OPD
until grep -q QUEUE_DONE tmp/runs/queue_base_v2.log; do sleep 60; done
sleep 60
bash tmp/runs/queue_k20.sh > tmp/runs/queue_k20.log 2>&1
