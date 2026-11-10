#!/usr/bin/env bash
# Start the Qwen3-4B-Base trio once the Qwen3-1.7B-Base queue has finished.
cd /opt/tiger/Lightning-OPD
until grep -q QUEUE_DONE tmp/runs/queue_base.log; do sleep 60; done
sleep 60
bash tmp/runs/queue_base4b.sh > tmp/runs/queue_base4b.log 2>&1
