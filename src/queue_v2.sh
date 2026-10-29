#!/usr/bin/env bash
# Run a queue of phase 1 jobs one after another on this GPU: each line holds src/run_v2.sh's arguments
# (blank lines and # comments are skipped). A job whose run already has a registry row is skipped, so a
# restarted queue resumes, and lines appended while the queue runs are picked up.
# Usage: nohup src/queue_v2.sh QUEUE_FILE > LOG 2>&1 &
q=$1; R=/mnt/ml-data/mewc-synth/v2_runs
while IFS= read -r line; do
  case "$line" in ""|\#*) continue ;; esac
  set -- $line
  grep -q "\"name\": \"$2\"" $R/registry.jsonl 2>/dev/null && { echo "[$(date +%H:%M:%S)] skip $2"; continue; }
  echo "[$(date +%H:%M:%S)] start $2"
  bash $HOME/mewc-synth/src/run_v2.sh "$@" < /dev/null
  echo "[$(date +%H:%M:%S)] end $2 ($?)"
done < $q
echo "[$(date +%H:%M:%S)] QUEUE DONE"
