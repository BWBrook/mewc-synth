#!/usr/bin/env bash
# Run a queue of phase 1 jobs one after another on this GPU: each line holds src/run_v2.sh's arguments
# (blank lines and # comments are skipped). A job whose run already has a registry row is skipped, so a
# restarted queue resumes, and lines appended while the queue runs are picked up. Each job is claimed
# (v2_runs/claims/NAME) before it starts, so two lanes can share one queue file; to rerun a job, remove its
# claim. A failed job is retried up to three times, five minutes apart (training resumes from its last
# checkpoint), since a second CUDA process on the Nectar vGPU sometimes fails at its first kernel with
# CUDA_ERROR_LAUNCH_FAILED while the GPU's other lane is busy. With PRUNE set, a finished run keeps only best.pt.
# Usage: [PRUNE=1] nohup src/queue_v2.sh QUEUE_FILE > LOG 2>&1 &
q=$1; R=/mnt/ml-data/mewc-synth/v2_runs
mkdir -p $R/claims
while IFS= read -r line; do
  case "$line" in ""|\#*) continue ;; esac
  set -- $line
  grep -q "\"name\": \"$2\"" $R/registry.jsonl 2>/dev/null && { echo "[$(date +%H:%M:%S)] skip $2"; continue; }
  mkdir $R/claims/$2 2>/dev/null || { echo "[$(date +%H:%M:%S)] skip $2 (claimed)"; continue; }
  echo "[$(date +%H:%M:%S)] start $2"
  for try in 1 2 3 4; do
    bash $HOME/mewc-synth/src/run_v2.sh "$@" < /dev/null; rc=$?
    { [ $rc = 0 ] || [ $try = 4 ]; } && break
    echo "[$(date +%H:%M:%S)] retry $2"; sleep 300
  done
  [ $rc = 0 ] && [ -n "$PRUNE" ] && rm -f $R/runs/$2/last.pt $R/runs/$2/frozen_best.pt
  echo "[$(date +%H:%M:%S)] end $2 ($rc)"
done < $q
echo "[$(date +%H:%M:%S)] QUEUE DONE"
