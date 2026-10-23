#!/usr/bin/env bash
# Extra screening seeds on ngx-gpu-5 (2026-10-22), complementing the ngx-gpu-4 queue (A0/A1 seeds 1-2, R seed 1).
set -uo pipefail
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
for run in "A0 3" "A1 3" "R 2"; do
  set -- $run; stamp "run $1 seed $2"
  bash $HOME/mewc-synth/src/run_screen.sh $1 $2 || stamp "run $1 seed $2 FAILED"
done
stamp "G5 QUEUE DONE"
