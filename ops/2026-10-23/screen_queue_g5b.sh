#!/usr/bin/env bash
# S50: translated synthetic at a 50% share (real x4 links), seeds 1-3, ngx-gpu-5, 2026-10-22.
set -uo pipefail
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
for seed in 1 2 3; do
  stamp "run S50 seed $seed"
  bash $HOME/mewc-synth/src/run_screen.sh S50 $seed || stamp "run S50 seed $seed FAILED"
done
stamp "G5B QUEUE DONE"
