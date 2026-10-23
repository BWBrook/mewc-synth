#!/usr/bin/env bash
# Seeds 6-7 for A0 and S20 (VTB) on ngx-gpu-5 after the VTL check, 2026-10-23.
set -uo pipefail
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "VTL G5 DONE" /mnt/scratch/logs/vtl_g5.log; do sleep 60; done
for run in "A0 6" "S20 6" "A0 7" "S20 7"; do
  set -- $run; stamp "run $1 seed $2"; bash $HOME/mewc-synth/src/run_screen.sh $1 $2 || stamp "run $1 seed $2 FAILED"
done
stamp "SEEDS G5 DONE"
