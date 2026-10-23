#!/usr/bin/env bash
# Production-model check (PLAN: VTL for the best option against baseline): A0 vs S20, seeds 1-2, ngx-gpu-5, 2026-10-23.
set -uo pipefail
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
[ -d /mnt/ml-data/mewc-synth/screen/S20 ] || { stamp "S20 arm missing"; exit 1; }
for run in "A0 1" "S20 1" "A0 2" "S20 2"; do
  set -- $run; stamp "run VTL $1 seed $2"; MODEL=VTL bash $HOME/mewc-synth/src/run_screen.sh $1 $2 || stamp "run VTL $1 seed $2 FAILED"
done
stamp "VTL G5 DONE"
