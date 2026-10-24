#!/usr/bin/env bash
# Eight-seed VTL check (A0 vs S20), seeds 4 6 8 on this machine, 2026-10-24.
set -uo pipefail
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
for seed in 4 6 8; do
  for arm in A0 S20; do
    stamp "run VTL $arm seed $seed"; MODEL=VTL bash $HOME/mewc-synth/src/run_screen.sh $arm $seed || stamp "run VTL $arm seed $seed FAILED"
  done
done
stamp "VTL8 g5 DONE"
