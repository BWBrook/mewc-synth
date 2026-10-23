#!/usr/bin/env bash
# Bird arm on ngx-gpu-5 (2026-10-22): after S50, build Brep and run R s3, Brep s1-2.
set -uo pipefail
M=/mnt/ml-data/mewc-synth; W=$M/work; P=$HOME/mewc-synth/.venv/bin/python; R=$M/screen
cd "$HOME/mewc-synth/src"
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "G5B QUEUE DONE" /mnt/scratch/logs/screen_queue_g5b.log; do sleep 60; done
stamp "build Brep"
$P build_screen.py $W/audit split_v1 $W/snips_meta/class_table.csv $R '{"arm": "Brep", "train_cap": 4000, "val_cap": 100, "seed": 41, "extra_real": {"pacific_black_duck": 400, "flame_robin": 400, "yellow_throated_honeyeater": 400, "strong_billed_honeyeater": 400, "pink_robin": 400, "crescent_honeyeater": 400, "wedge_tailed_eagle": 400, "grey_shrikethrush": 400, "maned_duck": 400}}' || { stamp "build Brep FAILED"; exit 1; }
for run in "R 3" "Brep 1" "Brep 2"; do
  set -- $run; stamp "run $1 seed $2"
  bash $HOME/mewc-synth/src/run_screen.sh $1 $2 || stamp "run $1 seed $2 FAILED"
done
stamp "BIRD G5 DONE"
