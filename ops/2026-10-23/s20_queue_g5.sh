#!/usr/bin/env bash
# S20 (translated synthetic at a 20% share: real x16 links), queued behind the bird arm, 2026-10-22.
set -uo pipefail
M=/mnt/ml-data/mewc-synth; W=$M/work; P=$HOME/mewc-synth/.venv/bin/python
cd "$HOME/mewc-synth/src"
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "BIRD G5 DONE" /mnt/scratch/logs/bird_queue_g5.log; do sleep 60; done
[ -d $M/screen/S20 ] || $P build_screen.py $W/audit split_v1 $W/snips_meta/class_table.csv $M/screen '{"arm": "S20", "train_cap": 4000, "val_cap": 100, "seed": 41, "pseudo_rare": {"spotted_tail_quoll": 50, "forester_kangaroo": 50, "black_currawong": 50, "black_rat": 50}, "synthetic": {"spotted_tail_quoll": "translate_screen", "forester_kangaroo": "translate_screen", "black_currawong": "translate_screen", "black_rat": "translate_screen"}, "synthetic_n": 200, "real_repeat": {"spotted_tail_quoll": 16, "forester_kangaroo": 16, "black_currawong": 16, "black_rat": 16}}' || { stamp "build S20 FAILED"; exit 1; }
for seed in 1 2; do
  stamp "run S20 seed $seed"
  bash $HOME/mewc-synth/src/run_screen.sh S20 $seed || stamp "run S20 seed $seed FAILED"
done
stamp "S20 QUEUE DONE"
