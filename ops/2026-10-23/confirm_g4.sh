#!/usr/bin/env bash
# Confirmation seeds (A0, S20 seeds 4-5) and the S10 share arm on ngx-gpu-4, 2026-10-23.
set -uo pipefail
M=/mnt/ml-data/mewc-synth; W=$M/work; P=$HOME/mewc-synth/.venv/bin/python
cd "$HOME/mewc-synth/src"
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
for run in "A0 4" "S20 4" "A0 5" "S20 5"; do
  set -- $run; stamp "run $1 seed $2"; bash $HOME/mewc-synth/src/run_screen.sh $1 $2 || stamp "run $1 seed $2 FAILED"
done
$P build_screen.py $W/audit split_v1 $W/snips_meta/class_table.csv $M/screen '{"arm": "S10", "train_cap": 4000, "val_cap": 100, "seed": 41, "pseudo_rare": {"spotted_tail_quoll": 50, "forester_kangaroo": 50, "black_currawong": 50, "black_rat": 50}, "synthetic": {"spotted_tail_quoll": "translate_screen", "forester_kangaroo": "translate_screen", "black_currawong": "translate_screen", "black_rat": "translate_screen"}, "synthetic_n": 200, "real_repeat": {"spotted_tail_quoll": 36, "forester_kangaroo": 36, "black_currawong": 36, "black_rat": 36}}' || { stamp "build S10 FAILED"; exit 1; }
for seed in 1 2 3; do
  stamp "run S10 seed $seed"; bash $HOME/mewc-synth/src/run_screen.sh S10 $seed || stamp "run S10 seed $seed FAILED"
done
stamp "CONFIRM G4 DONE"
