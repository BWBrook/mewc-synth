#!/usr/bin/env bash
# Second synthetic set (S20b) on ngx-gpu-4 after the confirmation queue, 2026-10-23.
set -uo pipefail
M=/mnt/ml-data/mewc-synth; W=$M/work; P=$HOME/mewc-synth/.venv/bin/python; L=$M/stage_a/a_v2/lora; O=$M/translate_screen2
cd "$HOME/mewc-synth/src"; export HF_HOME=/mnt/scratch/hf
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "CONFIRM G4 DONE" /mnt/scratch/logs/confirm_g4.log; do sleep 60; done
tr() { stamp "translate $2 -> $3"; $P translate.py $L/$1 $W/snips_meta/class_table.csv $W/audit split_v1 $O \
  "{\"source\": \"$2\", \"target\": \"$3\", \"n\": 200, \"strengths\": [0.65], \"guidance\": 3.5, \"steps\": 30, \"seed\": 52, \"ir_chroma\": 1.0, \"q95_share\": $4}" 2>&1 | grep "^{" || { stamp "translate FAILED"; exit 1; }; }
tr b_quoll_short/step_000200 eastern_quoll spotted_tail_quoll 0.089
tr b_currawong/step_000400 grey_currawong black_currawong 0.332
tr b_kangaroo/step_000400 bennetts_wallaby forester_kangaroo 0.0
tr b_rat/step_000400 swamp_rat black_rat 0.014
stamp "identity and probe"
find $O -name "synth_*.jpg" | sort | sed "s#^$M#/data#" > $O/paths.txt
sudo docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 -v $M:/data -v $HOME/mewc-synth/src/vtl_score.py:/code/vtl_score.py \
  zaandahl/mewc-train:2.0.9 python /code/vtl_score.py /data/models/vtl_mewc.keras /data/models/class_list_96sp.yaml /data/translate_screen2/paths.txt /data/translate_screen2/vtl 2>&1 | grep "^scored"
$P identity_report.py $O $M/models/class_list_96sp.yaml $M/models/sanity $M/models/sanity_truth.csv > $O/report.txt 2>&1
$P shortcut_probe.py $O pair 2>&1 | grep "^{\"mode" > $O/probe.txt
$P build_screen.py $W/audit split_v1 $W/snips_meta/class_table.csv $M/screen '{"arm": "S20b", "train_cap": 4000, "val_cap": 100, "seed": 41, "pseudo_rare": {"spotted_tail_quoll": 50, "forester_kangaroo": 50, "black_currawong": 50, "black_rat": 50}, "synthetic": {"spotted_tail_quoll": "translate_screen2", "forester_kangaroo": "translate_screen2", "black_currawong": "translate_screen2", "black_rat": "translate_screen2"}, "synthetic_n": 200, "real_repeat": {"spotted_tail_quoll": 16, "forester_kangaroo": 16, "black_currawong": 16, "black_rat": 16}}' || { stamp "build S20b FAILED"; exit 1; }
for seed in 1 2 3; do
  stamp "run S20b seed $seed"; bash $HOME/mewc-synth/src/run_screen.sh S20b $seed || stamp "run S20b seed $seed FAILED"
done
stamp "SET2 G4 DONE"
