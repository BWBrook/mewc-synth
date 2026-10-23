#!/usr/bin/env bash
set -uo pipefail
M=/mnt/ml-data/mewc-synth; W=$M/work; P=$HOME/mewc-synth/.venv/bin/python; L=$M/stage_a/a_v2/lora; O=$M/translate_screen; R=$M/screen
cd "$HOME/mewc-synth/src"; export HF_HOME=/mnt/scratch/hf
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q -E "STAGE B2 DONE|FAILED" $W/stage_b2.log; do sleep 20; done
grep -q FAILED $W/stage_b2.log && { stamp "stage B2 failed"; exit 1; }
tr() { stamp "translate $2 -> $3"; $P translate.py $L/$1 $W/snips_meta/class_table.csv $W/audit split_v1 $O \
  "{\"source\": \"$2\", \"target\": \"$3\", \"n\": 200, \"strengths\": [0.65], \"guidance\": 3.5, \"steps\": 30, \"seed\": 51, \"ir_chroma\": 1.0, \"q95_share\": $4}" 2>&1 | grep "^{" || { stamp "translate FAILED"; exit 1; }; }
tr b_quoll_short/step_000200 eastern_quoll spotted_tail_quoll 0.089
tr b_currawong/step_000400 grey_currawong black_currawong 0.332
tr b_kangaroo/step_000400 bennetts_wallaby forester_kangaroo 0.0
tr b_rat/step_000400 swamp_rat black_rat 0.014
stamp "build A1"
$P build_screen.py $W/audit split_v1 $W/snips_meta/class_table.csv $R \
  '{"arm": "A1", "train_cap": 4000, "val_cap": 100, "seed": 41, "pseudo_rare": {"spotted_tail_quoll": 50, "forester_kangaroo": 50, "black_currawong": 50, "black_rat": 50}, "synthetic": {"spotted_tail_quoll": "translate_screen", "forester_kangaroo": "translate_screen", "black_currawong": "translate_screen", "black_rat": "translate_screen"}, "synthetic_n": 200}' || { stamp "build A1 FAILED"; exit 1; }
for run in "A0 1" "A1 1" "R 1" "A0 2" "A1 2"; do
  set -- $run; stamp "run $1 seed $2"
  bash $HOME/mewc-synth/src/run_screen.sh $1 $2 || stamp "run $1 seed $2 FAILED"
done
stamp "identity of translate_screen"
find $O -name "synth_*.jpg" | sort | sed "s#^$M#/data#" > $O/paths.txt
sudo docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 -v $M:/data -v $HOME/mewc-synth/src/vtl_score.py:/code/vtl_score.py \
  zaandahl/mewc-train:2.0.9 python /code/vtl_score.py /data/models/vtl_mewc.keras /data/models/class_list_96sp.yaml /data/translate_screen/paths.txt /data/translate_screen/vtl 2>&1 | grep "^scored"
$P identity_report.py $O $M/models/class_list_96sp.yaml $M/models/sanity $M/models/sanity_truth.csv > $O/report.txt 2>&1
$P memorisation.py $O $W/audit split_v1 a_v2 > $O/memorisation.txt 2>&1
stamp "evaluate"
$P screen_eval.py $R '{"focus": ["spotted_tail_quoll", "forester_kangaroo", "black_currawong", "black_rat"], "baseline": "A0", "arms": {"A0": ["A0_s1", "A0_s2"], "A1": ["A1_s1", "A1_s2"], "R": ["R_s1"]}}' > $R/eval.txt 2>&1
stamp "QUEUE DONE"
