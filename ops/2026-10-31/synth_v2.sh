#!/usr/bin/env bash
# Synthetic set v2 for the four pseudo-rare classes, the S20 recipe on the v2 domain adapter: Stage B adapters
# trained on a_sv2 (configs in ops/2026-10-31/synth), 200 look-alike translations per class at strength 0.65,
# then the identity (production VTL), memorisation and shortcut (pair) checks. Prints SYNTH V2 END on exit.
set -uo pipefail
M=/mnt/ml-data/mewc-synth; W=$M/work; C=$HOME/mewc-synth/ops/2026-10-31/synth; P=$HOME/mewc-synth/.venv/bin/python
L=$M/stage_a/a_sv2/lora; O=$M/translate_v2
cd "$HOME/mewc-synth/src"; export HF_HOME=/mnt/scratch/hf
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
trap 'stamp "SYNTH V2 END"' EXIT
for a in quoll_short currawong kangaroo rat; do
  stamp "train b_$a"
  $P train_lora.py a_sv2 b_$a "$(cat $C/cfg_$a.json)" > $W/synth_v2_train_$a.log 2>&1 \
    || { stamp "train b_$a FAILED"; tail -n 5 $W/synth_v2_train_$a.log; exit 1; }
done
tr() { stamp "translate $2 -> $3"; $P translate.py $L/$1 $W/snips_meta/class_table.csv $W/audit_v2 split_v2 $O \
  "{\"source\": \"$2\", \"target\": \"$3\", \"n\": 200, \"strengths\": [0.65], \"guidance\": 3.5, \"steps\": 30, \"seed\": 61, \"ir_chroma\": 1.0, \"q95_share\": $4}" 2>&1 | grep "^{" || { stamp "translate FAILED"; exit 1; }; }
tr b_quoll_short/step_000200 eastern_quoll spotted_tail_quoll 0.089
tr b_currawong/step_000400 grey_currawong black_currawong 0.332
tr b_kangaroo/step_000400 bennetts_wallaby forester_kangaroo 0.0
tr b_rat/step_000400 swamp_rat black_rat 0.014
stamp "TRANSLATED"
find $O -name "synth_*.jpg" | sort | sed "s#^$M#/data#" > $O/paths.txt
sudo docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 -v $M:/data -v $HOME/mewc-synth/src/vtl_score.py:/code/vtl_score.py \
  zaandahl/mewc-train:2.0.9 python /code/vtl_score.py /data/models/vtl_mewc.keras /data/models/class_list_96sp.yaml /data/translate_v2/paths.txt /data/translate_v2/vtl 2>&1 | grep "^scored" \
  && $P identity_report.py $O $M/models/class_list_96sp.yaml $M/models/sanity $M/models/sanity_truth.csv > $O/report.txt 2>&1 || stamp "identity FAILED"
$P memorisation.py $O $W/audit_v2 split_v2 a_sv2 > $O/memorisation.txt 2>&1 || stamp "memorisation FAILED"
$P shortcut_probe.py $O pair 2>&1 | grep "^{\"mode" > $O/probe.txt || stamp "probe FAILED"
stamp "SYNTH V2 DONE"
