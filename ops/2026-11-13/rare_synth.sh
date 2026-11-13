#!/usr/bin/env bash
# Q1 generation on ngx-gpu-5 (docs/plan-booking-3.md): re-encode the a_sv2 latents (cleared on 8 November) and
# check the rows match the saved ones, train a Stage B adapter per real low-count class on a_sv2, translate
# 1,600 look-alike snips per class with the S20 recipe, run the identity (production VTL), memorisation and
# shortcut checks, then build the S20 training list. Prints RARE SYNTH DONE on success.
set -uo pipefail
M=/mnt/ml-data/mewc-synth; W=$M/work; H=$HOME/mewc-synth; P=$H/.venv/bin/python; O13=$H/ops/2026-11-13
A=$M/stage_a/a_sv2; L=$A/lora; O=$M/translate_rare
cd "$H/src"; export HF_HOME=/mnt/scratch/hf
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { stamp "$1 FAILED"; exit 1; }
if [ ! -s $A/latents.npy ]; then
  stamp "re-encode a_sv2 latents"
  $P prep_stage_a.py "$M/v2" "$W/audit_v2" split_v2 "$W/snips_meta/class_table.csv" a_sv2_lat \
    '{"cap": 1500, "seed": 7, "pseudo_rare": {"spotted_tail_quoll": 50, "forester_kangaroo": 50, "black_currawong": 50, "black_rat": 50}, "ir_chroma": 1.0, "photo_prefixes": ["ebird", "inat"]}' \
    > $W/prep_a_sv2_lat.log 2>&1 || fail "prep"
  $P -c "import pandas as pd,sys; a=pd.read_parquet('$A/rows.parquet'); b=pd.read_parquet('$M/stage_a/a_sv2_lat/rows.parquet'); sys.exit(0 if a.equals(b) else 1)" \
    || fail "rows check"
  mv $M/stage_a/a_sv2_lat/latents.npy $A/latents.npy || fail "move latents"
  stamp "latents in place, rows identical"
fi
$P $O13/rare_cfgs.py $O13/rare_profile.csv $O13/rare_cfgs || fail "configs"
while IFS=, read -r cls look steps; do
  ls -d $L/b_$cls/step_* > /dev/null 2>&1 && continue
  stamp "train b_$cls ($look, $steps steps)"
  $P train_lora.py a_sv2 b_$cls "$(cat $O13/rare_cfgs/cfg_$cls.json)" > $W/rare_train_$cls.log 2>&1 || fail "train b_$cls"
done < $O13/rare_cfgs/plan.csv
while IFS=, read -r cls look steps; do
  [ -f $O/$cls/manifest_done ] && continue
  stamp "translate $look -> $cls"
  $P translate.py $L/b_$cls/step_$(printf %06d $steps) $W/snips_meta/class_table.csv $W/audit_v2 split_v2 $O \
    "{\"source\": \"$look\", \"target\": \"$cls\", \"n\": 1600, \"strengths\": [0.65], \"guidance\": 3.5, \"steps\": 30, \"seed\": 71, \"ir_chroma\": 1.0, \"q95_share\": 0.0, \"snip_root\": \"$M/v2\"}" \
    > $W/rare_translate_$cls.log 2>&1 || fail "translate $cls"
  touch $O/$cls/manifest_done
done < $O13/rare_cfgs/plan.csv
stamp "TRANSLATED"
find $O -name "synth_*.jpg" | sort | sed "s#^$M#/data#" > $O/paths.txt
sudo docker pull -q zaandahl/mewc-train:2.0.9 > /dev/null || stamp "docker pull FAILED"
sudo docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 -v $M:/data -v $H/src/vtl_score.py:/code/vtl_score.py \
  zaandahl/mewc-train:2.0.9 python /code/vtl_score.py /data/models/vtl_mewc.keras /data/models/class_list_96sp.yaml /data/translate_rare/paths.txt /data/translate_rare/vtl 2>&1 | grep "^scored" \
  && $P identity_report.py $O $M/models/class_list_96sp.yaml $M/models/sanity $M/models/sanity_truth.csv > $O/report.txt 2>&1 || stamp "identity FAILED"
$P memorisation.py $O $W/audit_v2 split_v2 a_sv2 > $O/memorisation.txt 2>&1 || stamp "memorisation FAILED"
$P shortcut_probe.py $O pair 2>&1 | grep "^{\"mode" > $O/probe.txt || stamp "probe FAILED"
$P $O13/rare_arms.py $M/v2/train.csv $O $O13/rare_profile.csv $M/v2_runs/arms/rare_S20.csv || fail "arms"
stamp "RARE SYNTH DONE"
