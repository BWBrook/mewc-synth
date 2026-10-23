#!/usr/bin/env bash
# Bird arm on ngx-gpu-4 (2026-10-22): translate own camera-trap snips, identity, probe, build, runs.
set -uo pipefail
M=/mnt/ml-data/mewc-synth; W=$M/work; P=$HOME/mewc-synth/.venv/bin/python; O=$M/translate_birds; R=$M/screen
cd "$HOME/mewc-synth/src"; export HF_HOME=/mnt/scratch/hf
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
for t in pacific_black_duck flame_robin yellow_throated_honeyeater strong_billed_honeyeater pink_robin crescent_honeyeater wedge_tailed_eagle grey_shrikethrush maned_duck; do
  stamp "translate $t"
  $P translate.py $M/stage_a/a_v2/lora/r64_s10k/step_010000 $W/snips_meta/class_table.csv $W/audit split_v1 $O \
    "{\"source\": \"$t\", \"target\": \"$t\", \"n\": 400, \"strengths\": [0.55], \"guidance\": 3.5, \"steps\": 30, \"seed\": 71, \"ir_chroma\": 1.0, \"q95_share\": 0.0}" 2>&1 | grep "^{" || { stamp "translate $t FAILED"; exit 1; }
done
stamp "identity"
find $O -name "synth_*.jpg" | sort | sed "s#^$M#/data#" > $O/paths.txt
sudo docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 -v $M:/data -v $HOME/mewc-synth/src/vtl_score.py:/code/vtl_score.py \
  zaandahl/mewc-train:2.0.9 python /code/vtl_score.py /data/models/vtl_mewc.keras /data/models/class_list_96sp.yaml /data/translate_birds/paths.txt /data/translate_birds/vtl 2>&1 | grep "^scored"
$P identity_report.py $O $M/models/class_list_96sp.yaml $M/models/sanity $M/models/sanity_truth.csv > $O/report.txt 2>&1
stamp "shortcut probe (pair: translation vs its own source)"
$P shortcut_probe.py $O pair 2>&1 | grep "^{\"mode" > $O/probe.txt
stamp "build Bsyn and Brep"
$P build_screen.py $W/audit split_v1 $W/snips_meta/class_table.csv $R '{"arm": "Bsyn", "train_cap": 4000, "val_cap": 100, "seed": 41, "synthetic": {"pacific_black_duck": "translate_birds", "flame_robin": "translate_birds", "yellow_throated_honeyeater": "translate_birds", "strong_billed_honeyeater": "translate_birds", "pink_robin": "translate_birds", "crescent_honeyeater": "translate_birds", "wedge_tailed_eagle": "translate_birds", "grey_shrikethrush": "translate_birds", "maned_duck": "translate_birds"}, "synthetic_n": 400}' || { stamp "build Bsyn FAILED"; exit 1; }
$P build_screen.py $W/audit split_v1 $W/snips_meta/class_table.csv $R '{"arm": "Brep", "train_cap": 4000, "val_cap": 100, "seed": 41, "extra_real": {"pacific_black_duck": 400, "flame_robin": 400, "yellow_throated_honeyeater": 400, "strong_billed_honeyeater": 400, "pink_robin": 400, "crescent_honeyeater": 400, "wedge_tailed_eagle": 400, "grey_shrikethrush": 400, "maned_duck": 400}}' || { stamp "build Brep FAILED"; exit 1; }
for run in "Bsyn 1" "Bsyn 2" "Bsyn 3" "Brep 3"; do
  set -- $run; stamp "run $1 seed $2"
  bash $HOME/mewc-synth/src/run_screen.sh $1 $2 || stamp "run $1 seed $2 FAILED"
done
stamp "BIRD G4 DONE"
