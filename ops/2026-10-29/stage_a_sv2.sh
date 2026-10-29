#!/usr/bin/env bash
# Stage A domain adapter retrained on split v2's training pool (same recipe as a_v2), after gpu-5's
# parity lanes, their label-check scoring and the v2 embedding pass finish. Pseudo-rare classes keep only
# their n=50 v2 subsets.
set -uo pipefail
M=/mnt/ml-data/mewc-synth; W=$M/work; P=$HOME/mewc-synth/.venv/bin/python
cd "$HOME/mewc-synth/src"; export HF_HOME=/mnt/scratch/hf
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "^embedded" $W/embed_v2.log && grep -q "LABELCHECK SCORED" /mnt/scratch/logs/labelcheck.log; do sleep 60; done
stamp "prep start"
$P prep_stage_a.py "$M/v2" "$W/audit_v2" split_v2 "$W/snips_meta/class_table.csv" a_sv2 \
  '{"cap": 1500, "seed": 7, "pseudo_rare": {"spotted_tail_quoll": 50, "forester_kangaroo": 50, "black_currawong": 50, "black_rat": 50}, "ir_chroma": 1.0, "photo_prefixes": ["ebird", "inat"]}' \
  > "$W/prep_a_sv2.log" 2>&1 || { stamp "prep FAILED"; tail -5 "$W/prep_a_sv2.log"; exit 1; }
grep "^{" "$W/prep_a_sv2.log"
stamp "train start"
$P train_lora.py a_sv2 r64_s10k '{"rank": 64, "lr": 1e-4, "steps": 10000, "batch": 32, "alpha": 0.5, "warmup": 300, "min_snr": 5.0, "save_every": 2000, "sample_every": 2000, "seed": 7, "sample_prompts": [
 "camera trap snip of a flame or scarlet robin (Petroica phoenicea or Petroica boodang), daylight colour",
 "photograph of a flame or scarlet robin (Petroica phoenicea or Petroica boodang)",
 "camera trap snip of a scrubtit (Acanthornis magnus), daylight colour",
 "camera trap snip of a tasmanian boobook (Ninox leucopsis), infrared night",
 "camera trap snip of a spotted tail quoll (Dasyurus maculatus), infrared night",
 "camera trap snip of a black currawong (Strepera fuliginosa), daylight colour"]}' \
  > "$W/train_a_sv2.log" 2>&1 || { stamp "train FAILED"; tail -5 "$W/train_a_sv2.log"; exit 1; }
stamp "STAGE A SV2 DONE"
