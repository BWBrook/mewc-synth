#!/usr/bin/env bash
# Label check of the training pool, prompted by same-first-name misclicks (eastern quoll / eastern rosella and
# others) in the evaluation lists: once this GPU's race lanes finish, score every train.csv snip with four
# phase 1 runs (ConvNeXt-B batch 64 and ViT-B batch 16, two seeds each) and list confident disagreements.
# Usage: trainscan.sh QUEUE_LOG...
set -uo pipefail
M=/mnt/ml-data/mewc-synth; V=$M/v2; R=$M/v2_runs; L=$R/labelcheck; P=$HOME/mewc-synth/.venv/bin/python
runs="pt_cnb_bf16b64_s1 pt_cnb_bf16b64_s2 pt_vtb_bf16_s1 pt_vtb_bf16_s2"
for q in "$@"; do until grep -q "QUEUE DONE" $q; do sleep 60; done; done
echo "[$(date +%H:%M:%S)] scoring"
[ -f $V/train_paths.txt ] || $P -c "import pandas as pd; open('$V/train_paths.txt', 'w').write('\n'.join(pd.read_csv('$V/train.csv')['path']) + '\n')"
cd $HOME/mewc-synth/src
for run in $runs; do
  [ -f $L/${run}_train_probs.npy ] || $P -m mewc_torch.predict $R/runs/$run $V/train_paths.txt $L/${run}_train 256 \
    >> $L/trainscan.log 2>&1 || echo "$run FAILED"
  echo "[$(date +%H:%M:%S)] $run scored"
done
$P label_check.py $R $V $L/flagged_train_p90.csv 0.9 train $runs
echo "[$(date +%H:%M:%S)] TRAINSCAN DONE"
