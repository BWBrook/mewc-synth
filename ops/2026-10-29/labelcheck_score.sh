#!/usr/bin/env bash
# Label check, scoring step: once this GPU's parity lanes finish, score the test and project-held-out lists
# with every mewc-torch run on this instance. Output stays in v2_runs/labelcheck, apart from the run scores;
# no test metrics are computed from it. Usage: labelcheck_score.sh QUEUE_LOG...
set -uo pipefail
M=/mnt/ml-data/mewc-synth; V=$M/v2; R=$M/v2_runs; L=$R/labelcheck; P=$HOME/mewc-synth/.venv/bin/python
for q in "$@"; do until grep -q "QUEUE DONE" $q; do sleep 60; done; done
mkdir -p $L
for part in test test_proj; do
  [ -f $V/${part}_paths.txt ] || $P -c "import pandas as pd; open('$V/${part}_paths.txt', 'w').write('\n'.join(pd.read_csv('$V/$part.csv')['path']) + '\n')"
done
cd $HOME/mewc-synth/src
for run in $R/runs/pt_*/; do
  name=$(basename $run); [ -f $run/best.pt ] || continue
  for part in test test_proj; do
    [ -f $L/${name}_${part}_probs.npy ] || $P -m mewc_torch.predict $run $V/${part}_paths.txt $L/${name}_${part} 256 \
      >> $L/score.log 2>&1 || echo "$name $part FAILED"
  done
done
echo "[$(date +%H:%M:%S)] LABELCHECK SCORED"
