#!/usr/bin/env bash
# Q5 of the third booking (docs/plan-booking-3.md): the one-time scoring of the sealed test and held-out-project
# test snips by the final models. Scores every snip in the original label lists (v2/test.pre_lc.csv and
# v2/test_proj.pre_lc.csv), which hold every snip of the corrected lists, so both label versions can be
# evaluated from one pass. Writes v2_runs/scores/test_<run>.csv and testproj_<run>.csv. Nothing is tuned after.
# Usage: bash ops/2026-11-23/test_score.sh RUN...
set -uo pipefail
M=/mnt/ml-data/mewc-synth; V=$M/v2; R=$M/v2_runs; P=$HOME/mewc-synth/.venv/bin/python
for part in test test_proj; do
  [ -f $V/${part}_all_paths.txt ] || tail -n +2 $V/$part.pre_lc.csv | cut -d, -f1 > $V/${part}_all_paths.txt
done
for run in "$@"; do
  for part in test test_proj; do
    out=$R/scores/$( [ $part = test ] && echo test || echo testproj )_$run
    [ -f $out.csv ] && { echo "skip $out"; continue; }
    (cd $HOME/mewc-synth/src && $P -m mewc_torch.predict $R/runs/$run $V/${part}_all_paths.txt $out 256) \
      > $R/logs/score_${part}_$run.log 2>&1 && echo "scored $part $run" || echo "score $part $run FAILED"
  done
done
echo "TEST SCORING DONE"
