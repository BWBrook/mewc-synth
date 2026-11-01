#!/usr/bin/env bash
# Phase 1 synthetic re-test on gpu-5: after its mini-search lane, apply the training-pool corrections to this
# instance's train.csv, build the arm lists and run the 26 ViT-B runs in one lane.
set -uo pipefail
M=/mnt/ml-data/mewc-synth; H=$HOME/mewc-synth; P=$H/.venv/bin/python
until grep -q "QUEUE DONE" /mnt/scratch/logs/q_mini_g5a.log; do sleep 120; done
$P $H/src/apply_label_corrections.py $M/v2 $H/ops/2026-10-31/train_corrections.csv || { echo "corrections FAILED"; exit 1; }
$P $H/ops/2026-11-01/retest_build.py $M/v2 $M/translate_v2 $M/stage_a/a_sv2/rows.parquet $M/v2_runs/arms || { echo "arms FAILED"; exit 1; }
PRUNE=1 bash $H/src/queue_v2.sh $H/ops/2026-11-01/retest.queue
