#!/usr/bin/env bash
# Q6 on ngx-gpu-4: score the training list with the three train-only EVA-02-B medium runs, average their
# probabilities into the teacher targets, then run the distillation queue. Prints KD PREP DONE before the queue.
set -uo pipefail
M=/mnt/ml-data/mewc-synth; V=$M/v2; R=$M/v2_runs; H=$HOME/mewc-synth; P=$H/.venv/bin/python
[ -f $V/train_paths_ordered.txt ] || tail -n +2 $V/train.csv | cut -d, -f1 > $V/train_paths_ordered.txt
for s in 1 2 3; do
  run=md_eva02_b_anchor_s$s
  [ -f $R/scores/train_$run.csv ] && continue
  (cd $H/src && $P -m mewc_torch.predict $R/runs/$run $V/train_paths_ordered.txt $R/scores/train_$run 256) \
    > $R/logs/score_train_$run.log 2>&1 || { echo "train scoring $run FAILED"; exit 1; }
done
$P $H/ops/2026-11-23/kd_prep.py $V/train.csv $R/arms/teacher_eva02_b_soft.npy \
  $R/scores/train_md_eva02_b_anchor_s1 $R/scores/train_md_eva02_b_anchor_s2 $R/scores/train_md_eva02_b_anchor_s3 \
  || { echo "kd prep FAILED"; exit 1; }
echo "KD PREP DONE"
PRUNE=1 exec bash $H/src/queue_v2.sh $H/ops/2026-11-23/kd.queue
