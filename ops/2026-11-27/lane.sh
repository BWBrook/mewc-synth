#!/usr/bin/env bash
# Q7 lane (PLAN 2026-11-27): score the training snips with this lane's 8 teacher runs, build the ensemble teacher,
# then run the lane's queue. Prints TEACHER DONE before the queue.
# Usage: bash ops/2026-11-27/lane.sh A|B
set -uo pipefail
lane=$1; M=/mnt/ml-data/mewc-synth; V=$M/v2; R=$M/v2_runs; H=$HOME/mewc-synth; P=$H/.venv/bin/python
[ "$lane" = A ] && seeds="1 3" || seeds="2 4"
runs=""; for m in dinov3_l cnv2_l eva02_l eva02_b_anchor; do for s in $seeds; do runs="$runs md_${m}_s$s"; done; done
[ -f $V/train_paths_ordered.txt ] || tail -n +2 $V/train.csv | cut -d, -f1 > $V/train_paths_ordered.txt
for run in $runs; do
  [ -f $R/scores/train_${run}_probs.npy ] && continue
  (cd $H/src && $P -m mewc_torch.predict $R/runs/$run $V/train_paths_ordered.txt $R/scores/train_$run 256) \
    > $R/logs/score_train_$run.log 2>&1 || { echo "train scoring $run FAILED"; exit 1; }
  echo "[$(date +%H:%M:%S)] scored train $run"
done
$P $H/ops/2026-11-27/ens_teacher.py $R/scores $V/train.csv $V/val.csv $V/classes.txt $R/arms/teacher_ens_$lane $runs \
  || { echo "teacher FAILED"; exit 1; }
echo "TEACHER DONE"
PRUNE=1 exec bash $H/src/queue_v2.sh $H/ops/2026-11-27/q7_$lane.queue
