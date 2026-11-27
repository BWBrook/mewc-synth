#!/usr/bin/env bash
# Q7 follow-on for one lane (PLAN 2026-11-27), run after the lane's queue: the refit's teacher targets, the
# conditional refit (used only if Q7's rule passes for that student), then the one-time test scoring of the
# lane's students and the refit, reported whatever the outcome.
# Usage: wait_then.sh LANE_LOG -- bash ops/2026-11-27/followon.sh A|B
set -uo pipefail
lane=$1; M=/mnt/ml-data/mewc-synth; V=$M/v2; R=$M/v2_runs; H=$HOME/mewc-synth; P=$H/.venv/bin/python
$P $H/ops/2026-11-27/refit_prep.py $R/arms/train_val.csv $V/train.csv $V/val.csv $R/arms/teacher_ens_$lane || exit 1
PRUNE=1 bash $H/src/queue_v2.sh $H/ops/2026-11-27/refit_$lane.queue
if [ $lane = A ]; then runs="ekd_hgnetv2_b4_s1 ekd_hgnetv2_b4_s2 ekd_hgnetv2_b4_s3 ekd_eva02_b_s1 ekd_eva02_b_s2 ekd_eva02_b_s3 ekd_eva02_b_s4 rfekd_eva02_b_s1"
else runs="kd_hgnetv2_b4_s5 kd_hgnetv2_b4_s6 ekd_hgnetv2_b4_s4 ekd_hgnetv2_b4_s5 ekd_hgnetv2_b4_s6 rfekd_hgnetv2_b4_s1"; fi
bash $H/ops/2026-11-23/test_score.sh $runs
echo "FOLLOWON DONE"
