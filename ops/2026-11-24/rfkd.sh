#!/usr/bin/env bash
# ngx-gpu-4, after kd.queue (PLAN 2026-11-24): the distilled refit (train + validation), then the one-time test
# scoring of the four train-only distilled runs and the refit, then the remaining four-seed shortlist runs,
# which were claimed out of kd.queue so the refit goes first.
# Usage: wait_then.sh pid:KD_QUEUE_PID -- bash ops/2026-11-24/rfkd.sh
set -uo pipefail
H=$HOME/mewc-synth; R=/mnt/ml-data/mewc-synth/v2_runs
PRUNE=1 bash $H/src/queue_v2.sh $H/ops/2026-11-24/rfkd.queue
bash $H/ops/2026-11-23/test_score.sh kd_hgnetv2_b4_s1 kd_hgnetv2_b4_s2 kd_hgnetv2_b4_s3 kd_hgnetv2_b4_s4 rfkd_hgnetv2_b4_s1
rmdir $R/claims/md_eva02_l_s3 $R/claims/md_dinov3_l_s3 $R/claims/md_eva02_b_anchor_s4
PRUNE=1 exec bash $H/src/queue_v2.sh $H/ops/2026-11-24/rest.queue
