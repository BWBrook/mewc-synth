#!/usr/bin/env bash
# Start the third booking's decision-free work on this instance (docs/plan-booking-3.md). Run detached:
#   setsid nohup bash ops/2026-11-13/launch.sh g4|g5 > /mnt/ml-data/mewc-synth/v2_runs/logs/launch_<gpu>.log 2>&1 &
# Both build the Q4 real-equivalence lists. g4 then runs its queue; g5 runs rare_synth.sh first and, if it
# fails, runs its queue without the Q1 S20 runs.
set -uo pipefail
g=${1:?g4 or g5}; H=$HOME/mewc-synth; M=/mnt/ml-data/mewc-synth; R=$M/v2_runs; P=$H/.venv/bin/python; O=$H/ops/2026-11-13
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { stamp "no GPU"; exit 1; }
$P $O/realeq_arms.py $M/v2/train.csv $R/arms/A0.csv $R/arms || { stamp "realeq arms FAILED"; exit 1; }
md5sum $R/arms/A0_100.csv $R/arms/A0_200.csv
q=$O/$g.queue
if [ "$g" = g5 ]; then
  if ! bash $O/rare_synth.sh > $R/logs/rare_synth.log 2>&1; then
    stamp "rare synth failed; queue without rr_ runs"; grep -v " rr_" $O/g5.queue > $O/g5_norr.queue; q=$O/g5_norr.queue
  fi
fi
stamp "queue $q"
PRUNE=1 exec bash $H/src/queue_v2.sh $q
