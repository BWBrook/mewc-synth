#!/usr/bin/env bash
# gpu-5, after the ten HGNetV2-B4 reruns: pick HGNetV2-B4's stage-2 runs from the full stage 1, then run them
# followed by gpu-5's share of EVA-02-B stage 2 (tune_eva_s2_g5.queue), in one queue so one CUDA process runs.
set -uo pipefail
O=$HOME/mewc-synth/ops; R=/mnt/ml-data/mewc-synth/v2_runs
$HOME/mewc-synth/.venv/bin/python $O/2026-11-04/stage2.py $R/registry.jsonl hgnetv2_b4 $O/2026-11-03/tune > $O/2026-11-04/tune_s2_g5.queue
cat $O/2026-11-04/tune_eva_s2_g5.queue >> $O/2026-11-04/tune_s2_g5.queue
PRUNE=1 exec bash $HOME/mewc-synth/src/queue_v2.sh $O/2026-11-04/tune_s2_g5.queue
