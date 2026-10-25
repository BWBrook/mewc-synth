#!/usr/bin/env bash
# Longer CPU benchmark (8 cores, 8 batches of 8) for every seed-1 R model on this machine, after the CNN pilot.
set -uo pipefail
M=/mnt/ml-data/mewc-synth
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
until grep -q "CNN g4 DONE" /mnt/scratch/logs/cnn_g4.log; do sleep 60; done
for k in $M/screen/runs/R_s1/VTB/R_s1_VTB.keras $M/screen/runs/R_*_s1/*/R_*_s1_*.keras; do
  [ -f "$k" ] || continue
  name=$(basename "$k" .keras); name=${name%_*}
  sudo docker run --rm --cpuset-cpus=0-7 -e JAX_PLATFORMS=cpu -e PYTHONDONTWRITEBYTECODE=1 -v $M:/data -v $HOME/mewc-synth/src/bench_infer.py:/code/bench_infer.py \
    zaandahl/mewc-train:2.0.10 python /code/bench_infer.py ${k/$M//data} /data/screen/bench/${name}_cpu64.json 8 8 2>&1 | tail -1
done
stamp "CPUBENCH g4 DONE"
