#!/usr/bin/env bash
# Model-size pilot (roadmap, 24 October): train on R (full real data) with the screen recipe, two seeds
# per model, then time inference on the A100 (batch 64) and on 8 CPU cores (batch 8). After VTL8.
set -uo pipefail
M=/mnt/ml-data/mewc-synth
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
bench() {
  local name=$1 model=$2 k=/data/screen/runs/$1/$2/$1_$2.keras
  mkdir -p $M/screen/bench
  sudo docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 -v $M:/data -v $HOME/mewc-synth/src/bench_infer.py:/code/bench_infer.py \
    zaandahl/mewc-train:2.0.9 python /code/bench_infer.py $k /data/screen/bench/${name}_gpu.json 64 20 2>&1 | tail -1
  sudo docker run --rm --cpuset-cpus=0-7 -e JAX_PLATFORMS=cpu -e PYTHONDONTWRITEBYTECODE=1 -v $M:/data -v $HOME/mewc-synth/src/bench_infer.py:/code/bench_infer.py \
    zaandahl/mewc-train:2.0.9 python /code/bench_infer.py $k /data/screen/bench/${name}_cpu.json 8 4 2>&1 | tail -1
}
until grep -q "VTL8 g4 DONE" /mnt/scratch/logs/vtl8_g4.log; do sleep 60; done
[ "yes" = yes ] && { stamp "bench VTB"; bench R_s1 VTB; }
for model in EN0 EN2 ENS CNP CNB; do
  for seed in 1 2; do
    stamp "run $model R seed $seed"
    MODEL=$model bash $HOME/mewc-synth/src/run_screen.sh R $seed || { stamp "run $model R seed $seed FAILED"; continue; }
  done
  stamp "bench $model"; bench R_${model,,}_s1 $model
done
stamp "run VTL R seed 1"
MODEL=VTL bash $HOME/mewc-synth/src/run_screen.sh R 1 && { stamp "bench VTL"; bench R_vtl_s1 VTL; } || stamp "run VTL R seed 1 FAILED"
stamp "PILOT g4 DONE"
