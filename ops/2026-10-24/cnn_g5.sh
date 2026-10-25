#!/usr/bin/env bash
# Model-size pilot, convolutional models with BUF=2 (mewc-train's suggestion for EN/CN), 2026-10-25.
# Seed 1 for every model, then seed 2; a run starts only if its estimated duration ends before 11:00 UTC.
set -uo pipefail
M=/mnt/ml-data/mewc-synth; DEADLINE=$(date -u -d "today 11:00" +%s)
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
bench() {
  local name=$1 model=$2 k=/data/screen/runs/$1/$2/$1_$2.keras
  sudo docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 -v $M:/data -v $HOME/mewc-synth/src/bench_infer.py:/code/bench_infer.py \
    zaandahl/mewc-train:2.0.10 python /code/bench_infer.py $k /data/screen/bench/${name}_gpu.json 64 20 2>&1 | tail -1
  sudo docker run --rm --cpuset-cpus=0-7 -e JAX_PLATFORMS=cpu -e PYTHONDONTWRITEBYTECODE=1 -v $M:/data -v $HOME/mewc-synth/src/bench_infer.py:/code/bench_infer.py \
    zaandahl/mewc-train:2.0.10 python /code/bench_infer.py $k /data/screen/bench/${name}_cpu.json 8 4 2>&1 | tail -1
}
for seed in 1 2; do
  for spec in CNP:54 CNT:90 CNS:114 CNB:132; do
    model=${spec%%:*} mins=${spec##*:}
    if [ $(( $(date +%s) + mins * 60 )) -gt $DEADLINE ]; then stamp "skip $model seed $seed (would end after 11:00 UTC)"; continue; fi
    stamp "run $model R seed $seed"
    BUF=2 MODEL=$model bash $HOME/mewc-synth/src/run_screen.sh R $seed || { stamp "run $model R seed $seed FAILED"; continue; }
    [ $seed = 1 ] && { stamp "bench $model"; bench R_${model,,}_s1 $model; }
  done
done
stamp "CNN g5 DONE"
