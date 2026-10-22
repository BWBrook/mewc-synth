#!/usr/bin/env bash
# Train one screening classifier with the patched mewc-train (fix/progressive-training-and-speed) and score
# it on the shared test set with per-image predictions.
# Usage: [MODEL=VTL] src/run_screen.sh ARM SEED [CLASS_SAMPLES FROZ_EPOCH PROG_STAGE_LEN PROG_TOT_EPOCH]
# MODEL defaults to VTB; other models get the model code in the run name (ARM_vtl_sSEED).
set -uo pipefail
arm=$1 seed=$2 ns=${3:-400} froz=${4:-3} stage=${5:-3} tot=${6:-12}
model=${MODEL:-VTB}
M=/mnt/ml-data/mewc-synth; R=$M/screen; T=$HOME/mewc-train/src
name=${arm}_s${seed}; [ "$model" = VTB ] || name=${arm}_${model,,}_s${seed}
mkdir -p $R/logs $R/runs $R/scores
sudo docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 ${XLA_FRACTION:+-e XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_FRACTION} \
  -v $M:/data -v /mnt/scratch/keras-cache:/root/.keras \
  -v $T/lib_model.py:/code/lib_model.py -v $T/lib_data.py:/code/lib_data.py -v $T/mewc_train.py:/code/mewc_train.py \
  -e MODEL=$model -e SEED=$seed -e SAVEFILE=$name -e OUTPUT_PATH=/data/screen/runs \
  -e TRAIN_PATH=/data/screen/$arm/train -e VAL_PATH=/data/screen/shared/val -e TEST_PATH=/data/screen/shared/test_stub \
  -e CLASS_SAMPLES_DEFAULT=$ns -e FROZ_EPOCH=$froz -e PROG_STAGE_LEN=$stage -e PROG_TOT_EPOCH=$tot -e BUF=9 -e BATCH_SIZE=16 \
  zaandahl/mewc-train:2.0.9 > $R/logs/$name.log 2>&1
rc=$?
keras=$R/runs/$name/$model/${name}_$model.keras
[ $rc -eq 0 ] && [ -f "$keras" ] || { echo "train $name FAILED (rc $rc)"; exit 1; }
sudo docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 ${XLA_FRACTION:+-e XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_FRACTION} \
  -v $M:/data -v $HOME/mewc-synth/src/vtl_score.py:/code/vtl_score.py \
  zaandahl/mewc-train:2.0.9 python /code/vtl_score.py /data/screen/runs/$name/$model/${name}_$model.keras /data/screen/classes.yaml \
  /data/screen/shared_test_paths.txt /data/screen/scores/$name 2>&1 | grep "^scored" || { echo "score $name FAILED"; exit 1; }
echo "run $name done"
