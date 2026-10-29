#!/usr/bin/env bash
# One phase 1 run on split v2: train with mewc-torch (a JSON config) or mewc-train (Docker), score the full
# validation list, and append the run's validation metrics to the registry (src/v2_eval.py).
# Usage: src/run_v2.sh torch NAME CONFIG_JSON
#        src/run_v2.sh mewc NAME MODEL SEED BUF [CLASS_SAMPLES FROZ_EPOCH PROG_STAGE_LEN PROG_TOT_EPOCH BATCH]
# XLA_FRACTION (for example 0.45) caps JAX's GPU memory so that a second job can share the GPU.
set -uo pipefail
stack=$1 name=$2
M=/mnt/ml-data/mewc-synth; V=$M/v2; R=$M/v2_runs; P=$HOME/mewc-synth/.venv/bin/python; S=$HOME/mewc-synth/src
mkdir -p $R/logs $R/runs $R/scores
[ -f $R/classes.txt ] || cp $V/classes.txt $R/classes.txt
[ -f $R/classes.yaml ] || sed 's/.*/&: &/' $R/classes.txt > $R/classes.yaml
[ -f $V/val_paths.txt ] || tail -n +2 $V/val.csv | cut -d, -f1 > $V/val_paths.txt
t0=$(date +%s)
if [ "$stack" = torch ]; then
  cfg=$3
  python3 - "$cfg" "$R/runs/$name" "$V" > $R/runs/$name.json <<'EOF'
import json, sys
c = json.load(open(sys.argv[1])); v = sys.argv[3]
c.setdefault("classes", f"{v}/classes.txt"); c.setdefault("train", f"{v}/train.csv"); c.setdefault("val", f"{v}/val.csv")
c["out"] = sys.argv[2]
print(json.dumps(c))
EOF
  (cd $S && $P -m mewc_torch.train $R/runs/$name.json) > $R/logs/$name.log 2>&1 || { echo "train $name FAILED"; exit 1; }
  t1=$(date +%s)
  (cd $S && $P -m mewc_torch.predict $R/runs/$name $V/val_paths.txt $R/scores/$name 256) >> $R/logs/$name.log 2>&1 || { echo "score $name FAILED"; exit 1; }
  extra=$(python3 -c "import json; c=json.load(open('$R/runs/$name.json')); print(json.dumps({'stack': 'torch', 'backbone': c['backbone'], 'seed': c.get('seed'), 'train_s': $t1 - $t0}))")
else
  model=$3 seed=$4 buf=$5 ns=${6:-400} froz=${7:-3} stage=${8:-3} tot=${9:-12} bs=${10:-16}
  # the same 100 validation snips per class (first in val.csv) that mewc-torch validates on each epoch
  [ -d $V/val_100 ] || python3 - $V <<'EOF'
import csv, os, sys
v, n = sys.argv[1], {}
for r in csv.DictReader(open(f"{v}/val.csv")):
    c, f = r["label"], os.path.basename(r["path"])
    n[c] = n.get(c, 0) + 1
    for d in ["val_100"] * (n[c] <= 100) + ["test_stub"] * (n[c] == 1):
        os.makedirs(f"{v}/{d}/{c}", exist_ok=True)
        os.symlink(f"/data/v2/val/{c}/{f}", f"{v}/{d}/{c}/{f}")
EOF
  sudo docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 ${XLA_FRACTION:+-e XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_FRACTION} \
    -v $M:/data -v /mnt/scratch/keras-cache:/root/.keras \
    zaandahl/mewc-train:2.0.10 bash -c "cd /code && MODEL=$model SEED=$seed SAVEFILE=$name OUTPUT_PATH=/data/v2_runs/runs \
    TRAIN_PATH=/data/v2/train VAL_PATH=/data/v2/val_100 TEST_PATH=/data/v2/test_stub CLASS_SAMPLES_DEFAULT=$ns \
    FROZ_EPOCH=$froz PROG_STAGE_LEN=$stage PROG_TOT_EPOCH=$tot BUF=$buf BATCH_SIZE=$bs python mewc_train.py" \
    > $R/logs/$name.log 2>&1
  keras=$R/runs/$name/$model/${name}_$model.keras
  [ -f "$keras" ] || { echo "train $name FAILED"; exit 1; }
  t1=$(date +%s)
  sed "s#^$M#/data#" $V/val_paths.txt > $V/val_paths_container.txt
  sudo docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 ${XLA_FRACTION:+-e XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_FRACTION} \
    -v $M:/data -v $S/vtl_score.py:/code/vtl_score.py \
    zaandahl/mewc-train:2.0.10 python /code/vtl_score.py /data/v2_runs/runs/$name/$model/${name}_$model.keras \
    /data/v2_runs/classes.yaml /data/v2/val_paths_container.txt /data/v2_runs/scores/$name >> $R/logs/$name.log 2>&1 \
    || { echo "score $name FAILED"; exit 1; }
  sed -i "s#^/data/#$M/#" $R/scores/$name.csv
  extra="{\"stack\": \"mewc\", \"backbone\": \"$model\", \"seed\": $seed, \"buf\": $buf, \"train_s\": $((t1 - t0))}"
fi
(cd $S && $P v2_eval.py $R $name $M/work/snips_meta/class_table.csv "$extra") || { echo "eval $name FAILED"; exit 1; }
echo "run $name done"
