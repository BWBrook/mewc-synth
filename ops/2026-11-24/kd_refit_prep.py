"""Teacher targets for the distilled refit (PLAN 2026-11-24): one row per row of arms/train_val.csv. Training rows
take the Q6 targets (teacher_eva02_b_soft.npy, aligned to v2/train.csv); validation rows take the mean of the
three train-only EVA-02-B medium runs' validation probabilities, the same teacher, out of sample. Checks every
path order and saves float16.
Usage: python ops/2026-11-24/kd_refit_prep.py TRAIN_VAL_CSV TRAIN_CSV TRAIN_SOFT VAL_CSV OUT_NPY VAL_SCORE_PREFIX..."""
import sys

import numpy as np
import pandas as pd


def main(train_val_csv, train_csv, train_soft, val_csv, out, *prefixes):
    tv = pd.read_csv(train_val_csv).path.to_numpy()
    tr = pd.read_csv(train_csv).path.to_numpy()
    va = pd.read_csv(val_csv).path.to_numpy()
    assert (tv[:len(tr)] == tr).all() and (tv[len(tr):] == va).all() and len(tv) == len(tr) + len(va)
    acc = None
    for p in prefixes:
        assert (pd.read_csv(f"{p}.csv").path.to_numpy() == va).all(), p
        a = np.load(f"{p}_probs.npy").astype(np.float32)
        acc = a if acc is None else acc + a
    soft_tr = np.load(train_soft)
    assert soft_tr.shape == (len(tr), acc.shape[1])
    soft = np.concatenate([soft_tr, (acc / len(prefixes)).astype(np.float16)])
    np.save(out, soft)
    print(soft.shape, "mean max prob: train", float(soft_tr.max(1).mean()), "val", float(soft[len(tr):].max(1).mean()))


if __name__ == "__main__":
    main(*sys.argv[1:])
