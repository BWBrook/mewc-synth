"""Teacher targets for a Q7 refit (PLAN 2026-11-27): the lane's ensemble teacher on the training rows followed by
its validation rows (the mean of its members' validation probabilities), in arms/train_val.csv's order.
Usage: python ops/2026-11-27/refit_prep.py TRAIN_VAL_CSV TRAIN_CSV VAL_CSV TEACHER_PREFIX"""
import sys

import numpy as np
import pandas as pd


def main(train_val_csv, train_csv, val_csv, prefix):
    tv, tr, va = (pd.read_csv(f).path.to_numpy() for f in (train_val_csv, train_csv, val_csv))
    assert len(tv) == len(tr) + len(va) and (tv[:len(tr)] == tr).all() and (tv[len(tr):] == va).all()
    soft, val = np.load(f"{prefix}_soft.npy"), np.load(f"{prefix}_val.npy")
    assert soft.shape[0] == len(tr) and val.shape[0] == len(va)
    np.save(f"{prefix}_soft_train_val.npy", np.concatenate([soft, val]).astype(np.float16))
    print("saved", f"{prefix}_soft_train_val.npy", len(tv))


if __name__ == "__main__":
    main(*sys.argv[1:])
