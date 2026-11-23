"""Q6 teacher targets (docs/plan-booking-3.md): the mean of the three train-only EVA-02-B medium runs' class
probabilities on every training snip (scored without augmentation by mewc_torch.predict), aligned to
v2/train.csv's row order and saved as float16. Checks each score file's path order against the list.
Usage: python ops/2026-11-23/kd_prep.py TRAIN_CSV OUT_NPY SCORE_PREFIX..."""
import sys

import numpy as np
import pandas as pd


def main(train_csv, out, *prefixes):
    paths = pd.read_csv(train_csv).path.to_numpy()
    acc = None
    for p in prefixes:
        assert (pd.read_csv(f"{p}.csv").path.to_numpy() == paths).all(), p
        a = np.load(f"{p}_probs.npy").astype(np.float32)
        acc = a if acc is None else acc + a
    soft = (acc / len(prefixes)).astype(np.float16)
    np.save(out, soft)
    print(soft.shape, "mean max prob", float(soft.max(1).mean()))


if __name__ == "__main__":
    main(*sys.argv[1:])
