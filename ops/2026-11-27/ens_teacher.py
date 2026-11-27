"""Q7 teacher (PLAN 2026-11-27): the mean class probabilities of several train-only runs on the training snips
(SCORES/train_<run>.csv and _probs.npy, checked against TRAIN_CSV's row order) saved as float16 to OUT_PREFIX_soft.npy,
and the mean of their validation probabilities (SCORES/<run>_probs.npy) to OUT_PREFIX_val.npy, with the validation
macro-F1 of each member and of the ensemble.
Usage: python ops/2026-11-27/ens_teacher.py SCORES TRAIN_CSV VAL_CSV CLASSES_TXT OUT_PREFIX RUN..."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def macro_f1(truth, pred, k):
    tp = np.bincount(truth[pred == truth], minlength=k)
    denom = np.bincount(truth, minlength=k) + np.bincount(pred, minlength=k)
    present = np.bincount(truth, minlength=k) > 0
    return float((2 * tp[present] / denom[present]).mean())


def main(scores, train_csv, val_csv, classes_txt, out, *runs):
    scores = Path(scores)
    classes = sorted(Path(classes_txt).read_text().split())
    ci = {c: i for i, c in enumerate(classes)}
    tr = pd.read_csv(train_csv).path.to_numpy()
    acc = None
    for r in runs:
        assert (pd.read_csv(scores / f"train_{r}.csv").path.to_numpy() == tr).all(), r
        a = np.load(scores / f"train_{r}_probs.npy").astype(np.float32)
        acc = a if acc is None else acc + a
    np.save(f"{out}_soft.npy", (acc / len(runs)).astype(np.float16))
    v = pd.read_csv(val_csv)
    truth = v.label.map(ci).to_numpy()
    vacc, res = None, {"members": {}}
    for r in runs:
        s = pd.read_csv(scores / f"{r}.csv")
        idx = pd.Series(np.arange(len(s)), index=s.path).loc[v.path].to_numpy()
        p = np.load(scores / f"{r}_probs.npy")[idx].astype(np.float32)
        res["members"][r] = round(macro_f1(truth, p.argmax(1), len(classes)), 4)
        vacc = p if vacc is None else vacc + p
    np.save(f"{out}_val.npy", (vacc / len(runs)).astype(np.float16))
    res["ensemble_val_macro_f1"] = round(macro_f1(truth, vacc.argmax(1), len(classes)), 4)
    res["train_mean_max_prob"] = round(float((acc / len(runs)).max(1).mean()), 4)
    Path(f"{out}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main(*sys.argv[1:])
