"""Real-equivalence arms for the pseudo-rare classes (docs/plan-booking-3.md, Q4): A0_100 and A0_200 hold 100
and 200 real training snips per pseudo-rare class, nested supersets of A0's 50 (the extra snips are a seeded
random draw from the rest of the class's training rows); every other class keeps its full pool. Deterministic,
so building on each instance from identical inputs gives identical lists.
Usage: python ops/2026-11-13/realeq_arms.py TRAIN_CSV A0_CSV OUT_DIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PR = ["spotted_tail_quoll", "forester_kangaroo", "black_currawong", "black_rat"]


def main(train_csv, a0_csv, out):
    train, a0 = pd.read_csv(train_csv), pd.read_csv(a0_csv)
    rest = a0[~a0.label.isin(PR)]
    rng = np.random.default_rng(20261113)
    extra = {}
    for c in PR:
        keep = set(a0[a0.label == c].path)
        pool = train[(train.label == c) & ~train.path.isin(keep)].sort_values("path")
        extra[c] = pool.iloc[rng.permutation(len(pool))[:150]]
    for n in (100, 200):
        parts = [rest] + [pd.concat([a0[a0.label == c], extra[c].iloc[:n - 50]]) for c in PR]
        df = pd.concat(parts, ignore_index=True)[["path", "label"]]
        df.to_csv(Path(out) / f"A0_{n}.csv", index=False)
        print(n, df[df.label.isin(PR)].groupby("label").path.nunique().to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:4])
