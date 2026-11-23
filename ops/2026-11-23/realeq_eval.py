"""Q4 of the third booking (docs/plan-booking-3.md), descriptive: focus macro-F1 (the four pseudo-rare classes) on
the v2c validation snips for HGNetV2-B4 at production, by arm and seed: real-only with 50 (A0), 100 and 200 real
snips per class, and the full pool (R); and S20 (50 real plus 200 translations). S20's real-snip equivalent is
read off the real-only curve by linear interpolation in log2(real snips), with a bootstrap over seeds (paired
within seed) for its interval. R uses the full pool, not part of the interpolation.
Usage: python ops/2026-11-23/realeq_eval.py SCORES_DIR VAL_CSV OUT_JSON"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FOCUS = ["spotted_tail_quoll", "forester_kangaroo", "black_currawong", "black_rat"]
ARMS = {"A0_50": ("ss_prod_A0_s{s}", 50), "A0_100": ("re_A0_100_s{s}", 100), "A0_200": ("re_A0_200_s{s}", 200),
        "S20": ("ss_prod_S20_s{s}", None), "R": ("pd_hgnetv2_b4_anchor_s{s}", None)}
SEEDS = range(1, 7)


def f1_focus(truth, pred, k, fi):
    tp = np.bincount(truth[pred == truth], minlength=k)
    denom = np.bincount(truth, minlength=k) + np.bincount(pred, minlength=k)
    return float(np.mean(2 * tp[fi] / denom[fi]))


def equivalent(curve, y):
    xs, ys = np.log2([50, 100, 200]), np.array(curve)
    if not np.all(np.diff(ys) > 0):
        return float("nan")
    return float(2 ** np.interp(y, ys, xs, left=np.nan, right=np.nan))


def main(scores, val_csv, out, n_boot=2000):
    v = pd.read_csv(val_csv)
    ci = {c: i for i, c in enumerate(sorted(v.label.unique()))}
    k, truth, fi = len(ci), v.label.map(ci).to_numpy(), [ci[c] for c in FOCUS]
    f = {a: np.array([f1_focus(truth, pd.read_csv(Path(scores) / f"{t.format(s=s)}.csv").set_index("path")
                               .loc[v.path, "top_name"].map(ci).fillna(-1).astype(int).to_numpy(), k, fi)
                      for s in SEEDS]) for a, (t, _) in ARMS.items()}
    mean = {a: float(x.mean()) for a, x in f.items()}
    eq = equivalent([mean["A0_50"], mean["A0_100"], mean["A0_200"]], mean["S20"])
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        ss = rng.integers(0, len(SEEDS), len(SEEDS))
        boots.append(equivalent([f[a][ss].mean() for a in ("A0_50", "A0_100", "A0_200")], f["S20"][ss].mean()))
    b = np.array(boots)
    res = {"focus_by_arm": {a: {"mean": round(mean[a], 4), "sd": round(float(f[a].std(ddof=1)), 4),
                                "by_seed": [round(float(x), 4) for x in f[a]]} for a in ARMS},
           "s20_real_equivalent": round(eq, 1),
           "ci95": [round(float(q), 1) for q in np.nanquantile(b, [0.025, 0.975])],
           "boot_beyond_200": round(float(np.mean(np.isnan(b))), 3)}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
