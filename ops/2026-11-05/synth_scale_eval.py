"""Evaluate the synthetic scale test (PLAN 2026-11-05): focus macro-F1 (the four pseudo-rare classes) on the v2c
validation snips for every ss_* run, the paired difference S20 - A0 at each scale, and the scale interaction
(medium gain - screen gain). Two-level bootstrap: validation snips resampled once per draw and shared by all
runs, seeds resampled within each scale. No-harm as in the re-test: a non-focus class is harmed at a scale if
its F1 difference has an interval entirely below -0.01.
Usage: python ops/2026-11-05/synth_scale_eval.py SCORES_DIR VAL_CSV OUT_JSON"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FOCUS = ["spotted_tail_quoll", "forester_kangaroo", "black_currawong", "black_rat"]
SCALES = ("screen", "medium")


def f1_by_class(truth, pred, k):
    tp = np.bincount(truth[pred == truth], minlength=k)
    denom = np.bincount(truth, minlength=k) + np.bincount(pred, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * tp / denom, np.nan)


def main(scores, val_csv, out, n_boot=1000):
    v = pd.read_csv(val_csv)
    classes = sorted(v.label.unique())
    ci = {c: i for i, c in enumerate(classes)}
    k, fi = len(classes), [ci[c] for c in FOCUS]
    truth = v.label.map(ci).to_numpy()
    preds = {}
    for f in sorted(Path(scores).glob("ss_*_s*.csv")):
        p = pd.read_csv(f).set_index("path").loc[v.path, "top_name"]
        preds[f.stem] = p.map(ci).fillna(-1).astype(int).to_numpy()
    seeds = {sc: sorted(int(r.split("_s")[-1]) for r in preds if r.startswith(f"ss_{sc}_A0_")
                        and f"ss_{sc}_S20_s{r.split('_s')[-1]}" in preds) for sc in SCALES}
    pair = lambda sc, s, idx: (f1_by_class(truth[idx], preds[f"ss_{sc}_S20_s{s}"][idx], k)
                               - f1_by_class(truth[idx], preds[f"ss_{sc}_A0_s{s}"][idx], k))
    full = np.arange(len(truth))
    res = {"seeds": seeds}
    for sc in SCALES:
        arms = {a: [f1_by_class(truth, preds[f"ss_{sc}_{a}_s{s}"], k)[fi].mean() for s in seeds[sc]] for a in ("A0", "S20")}
        res[sc] = {"focus_A0": round(float(np.mean(arms["A0"])), 4), "focus_S20": round(float(np.mean(arms["S20"])), 4),
                   "focus_diff": round(float(np.mean([pair(sc, s, full)[fi].mean() for s in seeds[sc]])), 4)}
    rng = np.random.default_rng(0)
    boot = {sc: [] for sc in SCALES}
    for _ in range(n_boot):
        idx = rng.integers(0, len(truth), len(truth))
        for sc in SCALES:
            if seeds[sc]:
                boot[sc].append(np.nanmean([pair(sc, s, idx) for s in rng.choice(seeds[sc], len(seeds[sc]))], axis=0))
    for sc in SCALES:
        if not boot[sc]:
            continue
        b = pd.DataFrame(boot[sc], columns=classes)
        focus = b[FOCUS].mean(axis=1)
        res[sc].update({"ci95": [round(float(q), 4) for q in focus.quantile([0.025, 0.975])],
                        "per_class_diff": {c: round(float(b[c].mean()), 4) for c in FOCUS},
                        "harmed_nonfocus": sorted(c for c in classes if c not in FOCUS and b[c].quantile(0.975) < -0.01)})
    if all(boot[sc] for sc in SCALES):
        inter = (pd.DataFrame(boot["medium"], columns=classes)[FOCUS].mean(axis=1)
                 - pd.DataFrame(boot["screen"], columns=classes)[FOCUS].mean(axis=1))
        res["interaction"] = {"medium_minus_screen": round(res["medium"]["focus_diff"] - res["screen"]["focus_diff"], 4),
                              "ci95": [round(float(q), 4) for q in inter.quantile([0.025, 0.975])]}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
