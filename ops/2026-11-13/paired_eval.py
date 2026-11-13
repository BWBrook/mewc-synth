"""Paired comparison of two arms over a set of focus classes (docs/plan-booking-3.md, Q1 and Q4): per-class
F1 on the v2c validation snips for every seed both arms share, focus macro-F1 by arm, and B - A with a
two-level bootstrap (seed pairs resampled, validation snips resampled once per draw and shared by all runs).
A non-focus class is harmed if its F1 difference has an interval entirely below -0.01.
FOCUS is a profile CSV with a "class" column, or a comma-separated list. A and B are run-name templates with
{s} for the seed, for example pd_hgnetv2_b4_anchor_s{s}.
Usage: python ops/2026-11-13/paired_eval.py SCORES_DIR VAL_CSV FOCUS A B OUT_JSON"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def f1_by_class(truth, pred, k):
    tp = np.bincount(truth[pred == truth], minlength=k)
    denom = np.bincount(truth, minlength=k) + np.bincount(pred, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * tp / denom, np.nan)


def main(scores, val_csv, focus, a, b, out, n_boot=1000):
    v = pd.read_csv(val_csv)
    classes = sorted(v.label.unique())
    ci = {c: i for i, c in enumerate(classes)}
    k, truth = len(classes), v.label.map(ci).to_numpy()
    focus = pd.read_csv(focus)["class"].tolist() if focus.endswith(".csv") else focus.split(",")
    fi = [ci[c] for c in focus]
    load = lambda f: pd.read_csv(f).set_index("path").loc[v.path, "top_name"].map(ci).fillna(-1).astype(int).to_numpy()
    seeds = [s for s in range(1, 21) if all((Path(scores) / f"{t.format(s=s)}.csv").exists() for t in (a, b))]
    preds = {(t, s): load(Path(scores) / f"{t.format(s=s)}.csv") for t in (a, b) for s in seeds}
    per = {t: np.array([f1_by_class(truth, preds[(t, s)], k) for s in seeds]) for t in (a, b)}
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(truth), len(truth))
        ss = rng.choice(seeds, len(seeds))
        boot.append(np.nanmean([f1_by_class(truth[idx], preds[(b, s)][idx], k)
                                - f1_by_class(truth[idx], preds[(a, s)][idx], k) for s in ss], axis=0))
    bt = pd.DataFrame(boot, columns=classes)
    fb = bt[focus].mean(axis=1)
    res = {"a": a, "b": b, "seeds": seeds, "n_focus": len(focus),
           "focus_a": round(float(np.nanmean(per[a][:, fi], axis=1).mean()), 4),
           "focus_b": round(float(np.nanmean(per[b][:, fi], axis=1).mean()), 4),
           "focus_a_by_seed": [round(float(x), 4) for x in np.nanmean(per[a][:, fi], axis=1)],
           "focus_b_by_seed": [round(float(x), 4) for x in np.nanmean(per[b][:, fi], axis=1)],
           "overall_a": round(float(np.nanmean(per[a], axis=1).mean()), 4),
           "overall_b": round(float(np.nanmean(per[b], axis=1).mean()), 4)}
    res["focus_diff"] = round(res["focus_b"] - res["focus_a"], 4)
    res["ci95"] = [round(float(q), 4) for q in fb.quantile([0.025, 0.975])]
    res["per_class_diff"] = {c: round(float(np.nanmean(per[b][:, ci[c]] - per[a][:, ci[c]])), 4) for c in focus}
    res["harmed_nonfocus"] = sorted(c for c in classes if c not in focus and bt[c].quantile(0.975) < -0.01)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({x: res[x] for x in res if x not in ("per_class_diff",)}))


if __name__ == "__main__":
    main(*sys.argv[1:7])
