"""Evaluate the phase 1 synthetic re-test (PLAN 2026-11-01): per-class F1 on the v2c validation snips for every
run, the focus endpoint (macro-F1 over the four pseudo-rare classes) by arm, and paired differences against A0
with two-level bootstrap intervals (resample seeds, then validation snips within each resampled seed pair).
No-harm: a non-focus class is harmed if its F1 difference has a two-level interval entirely below -0.01.
Usage: python ops/2026-11-01/retest_eval.py RUNS_DIR V2_DIR OUT_JSON"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FOCUS = ["spotted_tail_quoll", "forester_kangaroo", "black_currawong", "black_rat"]


def f1_by_class(truth, pred, k):
    tp = np.bincount(truth[pred == truth], minlength=k)
    denom = np.bincount(truth, minlength=k) + np.bincount(pred, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * tp / denom, np.nan)


def main(runs_dir, v2, out):
    d, v = Path(runs_dir), pd.read_csv(Path(v2) / "val.csv")
    classes = sorted(v.label.unique()); ci = {c: i for i, c in enumerate(classes)}; k = len(classes)
    fi = [ci[c] for c in FOCUS]
    preds = {}
    for f in sorted((d / "scores").glob("rt_*.csv")):
        if f.name.endswith("_probs.csv"):
            continue
        s = pd.read_csv(f).merge(v, on="path")
        preds[f.stem] = s.set_index("path").loc[v.path, "top_name"].map(ci).to_numpy()
    truth = v.label.map(ci).to_numpy()
    rows = [{"run": r, "arm": r.split("_")[1], "seed": int(r.split("_s")[-1]), **dict(zip(classes, f1_by_class(truth, p, k)))}
            for r, p in preds.items()]
    t = pd.DataFrame(rows)
    t["focus"] = t[FOCUS].mean(axis=1)
    arm_means = t.groupby("arm")[FOCUS + ["focus"]].mean().round(4)
    rng = np.random.default_rng(0)
    res = {"per_class_by_arm": arm_means.to_dict("index"), "runs": t.round(4).to_dict("records")}
    for arm in ["S20", "S20f"]:
        seeds = sorted(set(t[t.arm == arm].seed) & set(t[t.arm == "A0"].seed))
        diff = t.set_index(["arm", "seed"])
        point = float(np.mean([diff.loc[(arm, s), "focus"] - diff.loc[("A0", s), "focus"] for s in seeds]))
        boots, harm_boot = [], []
        for _ in range(1000):
            ss = rng.choice(seeds, len(seeds))
            vals = []
            for s in ss:
                idx = rng.choice(len(truth), len(truth))
                vals.append(f1_by_class(truth[idx], preds[f"rt_{arm}_s{s}"][idx], k) - f1_by_class(truth[idx], preds[f"rt_A0_s{s}"][idx], k))
            m = np.nanmean(vals, axis=0)
            boots.append(m[fi].mean())
            harm_boot.append(m)
        hb = pd.DataFrame(harm_boot, columns=classes)
        upper = hb.quantile(0.975)
        harmed = sorted(c for c in classes if c not in FOCUS and upper[c] < -0.01)
        res[f"{arm}_vs_A0"] = {"seeds": len(seeds), "focus_diff": round(point, 4),
                               "ci95": [round(float(np.quantile(boots, 0.025)), 4), round(float(np.quantile(boots, 0.975)), 4)],
                               "per_class_diff": {c: round(float(hb[c].mean()), 4) for c in FOCUS},
                               "harmed_nonfocus": harmed}
    json.dump(res, open(out, "w"), indent=1)
    print(arm_means.to_string())
    for arm in ["S20", "S20f"]:
        print(arm, json.dumps(res[f"{arm}_vs_A0"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
