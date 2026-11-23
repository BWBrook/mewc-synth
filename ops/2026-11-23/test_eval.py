"""Evaluate the one-time test scoring (docs/plan-booking-3.md, Q5): for each final model and each test part
(the sealed test snips; the held-out-project test snips), macro-F1 over all classes and over the 25 real
low-count classes, and accuracy, on the corrected labels (v2/<part>.csv) and on the original labels
(v2/<part>.pre_lc.csv). Groups: train-only seeds (mean and SD) and the refit on train + validation.
Usage: python ops/2026-11-23/test_eval.py SCORES_DIR V2_DIR PROFILE_CSV RUNS_TXT OUT_JSON"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = {"hgnetv2_b4 train-only": "pd_hgnetv2_b4_anchor_s", "hgnetv2_b4 refit": "rf_hgnetv2_b4_s",
          "eva02_b train-only": "md_eva02_b_anchor_s", "eva02_b refit": "rf_eva02_b_s"}


def f1s(truth, pred, k):
    tp = np.bincount(truth[pred == truth], minlength=k)
    denom = np.bincount(truth, minlength=k) + np.bincount(pred, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(denom > 0, 2 * tp / denom, np.nan)
    return f, np.bincount(truth, minlength=k) > 0


def main(scores, v2, profile, runs_txt, out):
    v2, scores = Path(v2), Path(scores)
    classes = sorted((v2 / "classes.txt").read_text().split())
    ci, k = {c: i for i, c in enumerate(classes)}, len(classes)
    low = [ci[c] for c in pd.read_csv(profile)["class"]]
    runs = Path(runs_txt).read_text().split()
    rows = []
    for part, pre in (("test", "test"), ("test_proj", "testproj")):
        for labels in ("corrected", "original"):
            lab = pd.read_csv(v2 / (f"{part}.csv" if labels == "corrected" else f"{part}.pre_lc.csv"))
            lab = lab[lab.label.isin(ci)]
            truth = lab.label.map(ci).to_numpy()
            for run in runs:
                f = scores / f"{pre}_{run}.csv"
                if not f.exists():
                    continue
                pred = pd.read_csv(f).set_index("path").loc[lab.path, "top_name"].map(ci).fillna(-1).astype(int).to_numpy()
                fc, present = f1s(truth, pred, k)
                rows.append({"part": part, "labels": labels, "run": run, "n": len(truth),
                             "macro_f1": float(np.nanmean(fc[present])),
                             "macro_f1_lowcount": float(np.nanmean(fc[[i for i in low if present[i]]])),
                             "accuracy": float((pred == truth).mean())})
    t = pd.DataFrame(rows)
    t["group"] = t.run.map(lambda r: next(g for g, p in GROUPS.items() if r.startswith(p)))
    summary = (t.groupby(["part", "labels", "group"])[["macro_f1", "macro_f1_lowcount", "accuracy"]]
               .agg(["mean", "std", "count"]).round(4))
    flat = summary.copy()
    flat.columns = [f"{a}_{b}" for a, b in flat.columns]
    json.dump({"runs": t.round(5).to_dict("records"), "summary": flat.reset_index().to_dict("records")},
              open(out, "w"), indent=1, default=float)
    print(summary.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:6])
