"""Compare screening arms on the shared real test set. Per run: per-class precision, recall and F1 for
focus classes, macro-F1 over all classes and over non-focus classes (no-harm check), and the share of
each focus class's test snips predicted as named look-alikes. Arm differences use a paired bootstrap
over test snips (seed runs within an arm are averaged per bootstrap draw). Camera-trap-only results
are reported separately.
Usage: python src/screen_eval.py SCREEN_DIR ARMS_JSON
ARMS_JSON example: {"focus": ["black_rat"], "baseline": "A0", "arms": {"A0": ["A0_s1", "A0_s2"], "A1": ["A1_s1"]}}"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import source_prefix
from identity_report import LOOKALIKES

B = 1000


def prf(y, p, k):
    """Per-class precision, recall and F1 from integer-coded labels via one confusion count."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    pred, true = cm.sum(0), cm.sum(1)
    prec = np.divide(tp, pred, out=np.zeros(k), where=pred > 0)
    rec = np.divide(tp, true, out=np.zeros(k), where=true > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return prec, rec, f1, cm


def summarise(y, preds, classes, focus):
    k, ci = len(classes), {c: i for i, c in enumerate(classes)}
    present = np.bincount(y, minlength=k) > 0
    nonfocus = present & ~np.isin(np.arange(k), [ci[c] for c in focus])
    rows = {}
    for run, p in preds.items():
        prec, rec, f1, cm = prf(y, p, k)
        r = {"macro_f1_all": float(f1[present].mean()), "macro_f1_nonfocus": float(f1[nonfocus].mean()),
             "accuracy": float(np.mean(p == y))}
        for c in focus:
            i = ci[c]
            r[f"{c}_P"], r[f"{c}_R"], r[f"{c}_F1"] = float(prec[i]), float(rec[i]), float(f1[i])
            looks = [ci[l] for l in LOOKALIKES.get(c, []) if l in ci]
            r[f"{c}_to_lookalike"] = float(cm[i, looks].sum() / max(cm[i].sum(), 1))
        rows[run] = r
    return pd.DataFrame(rows).T


def main(screen_dir, arms_json):
    d, spec = Path(screen_dir), json.loads(arms_json)
    paths = [l.strip() for l in open(d / "shared_test_paths.txt") if l.strip()]
    names = np.array([p.split("/")[-2] for p in paths])
    camera = np.array([source_prefix("/".join(p.split("/")[-2:])) not in ("ebird", "inat") for p in paths])
    classes = sorted(set(names))
    ci = {c: i for i, c in enumerate(classes)}
    y = np.array([ci[n] for n in names])
    runs = [r for rs in spec["arms"].values() for r in rs]
    preds = {}
    for r in runs:
        sc = pd.read_csv(d / "scores" / f"{r}.csv")
        assert (sc["path"].to_numpy() == np.array(paths)).all(), f"{r}: score rows do not match test paths"
        preds[r] = np.array([ci[n] for n in sc["top_name"]])
    focus = spec["focus"]
    report = {}
    for subset, mask in (("all", np.ones(len(y), bool)), ("camera_trap", camera)):
        tab = summarise(y[mask], {r: p[mask] for r, p in preds.items()}, classes, focus)
        tab.round(4).to_csv(d / f"eval_{subset}.csv")
        report[subset] = {"per_run": tab.round(4).to_dict("index")}
        rng = np.random.default_rng(0)
        idx = np.flatnonzero(mask)
        metrics = ["macro_f1_all", "macro_f1_nonfocus"] + [f"{c}_F1" for c in focus] + [f"{c}_R" for c in focus]
        diffs = {a: {m: [] for m in metrics} for a in spec["arms"] if a != spec["baseline"]}
        for _ in range(B):
            s = rng.choice(idx, len(idx), replace=True)
            tb = summarise(y[s], {r: p[s] for r, p in preds.items()}, classes, focus)
            base = tb.loc[spec["arms"][spec["baseline"]]].mean()
            for a in diffs:
                arm = tb.loc[spec["arms"][a]].mean()
                for m in metrics:
                    diffs[a][m].append(arm[m] - base[m])
        report[subset]["vs_baseline"] = {a: {m: {"mean": round(float(np.mean(v)), 4),
                                                 "ci95": [round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)]}
                                             for m, v in ms.items()} for a, ms in diffs.items()}
    (d / "eval_report.json").write_text(json.dumps(report, indent=1))
    for subset in report:
        print(f"== {subset}")
        print(pd.DataFrame(report[subset]["per_run"]).T.round(3).to_string())
        for a, ms in report[subset]["vs_baseline"].items():
            print(f"-- {a} minus {spec['baseline']} (paired bootstrap, 95% CI)")
            for m, v in ms.items():
                print(f"   {m:28s} {v['mean']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main(*sys.argv[1:3])
