"""Compare screening arms on the shared real test set. Per run: per-class precision, recall and F1 for
focus classes, macro-F1 over all classes and over non-focus classes (no-harm check), and the share of
each focus class's test snips predicted as named look-alikes. Arm differences use a paired bootstrap
over test snips with seed runs averaged per draw (ci95, conditional on the trained models), and a
two-level bootstrap that also resamples seed runs within each arm (ci95_seeds). The no-harm check
lists non-focus classes whose seed-averaged F1 falls by more than 1 point against the baseline.
Camera-trap-only results are reported separately.
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
             "macro_f1_focus": float(f1[[ci[c] for c in focus if present[ci[c]]]].mean()),
             "accuracy": float(np.mean(p == y))}
        for c in focus:
            i = ci[c]
            r[f"{c}_P"], r[f"{c}_R"], r[f"{c}_F1"] = float(prec[i]), float(rec[i]), float(f1[i])
            looks = [ci[l] for l in LOOKALIKES.get(c, []) if l in ci]
            r[f"{c}_to_lookalike"] = float(cm[i, looks].sum() / max(cm[i].sum(), 1))
        rows[run] = r
    return pd.DataFrame(rows).T


def noharm(y, preds, classes, focus, arms, baseline):
    k = len(classes)
    present = np.flatnonzero(np.bincount(y, minlength=k) > 0)
    f1 = {r: prf(y, p, k)[2] for r, p in preds.items()}
    base = np.stack([f1[r] for r in arms[baseline]])
    out = {}
    for a, runs in arms.items():
        if a == baseline:
            continue
        d = np.stack([f1[r] for r in runs]).mean(0) - base.mean(0)
        rows = [(classes[i], round(float(d[i]), 4), round(float(base[:, i].std()), 4))
                for i in present if classes[i] not in focus and d[i] < -0.01]
        out[a] = {"nonfocus_classes_down_1pt": len(rows),
                  "worst": sorted(rows, key=lambda r: r[1])[:8]}
    return out


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
        report[subset]["noharm"] = noharm(y[mask], {r: p[mask] for r, p in preds.items()}, classes, focus,
                                          spec["arms"], spec["baseline"])
        rng, rng_seed = np.random.default_rng(0), np.random.default_rng(1)
        idx = np.flatnonzero(mask)
        metrics = ["macro_f1_all", "macro_f1_nonfocus", "macro_f1_focus"] + [f"{c}_F1" for c in focus] + [f"{c}_R" for c in focus]
        diffs = {a: {m: [] for m in metrics} for a in spec["arms"] if a != spec["baseline"]}
        diffs2 = {a: {m: [] for m in metrics} for a in diffs}
        pick = lambda runs: list(rng_seed.choice(runs, len(runs), replace=True))
        for _ in range(B):
            s = rng.choice(idx, len(idx), replace=True)
            tb = summarise(y[s], {r: p[s] for r, p in preds.items()}, classes, focus)
            base_runs = spec["arms"][spec["baseline"]]
            base, base2 = tb.loc[base_runs].mean(), tb.loc[pick(base_runs)].mean()
            for a in diffs:
                arm, arm2 = tb.loc[spec["arms"][a]].mean(), tb.loc[pick(spec["arms"][a])].mean()
                for m in metrics:
                    diffs[a][m].append(arm[m] - base[m])
                    diffs2[a][m].append(arm2[m] - base2[m])
        q = lambda v: [round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)]
        report[subset]["vs_baseline"] = {a: {m: {"mean": round(float(np.mean(v)), 4), "ci95": q(v),
                                                 "ci95_seeds": q(diffs2[a][m])}
                                             for m, v in ms.items()} for a, ms in diffs.items()}
    (d / "eval_report.json").write_text(json.dumps(report, indent=1))
    for subset in report:
        print(f"== {subset}")
        print(pd.DataFrame(report[subset]["per_run"]).T.round(3).to_string())
        for a, ms in report[subset]["vs_baseline"].items():
            print(f"-- {a} minus {spec['baseline']} (95% CI: test snips | test snips and seeds)")
            for m, v in ms.items():
                print(f"   {m:28s} {v['mean']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
                      f"  [{v['ci95_seeds'][0]:+.4f}, {v['ci95_seeds'][1]:+.4f}]")
        for a, nh in report[subset]["noharm"].items():
            print(f"-- no-harm, {a}: {nh['nonfocus_classes_down_1pt']} non-focus classes down >1 pt F1; "
                  f"worst (class, diff, baseline seed SD): {nh['worst']}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
