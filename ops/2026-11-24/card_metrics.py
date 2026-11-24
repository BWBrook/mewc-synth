"""Model-card figures for a release model, from scores already written (nothing is retrained or tuned on test):
- test and held-out-project macro-F1 on the corrected labels over all snips and over camera-trap snips only
  (eBird and iNaturalist photos excluded, as src/v2_eval.py does), and accuracy;
- per-class F1, precision, recall and support on the sealed test snips;
- a confidence-threshold table: for each threshold on the top probability, the share of snips kept and the
  accuracy of those kept. It is derived on the validation snips from the train-only seeds (mean over seeds),
  then checked on the held-out-project snips with the release model.
Usage: python ops/2026-11-24/card_metrics.py SCORES_DIR V2_DIR RELEASE_RUN TRAINONLY_TEMPLATE SEEDS OUT_JSON
       (TRAINONLY_TEMPLATE has {s} for the seed; SEEDS is comma-separated)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import source_prefix  # noqa: E402

THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def load(scores, prefix, lab, ci):
    sc = pd.read_csv(scores / f"{prefix}.csv")
    probs = np.load(scores / f"{prefix}_probs.npy", mmap_mode="r")
    idx = pd.Series(np.arange(len(sc)), index=sc.path).loc[lab.path].to_numpy()
    p = np.asarray(probs[idx])
    return p.argmax(1), p.max(1)


def per_class(truth, pred, k):
    tp = np.bincount(truth[pred == truth], minlength=k).astype(float)
    sup, npred = np.bincount(truth, minlength=k), np.bincount(pred, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 2 * tp / (sup + npred), tp / npred, tp / sup, sup


def summary(truth, pred, camera, k):
    out = {}
    for name, m in (("all", np.ones_like(camera)), ("camera_trap", camera)):
        f1, _, _, sup = per_class(truth[m], pred[m], k)
        out[name] = {"n": int(m.sum()), "classes": int((sup > 0).sum()), "macro_f1": round(float(np.nanmean(f1[sup > 0])), 4),
                     "accuracy": round(float((truth[m] == pred[m]).mean()), 4)}
    return out


def thresholds(truth, pred, conf):
    return [{"threshold": t, "kept": round(float((conf >= t).mean()), 4),
             "accuracy_kept": round(float((pred[conf >= t] == truth[conf >= t]).mean()), 4)} for t in THRESHOLDS]


def main(scores, v2, release, template, seeds, out):
    scores, v2 = Path(scores), Path(v2)
    classes = sorted((v2 / "classes.txt").read_text().split())
    ci, k = {c: i for i, c in enumerate(classes)}, len(classes)
    parts = {}
    for part, pre in (("val", ""), ("test", "test_"), ("test_proj", "testproj_")):
        lab = pd.read_csv(v2 / f"{part}.csv")
        lab = lab[lab.label.isin(ci)]
        camera = np.array([source_prefix("/".join(p.split("/")[-2:])) not in ("ebird", "inat") for p in lab.path])
        parts[part] = (lab, lab.label.map(ci).to_numpy(), camera, pre)
    res = {"release_run": release, "train_only": [template.format(s=s) for s in seeds.split(",")]}
    for part in ("test", "test_proj"):
        lab, truth, camera, pre = parts[part]
        pred, conf = load(scores, f"{pre}{release}", lab, ci)
        res[part] = summary(truth, pred, camera, k)
        if part == "test":
            f1, prec, rec, sup = per_class(truth, pred, k)
            res["per_class_test"] = [{"class": c, "f1": round(float(f1[i]), 4), "precision": round(float(prec[i]), 4),
                                      "recall": round(float(rec[i]), 4), "support": int(sup[i])} for i, c in enumerate(classes)]
        else:
            res["thresholds_test_proj_release"] = thresholds(truth, pred, conf)
    lab, truth, camera, _ = parts["val"]
    tabs = [thresholds(truth, *load(scores, r, lab, ci)) for r in res["train_only"]]
    res["thresholds_val_train_only"] = [{"threshold": t, "kept": round(float(np.mean([x[i]["kept"] for x in tabs])), 4),
                                         "accuracy_kept": round(float(np.mean([x[i]["accuracy_kept"] for x in tabs])), 4)}
                                        for i, t in enumerate(THRESHOLDS)]
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({x: res[x] for x in res if x != "per_class_test"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:7])
