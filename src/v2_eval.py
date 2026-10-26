"""Validation metrics for one phase 1 run and a registry row: macro-F1 over all classes, camera-trap snips
and the 25 low-count classes, accuracy and expected calibration error (15 bins), from the run's scores
on the full v2 validation list. Appends a JSON line to the registry.
Usage: python src/v2_eval.py RUNS_DIR RUN_NAME CLASS_TABLE_CSV [EXTRA_JSON]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import source_prefix
from screen_eval import prf


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(correct[k == b].mean() - conf[k == b].mean()) * (k == b).mean()
                     for b in range(bins) if (k == b).any()))


def main(runs_dir, name, class_table, extra="{}"):
    d = Path(runs_dir)
    sc = pd.read_csv(d / "scores" / f"{name}.csv")
    classes = sorted(Path(d / "classes.txt").read_text().split())
    ci = {c: i for i, c in enumerate(classes)}
    truth = np.array([ci[p.split("/")[-2]] for p in sc["path"]])
    pred = np.array([ci[n] for n in sc["top_name"]])
    camera = np.array([source_prefix("/".join(p.split("/")[-2:])) not in ("ebird", "inat") for p in sc["path"]])
    ct = pd.read_csv(class_table)
    rare = np.isin(np.arange(len(classes)), [ci[c] for c in ct[ct["tier"].isin(["need_more", "too_few"])]["class"] if c in ci])
    k = len(classes)

    def macro(mask, which=None):
        f1 = prf(truth[mask], pred[mask], k)[2]
        present = np.bincount(truth[mask], minlength=k) > 0
        sel = present if which is None else present & which
        return round(float(f1[sel].mean()), 4)

    allm = np.ones(len(truth), bool)
    row = {"name": name, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_val": len(truth),
           "macro_f1": macro(allm), "macro_f1_camera": macro(camera), "macro_f1_rare": macro(allm, rare),
           "accuracy": round(float((pred == truth).mean()), 4),
           "ece": round(ece(sc["top_prob"].to_numpy(), pred == truth), 4), **json.loads(extra)}
    with open(d / "registry.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


if __name__ == "__main__":
    main(*sys.argv[1:5])
