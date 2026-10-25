"""Model-size pilot table: for each architecture trained on the R composition, macro-F1 over all test
snips and over camera-trap snips (mean and range over seeds), parameters, and inference throughput on
the A100 (batch 64) and on 8 CPU cores (the longest CPU benchmark available per model).
Usage: python src/pilot_summary.py SCREEN_DIR"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import source_prefix
from screen_eval import prf

MODELS = ["EN0", "EN2", "ENS", "ENM", "CNP", "CNT", "CNS", "CNB", "VTS", "VTB", "VTL"]


def run_names(d, code):
    pat = "R_s*.csv" if code == "VTB" else f"R_{code.lower()}_s*.csv"
    return sorted(p.stem for p in (d / "scores").glob(pat) if not p.stem.endswith("_probs"))


def main(screen_dir):
    d = Path(screen_dir)
    paths = [l.strip() for l in open(d / "shared_test_paths.txt") if l.strip()]
    names = np.array([p.split("/")[-2] for p in paths])
    camera = np.array([source_prefix("/".join(p.split("/")[-2:])) not in ("ebird", "inat") for p in paths])
    classes = sorted(set(names))
    ci = {c: i for i, c in enumerate(classes)}
    y, k = np.array([ci[n] for n in names]), len(classes)
    rows = []
    for code in MODELS:
        runs = run_names(d, code)
        if not runs:
            continue
        f_all, f_ct = [], []
        for r in runs:
            p = np.array([ci[n] for n in pd.read_csv(d / "scores" / f"{r}.csv")["top_name"]])
            for f, m in ((f_all, np.ones(len(y), bool)), (f_ct, camera)):
                f1 = prf(y[m], p[m], k)[2]
                f.append(f1[np.bincount(y[m], minlength=k) > 0].mean())
        stem = "R_s1" if code == "VTB" else runs[0]
        bench = {}
        for dev in ("gpu", "cpu64", "cpu"):
            f = d / "bench" / f"{stem}_{dev}.json"
            if f.exists() and dev.startswith("cpu") and "cpu" in bench:
                continue
            if f.exists():
                bench["cpu" if dev.startswith("cpu") else dev] = json.loads(f.read_text())
        rows.append({"model": code, "seeds": len(runs),
                     "params_M": round(bench.get("gpu", bench.get("cpu", {})).get("params", np.nan) / 1e6, 1),
                     "img_size": bench.get("gpu", {}).get("img_size"),
                     "macro_f1_all": round(float(np.mean(f_all)), 4),
                     "range_all": f"{min(f_all):.3f}-{max(f_all):.3f}",
                     "macro_f1_camera": round(float(np.mean(f_ct)), 4),
                     "gpu_img_s": bench.get("gpu", {}).get("img_per_s"),
                     "cpu8_img_s": bench.get("cpu", {}).get("img_per_s")})
    t = pd.DataFrame(rows).sort_values("macro_f1_all", ascending=False)
    t.to_csv(d / "pilot_summary.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:2])
