"""Recipe-tuning trials for the shortlisted models (PLAN 2026-11-03): 40 scrambled-Sobol points per model over
eight recipe dimensions at fixed compute (screen scale, 400 per class, 3 frozen + 12 progressive epochs), seed 1.
The dropout and RandAugment ramps keep the MEWC shape (four stages rising linearly to a maximum) and tune the
maximum. Writes one config per trial, the trial table and a queue per GPU.
Usage: python ops/2026-11-03/tune_build.py OUT_DIR"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

MODELS = {
    "eva02_b": {"backbone": "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", "img_size": 392, "blocks": (3, 12)},
    "hgnetv2_b4": {"backbone": "hgnetv2_b4.ssld_stage2_ft_in1k", "img_size": 288, "blocks": (1, 4)},
}
BASE = {"samples_per_class": 400, "frozen_epochs": 3, "stage_len": 3, "total_epochs": 12, "workers": 12,
        "precision": "bf16", "batch": 64, "warmup_steps": 300, "seed": 1}
N = 40


def recipe(u, blocks):
    lo, hi = blocks
    d, m = 0.05 + 0.55 * u[4], 0.2 + 0.8 * u[5]
    return {"lr": float(10 ** (np.log10(2e-5) + u[0] * np.log10(6e-4 / 2e-5))),
            "weight_decay": float(10 ** (-5 + 4 * u[1])),
            "blocks": int(lo + min(int(u[2] * (hi - lo + 1)), hi - lo)),
            "min_lr_frac": float(10 ** (np.log10(0.02) + u[3] * np.log10(1 / 0.02))),
            "dropouts": [round(d * k / 4, 4) for k in (1, 2, 3, 4)],
            "magnitudes": [round(m * k / 4, 4) for k in (1, 2, 3, 4)],
            "n_aug": int(1 + min(int(u[6] * 4), 3)),
            "focal_gamma": float(3 * u[7])}


def main(out):
    out = Path(out)
    (out / "tune").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (name, spec) in enumerate(MODELS.items()):
        u = qmc.Sobol(d=8, scramble=True, seed=20261103 + i).random_base2(6)[:N]
        for t in range(N):
            r = recipe(u[t], spec["blocks"])
            run = f"tn_{name}_t{t:02d}_s1"
            cfg = {"backbone": spec["backbone"], "img_size": spec["img_size"], **BASE, **r}
            json.dump(cfg, open(out / "tune" / f"{run}.json", "w"), indent=1)
            rows.append({"run": run, "model": name, "trial": t, **{k: v for k, v in r.items() if k not in ("dropouts", "magnitudes")},
                         "dropout_max": r["dropouts"][-1], "magnitude_max": r["magnitudes"][-1]})
    t = pd.DataFrame(rows)
    t.to_csv(out / "tune_trials.csv", index=False)
    line = lambda r: f"torch {r} /home/ubuntu/mewc-synth/ops/2026-11-03/tune/{r}.json"
    eva, hg = list(t[t.model == "eva02_b"].run), list(t[t.model == "hgnetv2_b4"].run)
    g4 = eva[:25]
    g5 = []
    for j, e in enumerate(eva[25:]):
        g5 += hg[3 * j:3 * j + 3] + [e]
    g5 += hg[3 * len(eva[25:]):]
    for q, runs in [("tune_g4.queue", g4), ("tune_g5.queue", g5)]:
        (out / q).write_text("".join(line(r) + "\n" for r in runs))
    print(t.groupby("model")[["lr", "weight_decay", "blocks", "min_lr_frac", "dropout_max", "magnitude_max", "n_aug",
                               "focal_gamma"]].describe().T.round(4).to_string())
    print("g4", len(g4), "g5", len(g5), "total", len(set(g4 + g5)))


if __name__ == "__main__":
    main(sys.argv[1])
