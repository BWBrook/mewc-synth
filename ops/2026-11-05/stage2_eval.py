"""Recipe tuning stage 2 (PLAN 2026-11-03): for each model, the top 8 stage-1 trials at seeds 1 to 3 and the
default-recipe anchor (shortlist runs on the corrected pool: EVA-02-B seeds 2 to 4, HGNetV2-B4 seeds 1 to 3).
Overall macro-F1 on the v2c validation snips, three-seed means, and each recipe's difference from the anchor and
from the model's leader, with a two-level bootstrap interval (seeds resampled within each recipe, snips resampled
jointly). The best recipe is the highest three-seed mean; it and the runner-up go to medium-scale confirmation.
Usage: python ops/2026-11-05/stage2_eval.py SCORES_DIR VAL_CSV TRIALS_CSV OUT_JSON"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ANCHOR = {"eva02_b": ("sl_eva02_b_lr1", (2, 3, 4)), "hgnetv2_b4": ("sl_hgnetv2_b4_lr1", (1, 2, 3))}


def macro_f1(truth, pred, k):
    tp = np.bincount(truth[pred == truth], minlength=k)
    denom = np.bincount(truth, minlength=k) + np.bincount(pred, minlength=k)
    present = np.bincount(truth, minlength=k) > 0
    return float((2 * tp[present] / denom[present]).mean())


def load(f, paths, ci):
    return pd.read_csv(f).set_index("path").loc[paths, "top_name"].map(ci).fillna(-1).astype(int).to_numpy()


def model_eval(model, scores, v, ci, truth, trials, rng, n_boot=1000):
    k = len(ci)
    recipes = {}
    for t in trials[trials.model == model].trial:
        files = [scores / f"tn_{model}_t{t:02d}_s{s}.csv" for s in (1, 2, 3)]
        if all(f.exists() for f in files):
            recipes[f"t{t:02d}"] = [load(f, v.path, ci) for f in files]
    name, seeds = ANCHOR[model]
    recipes["anchor"] = [load(scores / f"{name}_s{s}.csv", v.path, ci) for s in seeds]
    per = {r: [macro_f1(truth, p, k) for p in ps] for r, ps in recipes.items()}
    mean = {r: float(np.mean(x)) for r, x in per.items()}
    tuned = sorted((r for r in recipes if r != "anchor"), key=mean.get, reverse=True)
    leader = tuned[0]
    boot = {r: np.empty(n_boot) for r in recipes}
    n = len(truth)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        tb = truth[idx]
        for r, ps in recipes.items():
            boot[r][b] = np.mean([macro_f1(tb, ps[j][idx], k) for j in rng.integers(0, len(ps), len(ps))])
    rows = []
    for r in tuned + ["anchor"]:
        da, dl = boot[r] - boot["anchor"], boot[r] - boot[leader]
        row = {"recipe": r, "per_seed": [round(x, 4) for x in per[r]], "mean": round(mean[r], 4),
               "sd": round(float(np.std(per[r], ddof=1)), 4),
               "vs_anchor": round(mean[r] - mean["anchor"], 4),
               "vs_anchor_ci95": [round(float(q), 4) for q in np.quantile(da, [0.025, 0.975])],
               "vs_leader": round(mean[r] - mean[leader], 4),
               "vs_leader_ci95": [round(float(q), 4) for q in np.quantile(dl, [0.025, 0.975])]}
        if r != "anchor":
            row.update(trials[(trials.model == model) & (trials.trial == int(r[1:]))].drop(columns=["run", "model", "trial"])
                       .iloc[0].to_dict())
        rows.append(row)
    return {"leader": leader, "runner_up": tuned[1], "recipes": rows}


def main(scores, val_csv, trials_csv, out):
    v = pd.read_csv(val_csv)
    ci = {c: i for i, c in enumerate(sorted(v.label.unique()))}
    truth = v.label.map(ci).to_numpy()
    trials = pd.read_csv(trials_csv)
    rng = np.random.default_rng(0)
    res = {m: model_eval(m, Path(scores), v, ci, truth, trials, rng) for m in ANCHOR}
    json.dump(res, open(out, "w"), indent=1, default=float)
    for m, r in res.items():
        print(f"\n{m}: best {r['leader']}, runner-up {r['runner_up']}")
        for x in r["recipes"]:
            print(f"  {x['recipe']:7s} {x['mean']:.4f} sd {x['sd']:.4f} vs anchor {x['vs_anchor']:+.4f} "
                  f"[{x['vs_anchor_ci95'][0]:+.4f}, {x['vs_anchor_ci95'][1]:+.4f}] vs best {x['vs_leader']:+.4f} "
                  f"[{x['vs_leader_ci95'][0]:+.4f}, {x['vs_leader_ci95'][1]:+.4f}] {x['per_seed']}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
