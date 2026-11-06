"""Medium-scale confirmation (PLAN 2026-11-05): two-seed mean macro-F1 on the v2c validation snips for each
model's best two stage-2 recipes and the anchor, each tuned recipe's difference from the anchor with a two-level
bootstrap interval (seeds resampled within recipe, snips resampled jointly), the registered rule's interim recipe,
and the screen-scale three-seed means beside them (from ops/2026-11-05/stage2_eval.json) as the fidelity check.
Usage: python ops/2026-11-06/medium_eval.py SCORES_DIR VAL_CSV STAGE2_JSON OUT_JSON"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RECIPES = {"eva02_b": ["t21", "t16", "anchor"], "hgnetv2_b4": ["t23", "t21", "anchor"]}


def macro_f1(truth, pred, k):
    tp = np.bincount(truth[pred == truth], minlength=k)
    denom = np.bincount(truth, minlength=k) + np.bincount(pred, minlength=k)
    present = np.bincount(truth, minlength=k) > 0
    return float((2 * tp[present] / denom[present]).mean())


def main(scores, val_csv, stage2, out, n_boot=1000):
    v = pd.read_csv(val_csv)
    ci = {c: i for i, c in enumerate(sorted(v.label.unique()))}
    k, truth = len(ci), v.label.map(ci).to_numpy()
    screen = {m: {r["recipe"]: r["mean"] for r in res["recipes"]} for m, res in json.load(open(stage2)).items()}
    rng = np.random.default_rng(0)
    res = {}
    for model, recipes in RECIPES.items():
        preds = {r: [pd.read_csv(Path(scores) / f"md_{model}_{r}_s{s}.csv").set_index("path").loc[v.path, "top_name"]
                     .map(ci).fillna(-1).astype(int).to_numpy() for s in (1, 2)] for r in recipes}
        per = {r: [macro_f1(truth, p, k) for p in ps] for r, ps in preds.items()}
        mean = {r: float(np.mean(x)) for r, x in per.items()}
        boot = {r: np.empty(n_boot) for r in recipes}
        for b in range(n_boot):
            idx = rng.integers(0, len(truth), len(truth))
            for r, ps in preds.items():
                boot[r][b] = np.mean([macro_f1(truth[idx], ps[j][idx], k) for j in rng.integers(0, 2, 2)])
        best = max(recipes[:2], key=mean.get)
        res[model] = {"interim": best if mean[best] > mean["anchor"] else "anchor", "recipes": [
            {"recipe": r, "per_seed": [round(x, 4) for x in per[r]], "mean": round(mean[r], 4),
             "vs_anchor": round(mean[r] - mean["anchor"], 4),
             "vs_anchor_ci95": [round(float(q), 4) for q in np.quantile(boot[r] - boot["anchor"], [0.025, 0.975])],
             "screen_mean": screen[model][r], "screen_vs_anchor": round(screen[model][r] - screen[model]["anchor"], 4)}
            for r in recipes]}
    json.dump(res, open(out, "w"), indent=1)
    for m, r in res.items():
        print(f"\n{m}: interim recipe {r['interim']}")
        for x in r["recipes"]:
            print(f"  {x['recipe']:7s} medium {x['mean']:.4f} {x['per_seed']} vs anchor {x['vs_anchor']:+.4f} "
                  f"[{x['vs_anchor_ci95'][0]:+.4f}, {x['vs_anchor_ci95'][1]:+.4f}]   screen {x['screen_mean']:.4f} "
                  f"vs anchor {x['screen_vs_anchor']:+.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
