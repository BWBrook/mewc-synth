"""Evaluate the seeded shortlist round (PLAN 2026-11-02): overall macro-F1 on the v2c validation snips for every
run, the mean over seeds 1 to 3 (the registered rule) and over seeds 1 to 4 (accurate track, reported beside it),
and for each model the difference from its track's leader with a two-level bootstrap interval: seeds resampled
within each model, validation snips resampled jointly across models. A model ties with the leader when that
interval contains zero; the tie goes to the cheaper model on the track's hardware (alien, ops/2026-11-02).
Accurate-track seed 1 is the mini-search run (uncorrected training pool); every other run is sl_*.
Usage: python ops/2026-11-03/shortlist_eval.py SCORES_DIR VAL_CSV HW_JSONL OUT_JSON"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ACCURATE = ["dinov3_l_lr1", "cnv2_l_base", "eva02_l_bmore", "caformer_s36_lr1", "eva02_b_lr1", "vtl_bmore",
            "caformer_b36_bmore"]
DEPLOYABLE = ["ens_lr1", "fastvit_sa24_lr1", "hgnetv2_b4_lr1", "mnv4_hl_lr1", "mnv4_cl_lr1"]


def macro_f1(truth, pred, k):
    tp = np.bincount(truth[pred == truth], minlength=k)
    denom = np.bincount(truth, minlength=k) + np.bincount(pred, minlength=k)
    present = np.bincount(truth, minlength=k) > 0
    return float((2 * tp[present] / denom[present]).mean())


def run_file(scores, model, seed):
    if model in ACCURATE and seed == 1:
        return scores / f"mini_{model}.csv"
    return scores / f"sl_{model}_s{seed}.csv"


def track(name, models, seeds, preds, truth, k, cost, rng, n_boot=1000):
    runs = {m: [preds[(m, s)] for s in seeds if (m, s) in preds] for m in models}
    if any(len(v) != len(seeds) for v in runs.values()):
        return None
    per_run = {m: [macro_f1(truth, p, k) for p in v] for m, v in runs.items()}
    means = {m: float(np.mean(v)) for m, v in per_run.items()}
    leader = max(means, key=means.get)
    n = len(truth)
    boot = {m: [] for m in models}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        t = truth[idx]
        for m, v in runs.items():
            boot[m].append(np.mean([macro_f1(t, v[j][idx], k) for j in rng.integers(0, len(v), len(v))]))
    rows = []
    for m in sorted(models, key=means.get, reverse=True):
        d = np.array(boot[m]) - np.array(boot[leader])
        lo, hi = np.quantile(d, [0.025, 0.975])
        rows.append({"model": m, "seeds": list(seeds), "per_seed": [round(x, 4) for x in per_run[m]],
                     "mean": round(means[m], 4), "sd": round(float(np.std(per_run[m], ddof=1)), 4),
                     "diff_vs_leader": round(means[m] - means[leader], 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                     "ties_leader": bool(lo <= 0 <= hi) or m == leader, **cost[m]})
    ties = [r for r in rows if r["ties_leader"]]
    key = "gpu_fp16_b32" if name == "accurate" else "cpu8"
    return {"leader": leader, "cost_key": key, "winner": max(ties, key=lambda r: r[key])["model"],
            "tied": [r["model"] for r in ties], "models": rows}


def main(scores, val_csv, hw, out):
    scores = Path(scores)
    v = pd.read_csv(val_csv)
    classes = sorted(v.label.unique())
    ci = {c: i for i, c in enumerate(classes)}
    truth = v.label.map(ci).to_numpy()
    preds = {}
    for m in ACCURATE + DEPLOYABLE:
        for s in (1, 2, 3, 4):
            f = run_file(scores, m, s)
            if f.exists():
                p = pd.read_csv(f).set_index("path").loc[v.path, "top_name"]
                preds[(m, s)] = p.map(ci).fillna(-1).astype(int).to_numpy()
    cost = {}
    for line in open(hw):
        r = json.loads(line)
        tag = "fused" if "fused_cpu8" in r else "plain"
        cost[r["name"]] = {"cpu8": r[f"{tag}_cpu8"], "gpu_fp32_b32": r[f"{tag}_cuda_fp32_b32"],
                           "gpu_fp16_b32": r[f"{tag}_cuda_fp16_b32"]}
    rng = np.random.default_rng(0)
    res = {"n_val": len(truth),
           "accurate_s1_3": track("accurate", ACCURATE, (1, 2, 3), preds, truth, len(classes), cost, rng),
           "accurate_s1_4": track("accurate", ACCURATE, (1, 2, 3, 4), preds, truth, len(classes), cost, rng),
           "deployable_s1_3": track("deployable", DEPLOYABLE, (1, 2, 3), preds, truth, len(classes), cost, rng)}
    json.dump(res, open(out, "w"), indent=1)
    for k, t in res.items():
        if k == "n_val" or t is None:
            continue
        print(f"\n{k}: leader {t['leader']}, tied {t['tied']}, winner on {t['cost_key']}: {t['winner']}")
        for r in t["models"]:
            print(f"  {r['model']:20s} {r['mean']:.4f} sd {r['sd']:.4f} diff {r['diff_vs_leader']:+.4f} "
                  f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] {'TIE' if r['ties_leader'] else '   '} "
                  f"cpu {r['cpu8']} gpu16 {r['gpu_fp16_b32']} seeds {r['per_seed']}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
