"""Q2 of the third booking (docs/plan-booking-3.md): the screen-scale tie sets re-run at medium scale, decided by
the shortlist rule (PLAN 2026-11-02): per track the leader is the highest two-seed mean macro-F1 on the v2c
validation snips; a model whose two-level bootstrap interval of the difference from the leader (seeds resampled
within model, snips resampled jointly) contains zero ties, and the tie goes to the cheaper model on the track's
cost key (alien: 2080 fp16 at batch 32 for accurate, 8-core CPU for deployable; ops/2026-11-02/hw_alien.jsonl).
A track is evaluated only once every model has every seed. SEEDS (default 1,2) extends it to more seeds (PLAN
2026-11-23: four seeds); each model is then also compared with the chosen model (EVA-02-B, HGNetV2-B4), and
"above_chosen" marks an interval entirely above zero.
Usage: python ops/2026-11-20/medium_shortlist_eval.py SCORES_DIR VAL_CSV HW_JSONL OUT_JSON [SEEDS]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRACKS = {
    "deployable": ("cpu8", {"hgnetv2_b4": ("md_hgnetv2_b4_anchor_s{s}", "hgnetv2_b4_lr1"),
                            "efficientnetv2_s": ("md_ens_s{s}", "ens_lr1"),
                            "fastvit_sa24": ("md_fastvit_sa24_s{s}", "fastvit_sa24_lr1"),
                            "mobilenetv4_hl": ("md_mnv4_hl_s{s}", "mnv4_hl_lr1")}),
    "accurate": ("gpu_fp16_b32", {"eva02_b": ("md_eva02_b_anchor_s{s}", "eva02_b_lr1"),
                                  "dinov3_l": ("md_dinov3_l_s{s}", "dinov3_l_lr1"),
                                  "convnextv2_l": ("md_cnv2_l_s{s}", "cnv2_l_base"),
                                  "eva02_l": ("md_eva02_l_s{s}", "eva02_l_bmore")}),
}
CHOSEN = {"deployable": "hgnetv2_b4", "accurate": "eva02_b"}


def macro_f1(truth, pred, k):
    tp = np.bincount(truth[pred == truth], minlength=k)
    denom = np.bincount(truth, minlength=k) + np.bincount(pred, minlength=k)
    present = np.bincount(truth, minlength=k) > 0
    return float((2 * tp[present] / denom[present]).mean())


def main(scores, val_csv, hw, out, seeds="1,2", n_boot=1000):
    SEEDS = tuple(int(s) for s in seeds.split(","))
    v = pd.read_csv(val_csv)
    ci = {c: i for i, c in enumerate(sorted(v.label.unique()))}
    k, truth = len(ci), v.label.map(ci).to_numpy()
    cost = {}
    for line in open(hw):
        r = json.loads(line)
        tag = "fused" if "fused_cpu8" in r else "plain"
        cost[r["name"]] = {"cpu8": r[f"{tag}_cpu8"], "gpu_fp16_b32": r[f"{tag}_cuda_fp16_b32"]}
    rng, res = np.random.default_rng(0), {}
    for track, (key, models) in TRACKS.items():
        files = {m: [Path(scores) / f"{t.format(s=s)}.csv" for s in SEEDS] for m, (t, _) in models.items()}
        dropped = sorted(m for m, fs in files.items() if len(SEEDS) > 2 and not all(f.exists() for f in fs))
        files = {m: fs for m, fs in files.items() if m not in dropped}
        models = {m: x for m, x in models.items() if m not in dropped}
        if not all(f.exists() for fs in files.values() for f in fs):
            res[track] = {"complete": False, "missing": [f.stem for fs in files.values() for f in fs if not f.exists()]}
            continue
        preds = {m: [pd.read_csv(f).set_index("path").loc[v.path, "top_name"].map(ci).fillna(-1).astype(int).to_numpy()
                     for f in fs] for m, fs in files.items()}
        per = {m: [macro_f1(truth, p, k) for p in ps] for m, ps in preds.items()}
        mean = {m: float(np.mean(x)) for m, x in per.items()}
        leader = max(mean, key=mean.get)
        boot = {m: np.empty(n_boot) for m in models}
        for b in range(n_boot):
            idx = rng.integers(0, len(truth), len(truth))
            for m, ps in preds.items():
                boot[m][b] = np.mean([macro_f1(truth[idx], ps[j][idx], k) for j in rng.integers(0, len(SEEDS), len(SEEDS))])
        rows = []
        for m in sorted(models, key=mean.get, reverse=True):
            lo, hi = np.quantile(boot[m] - boot[leader], [0.025, 0.975])
            clo, chi = np.quantile(boot[m] - boot[CHOSEN[track]], [0.025, 0.975])
            rows.append({"model": m, "per_seed": [round(x, 4) for x in per[m]], "mean": round(mean[m], 4),
                         "diff_vs_leader": round(mean[m] - mean[leader], 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                         "ties_leader": m == leader or bool(lo <= 0 <= hi), key: cost[models[m][1]][key],
                         "diff_vs_chosen": round(mean[m] - mean[CHOSEN[track]], 4),
                         "ci95_vs_chosen": [round(float(clo), 4), round(float(chi), 4)], "above_chosen": bool(clo > 0)})
        ties = [r for r in rows if r["ties_leader"]]
        res[track] = {"complete": True, "seeds": list(SEEDS), "dropped_without_all_seeds": dropped, "leader": leader, "cost_key": key, "tied": [r["model"] for r in ties],
                      "winner": max(ties, key=lambda r: r[key])["model"], "models": rows}
    json.dump(res, open(out, "w"), indent=1)
    for t, r in res.items():
        if not r["complete"]:
            print(t, "incomplete:", r["missing"])
            continue
        print(f"\n{t}: leader {r['leader']}, tied {r['tied']}, winner on {r['cost_key']}: {r['winner']}")
        for x in r["models"]:
            print(f"  {x['model']:18s} {x['mean']:.4f} {x['per_seed']} diff {x['diff_vs_leader']:+.4f} "
                  f"[{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] {'TIE' if x['ties_leader'] else '   '} {r['cost_key']} {x[r['cost_key']]} | vs chosen {x['diff_vs_chosen']:+.4f} "
                  f"[{x['ci95_vs_chosen'][0]:+.4f}, {x['ci95_vs_chosen'][1]:+.4f}]{' ABOVE' if x['above_chosen'] else ''}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
