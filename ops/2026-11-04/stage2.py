"""Recipe tuning stage 2 (PLAN 2026-11-03): the top 8 stage-1 trials of a model by seed-1 macro-F1 on the v2c
validation snips get seeds 2 and 3. Reads the registry rows (JSON lines), writes the seed-2 and seed-3 configs
beside the seed-1 ones and prints their queue lines in seed order.
Usage: python ops/2026-11-04/stage2.py REGISTRY_JSONL MODEL TUNE_DIR"""
import json
import sys
from pathlib import Path

rows = [json.loads(l) for l in open(sys.argv[1])]
model, tune = sys.argv[2], Path(sys.argv[3])
f1 = {r["name"]: r["macro_f1"] for r in rows if r["name"].startswith(f"tn_{model}_t") and r["name"].endswith("_s1")}
top = sorted(f1, key=f1.get, reverse=True)[:8]
print(f"# {model} top 8: " + ", ".join(f"{n} {f1[n]:.4f}" for n in top), file=sys.stderr)
for seed in (2, 3):
    for n in top:
        cfg = json.load(open(tune / f"{n}.json"))
        cfg["seed"] = seed
        out = n[:-1] + str(seed)
        json.dump(cfg, open(tune / f"{out}.json", "w"), indent=1)
        print(f"torch {out} /home/ubuntu/mewc-synth/ops/2026-11-03/tune/{out}.json")
