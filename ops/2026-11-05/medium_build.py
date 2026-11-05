"""Medium-scale confirmation (PLAN 2026-11-05): the best two stage-2 recipes per model and the default-recipe
anchor at 20% of production compute (mewc-train production: 4,000 per class, 10 frozen + 30 progressive epochs;
medium: 1,600 per class, 4 frozen + 16 progressive in four stages of 4), seeds 1 and 2. Seed 1 runs on
ngx-gpu and seed 2 on ngx-gpu-5, HGNetV2-B4 first on each.
Usage: python ops/2026-11-05/medium_build.py OPS_DIR"""
import json
import sys
from pathlib import Path

MEDIUM = {"samples_per_class": 1600, "frozen_epochs": 4, "stage_len": 4, "total_epochs": 16}
RECIPES = {
    "hgnetv2_b4": {"t23": "2026-11-03/tune/tn_hgnetv2_b4_t23_s1.json", "t21": "2026-11-03/tune/tn_hgnetv2_b4_t21_s1.json",
                   "anchor": "2026-11-02/shortlist/sl_hgnetv2_b4_lr1_s1.json"},
    "eva02_b": {"t21": "2026-11-03/tune/tn_eva02_b_t21_s1.json", "t16": "2026-11-03/tune/tn_eva02_b_t16_s1.json",
                "anchor": "2026-11-02/shortlist/sl_eva02_b_lr1_s4.json"},
}


def main(ops):
    ops = Path(ops)
    out = ops / "2026-11-05" / "medium"
    out.mkdir(exist_ok=True)
    queues = {1: [], 2: []}
    for model, recipes in RECIPES.items():
        for r, src in recipes.items():
            base = json.loads((ops / src).read_text())
            for seed in (1, 2):
                run = f"md_{model}_{r}_s{seed}"
                json.dump({**base, **MEDIUM, "seed": seed}, open(out / f"{run}.json", "w"), indent=1)
                queues[seed].append(f"torch {run} /home/ubuntu/mewc-synth/ops/2026-11-05/medium/{run}.json\n")
    for seed, host in ((1, "g4"), (2, "g5")):
        (ops / "2026-11-05" / f"medium_{host}.queue").write_text("".join(queues[seed]))
    print("".join(queues[1] + queues[2]))


if __name__ == "__main__":
    main(sys.argv[1])
