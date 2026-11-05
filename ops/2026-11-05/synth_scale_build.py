"""Synthetic-snip gain against training scale (PLAN 2026-11-05): HGNetV2-B4 on the current recipe (lr 1e-4,
2 stages), arms A0 and S20 of the phase 1 re-test (`v2_runs/arms`, ops/2026-11-01), at screen scale (400 per
class, 3 frozen + 12 progressive) and medium scale (1,600 per class, 4 frozen + 16 progressive), seeds 1 to 8
paired by seed within each scale. Screen runs first, then medium in seed order, all on ngx-gpu (gpu-5 has too little disk).
Usage: python ops/2026-11-05/synth_scale_build.py OPS_DIR"""
import json
import sys
from pathlib import Path

SCALES = {"screen": {"samples_per_class": 400, "frozen_epochs": 3, "stage_len": 3, "total_epochs": 12},
          "medium": {"samples_per_class": 1600, "frozen_epochs": 4, "stage_len": 4, "total_epochs": 16}}
ARMS = "/mnt/ml-data/mewc-synth/v2_runs/arms"


def main(ops):
    ops = Path(ops)
    base = json.loads((ops / "2026-11-02/shortlist/sl_hgnetv2_b4_lr1_s1.json").read_text())
    out = ops / "2026-11-05" / "synth_scale"
    out.mkdir(exist_ok=True)
    lines = []
    for scale, sc in SCALES.items():
        for seed in range(1, 9):
            for arm in ("A0", "S20"):
                run = f"ss_{scale}_{arm}_s{seed}"
                json.dump({**base, **sc, "seed": seed, "train": f"{ARMS}/{arm}.csv"}, open(out / f"{run}.json", "w"), indent=1)
                lines.append(f"torch {run} /home/ubuntu/mewc-synth/ops/2026-11-05/synth_scale/{run}.json\n")
    (ops / "2026-11-05" / "synth_scale_g4.queue").write_text("".join(lines))
    print(len(lines), "runs")


if __name__ == "__main__":
    main(sys.argv[1])
