"""Synthetic scale test at production scale (PLAN 2026-11-06): the A0 and S20 arms of ops/2026-11-05/synth_scale_*
on HGNetV2-B4's current recipe at mewc-train production compute (4,000 per class, 10 frozen + 30 progressive),
seeds 1 to 6 paired: seeds 1 to 4 on gpu-5, 5 and 6 on ngx-gpu.
Usage: python ops/2026-11-06/synth_prod_build.py OPS_DIR"""
import json
import sys
from pathlib import Path

PRODUCTION = {"samples_per_class": 4000, "frozen_epochs": 10, "stage_len": 7, "total_epochs": 30}


def main(ops):
    ops = Path(ops)
    out = ops / "2026-11-06" / "synth_prod"
    out.mkdir(exist_ok=True)
    queues = {"g5": [], "g4": []}
    for seed in range(1, 7):
        for arm in ("A0", "S20"):
            run = f"ss_prod_{arm}_s{seed}"
            src = json.loads((ops / f"2026-11-05/synth_scale/ss_medium_{arm}_s{seed}.json").read_text())
            json.dump({**src, **PRODUCTION}, open(out / f"{run}.json", "w"), indent=1)
            queues["g5" if seed <= 4 else "g4"].append(f"torch {run} /home/ubuntu/mewc-synth/ops/2026-11-06/synth_prod/{run}.json\n")
    for q, lines in queues.items():
        (ops / "2026-11-06" / f"synth_prod_{q}.queue").write_text("".join(lines))
    print({q: len(v) for q, v in queues.items()})


if __name__ == "__main__":
    main(sys.argv[1])
