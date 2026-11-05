"""Production-scale config (PLAN 2026-11-05): a recipe's config at mewc-train production compute (4,000 per
class, 10 frozen + 30 progressive epochs in stages of 7, train.py's defaults). Writes
ops/2026-11-05/prod/pd_NAME_sSEED.json and prints its queue line.
Usage: python ops/2026-11-05/prod_build.py OPS_DIR NAME SOURCE_CONFIG SEED"""
import json
import sys
from pathlib import Path

PRODUCTION = {"samples_per_class": 4000, "frozen_epochs": 10, "stage_len": 7, "total_epochs": 30}


def main(ops, name, src, seed):
    out = Path(ops) / "2026-11-05" / "prod"
    out.mkdir(exist_ok=True)
    run = f"pd_{name}_s{seed}"
    cfg = {**json.loads(Path(src).read_text()), **PRODUCTION, "seed": int(seed)}
    (out / f"{run}.json").write_text(json.dumps(cfg, indent=1))
    print(f"torch {run} /home/ubuntu/mewc-synth/ops/2026-11-05/prod/{run}.json")


if __name__ == "__main__":
    main(*sys.argv[1:5])
