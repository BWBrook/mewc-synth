"""Stage B adapter configs for the real low-count classes (docs/plan-booking-3.md, Q1): the phase 1 recipe
(ops/2026-10-31/synth) on a_sv2, focus 0.45 on the class and 0.25 on its look-alike (lookalike_1 in the
profile), 400 steps, or 200 below 100 training snips. Writes cfg_<class>.json and plan.csv (class, look-alike,
steps).
Usage: python ops/2026-11-13/rare_cfgs.py PROFILE_CSV OUT_DIR"""
import json
import sys
from pathlib import Path

import pandas as pd

INIT = "/mnt/ml-data/mewc-synth/stage_a/a_sv2/lora/r64_s10k/step_010000"


def main(profile, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = dict(pd.read_csv("/mnt/ml-data/mewc-synth/work/snips_meta/class_table.csv")[["class", "binomial"]].values)
    lines = []
    for r in pd.read_csv(profile).itertuples():
        steps = 200 if r.train < 100 else 400
        light = "infrared night" if r.ir_share_camera > 0.5 else "daylight colour"
        cfg = {"rank": 64, "lr": 0.00005, "steps": steps, "batch": 16, "alpha": 0.5, "warmup": 50, "min_snr": 5.0,
               "save_every": steps, "sample_every": steps, "seed": 7,
               "sample_prompts": [f"camera trap snip of a {r[1].replace('_', ' ')} ({names[r[1]]}), {light}",
                                  f"camera trap snip of a {r.lookalike_1.replace('_', ' ')} ({names[r.lookalike_1]}), {light}"],
               "init_lora": INIT,
               "focus": {"classes": {r[1]: 0.45, r.lookalike_1: 0.25}, "prior_mass": 0.3, "prior_n": 3000}}
        (out / f"cfg_{r[1]}.json").write_text(json.dumps(cfg, indent=1))
        lines.append(f"{r[1]},{r.lookalike_1},{steps}\n")
    (out / "plan.csv").write_text("".join(lines))
    print(len(lines), "classes")


if __name__ == "__main__":
    main(*sys.argv[1:3])
