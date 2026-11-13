"""Configs and queues for the decision-free runs of the third booking (docs/plan-booking-3.md), written to
ops/2026-11-13/configs and one queue per instance.
- ngx-gpu-4: Q1 A0 (HGNetV2-B4 production, full real pool) seeds 2 to 8; Q3 EVA-02-B at 2,400 per class
  (6 + 24) seeds 1 and 2 and medium seed 3; Q2 deployable track at medium, seeds 1 and 2; Q2 accurate track
  at medium, seed 1; Q4 seeds 1 to 3.
- ngx-gpu-5 (after rare_synth.sh): Q1 S20 seeds 1 to 8; Q2 accurate track, seed 2; Q4 seeds 4 to 6.
Usage: python ops/2026-11-13/build.py OPS_DIR"""
import json
import sys
from pathlib import Path

PROD = {"samples_per_class": 4000, "frozen_epochs": 10, "stage_len": 7, "total_epochs": 30}
MEDIUM = {"samples_per_class": 1600, "frozen_epochs": 4, "stage_len": 4, "total_epochs": 16}
L2400 = {"samples_per_class": 2400, "frozen_epochs": 6, "stage_len": 6, "total_epochs": 24}
ARMS = "/mnt/ml-data/mewc-synth/v2_runs/arms"
SL = {"dinov3_l": "dinov3_l_lr1", "cnv2_l": "cnv2_l_base", "eva02_l": "eva02_l_bmore",
      "ens": "ens_lr1", "fastvit_sa24": "fastvit_sa24_lr1", "mnv4_hl": "mnv4_hl_lr1"}


def main(ops):
    ops = Path(ops)
    out = ops / "2026-11-13" / "configs"
    out.mkdir(exist_ok=True)
    load = lambda p: json.loads((ops / p).read_text())
    hg_prod = load("2026-11-05/prod/pd_hgnetv2_b4_anchor_s1.json")
    eva_med = load("2026-11-05/medium/md_eva02_b_anchor_s1.json")
    q = {"g4": [], "g5": []}

    def add(gpu, run, cfg):
        (out / f"{run}.json").write_text(json.dumps(cfg, indent=1))
        q[gpu].append(f"torch {run} /home/ubuntu/mewc-synth/ops/2026-11-13/configs/{run}.json\n")

    for s in range(2, 9):
        add("g4", f"pd_hgnetv2_b4_anchor_s{s}", {**hg_prod, "seed": s})
    for s in (1, 2):
        add("g4", f"el_eva02_b_anchor_s{s}", {**eva_med, **L2400, "seed": s})
    add("g4", "md_eva02_b_anchor_s3", {**eva_med, "seed": 3})
    for s in (1, 2):
        for m in ("ens", "fastvit_sa24", "mnv4_hl"):
            add("g4", f"md_{m}_s{s}", {**load(f"2026-11-02/shortlist/sl_{SL[m]}_s2.json"), **MEDIUM, "seed": s})
    for s in range(1, 9):
        add("g5", f"rr_hgnetv2_b4_S20_s{s}", {**hg_prod, "seed": s, "train": f"{ARMS}/rare_S20.csv"})
    for s, gpu in ((1, "g4"), (2, "g5")):
        for m in ("dinov3_l", "cnv2_l", "eva02_l"):
            add(gpu, f"md_{m}_s{s}", {**load(f"2026-11-02/shortlist/sl_{SL[m]}_s2.json"), **MEDIUM, "seed": s})
    for s in range(1, 7):
        for n in (100, 200):
            add("g4" if s <= 3 else "g5", f"re_A0_{n}_s{s}", {**hg_prod, "seed": s, "train": f"{ARMS}/A0_{n}.csv"})
    for gpu, lines in q.items():
        (ops / "2026-11-13" / f"{gpu}.queue").write_text("".join(lines))
        print(gpu, len(lines))


if __name__ == "__main__":
    main(sys.argv[1])
