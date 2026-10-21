"""Draw nested pseudo-rare training subsets from the train split: groups in seeded order,
at most PER_GROUP snips from each, until the largest n. Smaller n are prefixes of the same
ordering, so the n=20 subset lies inside the n=50 subset. Written once and reused by every
generator and classifier run.
Usage: python src/pseudo_rare.py INV_DIR SPLIT_NAME CONFIG_JSON
CONFIG_JSON example: {"classes": ["black_rat"], "ns": [50, 20], "per_group": 5, "seed": 21}"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(inv_dir, split_name, config):
    cfg, out = json.loads(config), Path(inv_dir) / "splits" / split_name
    split = pd.read_parquet(out / "split.parquet")
    train = split[split["split"] == "train"]
    rng = np.random.default_rng(cfg["seed"])
    n_max, rows = max(cfg["ns"]), []
    for cls in cfg["classes"]:
        cdf = train[train["class_code"] == cls]
        members = cdf.groupby("group").indices
        order = []
        for g in rng.permutation(sorted(members)):
            idx = members[g]
            take = rng.choice(idx, min(cfg["per_group"], len(idx)), replace=False)
            order.extend(cdf["relpath"].values[take])
            if len(order) >= n_max:
                break
        order = order[:n_max]
        for n in cfg["ns"]:
            rows += [{"class_code": cls, "n": n, "rank": r, "relpath": p} for r, p in enumerate(order[:n])]
    df = pd.DataFrame(rows)
    df.to_parquet(out / "pseudo_rare.parquet", index=False)
    (out / "pseudo_rare_config.json").write_text(json.dumps(cfg, indent=1))
    summary = df.merge(split[["relpath", "group"]], on="relpath").groupby(["class_code", "n"])["group"].nunique()
    print(json.dumps({f"{c}/{n}": int(g) for (c, n), g in summary.items()}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
