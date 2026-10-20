"""Assign whole near-duplicate groups to test, validation and train, per class, with a fixed seed.
Classes with fewer than SMALL snips: 20% test, 10% validation. Larger classes: 10% test
(capped at TEST_CAP snips) and 5% validation (capped at VAL_CAP). Writes the split table plus
test and validation file lists with SHA-256 for committing.
Usage: python src/splits.py INV_DIR SPLIT_NAME [SEED]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SMALL, TEST_CAP, VAL_CAP = 500, 2000, 1000


def targets(n):
    if n < SMALL:
        return max(1, round(0.2 * n)), max(1, round(0.1 * n))
    return min(round(0.1 * n), TEST_CAP), min(round(0.05 * n), VAL_CAP)


def main(inv_dir, name, seed=20261020):
    inv_dir, seed = Path(inv_dir), int(seed)
    inv = pd.read_parquet(inv_dir / "inventory.parquet")
    grp = pd.read_parquet(inv_dir / "groups.parquet")
    df = inv[["relpath", "class_code", "sha256"]].merge(grp[["relpath", "group"]], on="relpath")
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index)
    for code, cdf in df.groupby("class_code"):
        n_test, n_val = targets(len(cdf))
        members = cdf.groupby("group").indices
        order = rng.permutation(sorted(members))
        filled = {"test": 0, "val": 0}
        for g in order:
            for part, want in (("test", n_test), ("val", n_val)):
                if filled[part] < want:
                    split[cdf.index[members[g]]] = part
                    filled[part] += len(members[g])
                    break
            else:
                break
    df["split"] = split
    out = inv_dir / "splits" / name
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "split.parquet", index=False)
    for part in ("test", "val"):
        df.loc[df["split"] == part, ["relpath", "class_code", "sha256"]].sort_values("relpath").to_csv(out / f"{part}.csv", index=False)
    counts = df.groupby(["class_code", "split"]).size().unstack(fill_value=0)
    counts.to_csv(out / "counts.csv")
    summary = {"name": name, "seed": seed, "totals": df["split"].value_counts().to_dict(),
               "classes_without_val": int((counts.get("val", 0) == 0).sum()),
               "groups_split_across_parts": int((df.groupby("group")["split"].nunique() > 1).sum())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main(*sys.argv[1:4])
