"""Assign whole near-duplicate groups to test, validation and train, per class, with a fixed seed.
Classes with fewer than SMALL snips: 20% test, 10% validation. Larger classes: 10% test
(capped at TEST_CAP snips) and 5% validation (capped at VAL_CAP). A group joins test or
validation only if it fits the remaining quota (plus SLACK), so giant groups (often many visits
to one camera) stay in train. Byte-identical files carrying more than one class label are
excluded from every split. Writes the split table plus test and validation file lists with
SHA-256 for committing.
Usage: python src/splits.py INV_DIR SPLIT_NAME [SEED]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SMALL, TEST_CAP, VAL_CAP, SLACK = 500, 2000, 1000, 10


def targets(n):
    if n < SMALL:
        return max(1, round(0.2 * n)), max(1, round(0.1 * n))
    return min(round(0.1 * n), TEST_CAP), min(round(0.05 * n), VAL_CAP)


def main(inv_dir, name, seed=20261020):
    inv_dir, seed = Path(inv_dir), int(seed)
    inv = pd.read_parquet(inv_dir / "inventory.parquet")
    grp = pd.read_parquet(inv_dir / "groups.parquet")
    df = inv[["relpath", "class_code", "sha256"]].merge(grp[["relpath", "group"]], on="relpath")
    conflict = df.groupby("sha256")["class_code"].transform("nunique") > 1
    rng = np.random.default_rng(seed)
    split = np.full(len(df), "train", dtype=object)
    split[conflict.to_numpy()] = "excluded"
    for code, cdf in df[~conflict].groupby("class_code"):
        n_test, n_val = targets(len(cdf))
        members = cdf.groupby("group").indices
        filled = {"test": 0, "val": 0}
        for g in rng.permutation(sorted(members)):
            size = len(members[g])
            for part, want in (("test", n_test), ("val", n_val)):
                if filled[part] < want and size <= want - filled[part] + SLACK:
                    split[cdf.index.to_numpy()[members[g]]] = part
                    filled[part] += size
                    break
            if filled["test"] >= n_test and filled["val"] >= n_val:
                break
    df["split"] = split
    out = inv_dir / "splits" / name
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "split.parquet", index=False)
    for part in ("test", "val", "excluded"):
        df.loc[df["split"] == part, ["relpath", "class_code", "sha256"]].sort_values("relpath").to_csv(out / f"{part}.csv", index=False)
    counts = df.groupby(["class_code", "split"]).size().unstack(fill_value=0)
    counts.to_csv(out / "counts.csv")
    kept = df[df["split"] != "excluded"]
    summary = {"name": name, "seed": seed, "totals": df["split"].value_counts().to_dict(),
               "classes_without_val": int((counts.get("val", 0) == 0).sum()),
               "groups_split_across_parts": int((kept.groupby("group")["split"].nunique() > 1).sum())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main(*sys.argv[1:4])
