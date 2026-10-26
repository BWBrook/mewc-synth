"""Split v2 over the full library (96 model classes), built to resist leakage (docs/plan-next-booking.md):
1. Exact duplicates (same size and CRC32): copies across classes are dropped; within a class one is kept.
2. Project-disjoint test stratum: whole source projects (HOLDOUT prefixes) become test_proj, except in
   classes where they hold more than PROJ_MAX_SHARE of the snips.
3. Validation and test drawn per class by whole cosine-0.90 groups no larger than MAX_GROUP, at most
   PER_GROUP snips per group; the rest of a drawn group is excluded, never trained on.
   Quotas: test min(2000, 20% of the class), validation min(500, 10%).
4. Purge: training snips within cosine PURGE of any validation or test snip of their class are removed
   (0.88: at 0.85 the band was mostly different animals in similar poses, especially eBird photos).
5. Training pool: training snips capped at POOL_CAP per class, preferring snips kept by the 0.90
   thinning, then a seeded random fill.
Usage: python src/split_v2.py CURATION_DIR CLASS_TABLE_CSV OUT_DIR"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import source_prefix

HOLDOUT = ("mtwt", "hr")
PROJ_MAX_SHARE, MAX_GROUP, PER_GROUP = 0.20, 200, 20
TEST_CAP, VAL_CAP, TEST_FRAC, VAL_FRAC = 2000, 500, 0.20, 0.10
PURGE, POOL_CAP, SEED = 0.88, 40_000, 20261026


def draw(rows, labels, quota, rng, skip=frozenset()):
    """Whole groups in random order until the quota is filled; returns (taken, excluded) row arrays."""
    order = np.argsort(labels, kind="stable")
    uniq, start, count = np.unique(labels[order], return_index=True, return_counts=True)
    taken, excluded, n = [], [], 0
    for k in rng.permutation(len(uniq)):
        if n >= quota:
            break
        if count[k] > MAX_GROUP or uniq[k] in skip:
            continue
        members = rows[order[start[k]:start[k] + count[k]]]
        pick = rng.permutation(members)[:min(PER_GROUP, quota - n)]
        taken.extend(pick)
        excluded.extend(np.setdiff1d(members, pick))
        n += len(pick)
    return np.array(taken, dtype=int), np.array(excluded, dtype=int)


def main(curation_dir, class_table, out_dir):
    d, out = Path(curation_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    classes = set(pd.read_csv(class_table).query("in_model")["class"])
    idx = pd.read_parquet(d / "index.parquet", columns=["zip_name", "class_code", "size", "crc32"])
    g = pd.read_parquet(d / "groups.parquet", columns=["zip_name", "group_90"])
    t = pd.read_parquet(d / "thin.parquet", columns=["zip_name", "keep_90"])
    assert (idx["zip_name"].to_numpy() == g["zip_name"].to_numpy()).all()
    assert (idx["zip_name"].to_numpy() == t["zip_name"].to_numpy()).all()
    df = idx.assign(group_90=g["group_90"].to_numpy(), keep_90=t["keep_90"].to_numpy())
    emb_rows = np.flatnonzero(df["class_code"].isin(classes).to_numpy())
    df = df.iloc[emb_rows].reset_index(drop=True)
    df["emb_row"] = emb_rows
    df["prefix"] = [source_prefix("/".join(z.split("/")[-2:])) for z in df["zip_name"]]
    df["split"] = "train"

    key = df["size"].astype(str) + ":" + df["crc32"].astype(str)
    cross = df.groupby(key)["class_code"].transform("nunique") > 1
    df.loc[cross, "split"] = "dup_cross"
    within = ~cross & key.duplicated()
    df.loc[within, "split"] = "dup"

    proj = df["prefix"].isin(HOLDOUT) & (df["split"] == "train")
    share = proj.groupby(df["class_code"]).transform("mean")
    df.loc[proj & (share <= PROJ_MAX_SHARE), "split"] = "test_proj"

    for cls, rows in df[df["split"] == "train"].groupby("class_code").groups.items():
        rows = np.asarray(rows)
        labels = df.loc[rows, "group_90"].to_numpy()
        n = len(rows)
        test, ex_t = draw(rows, labels, min(TEST_CAP, round(TEST_FRAC * n)), rng)
        val, ex_v = draw(rows, labels, min(VAL_CAP, round(VAL_FRAC * n)), rng,
                         skip=frozenset(df.loc[test, "group_90"]))
        df.loc[test, "split"], df.loc[val, "split"] = "test", "val"
        df.loc[np.concatenate([ex_t, ex_v]), "split"] = "excl_group"

    shards = sorted((d / "shards").glob("emb_*.npy"))
    emb = np.concatenate([np.load(p, mmap_mode="r") for p in shards])
    for cls, sub in df.groupby("class_code"):
        ev = sub[sub["split"].isin(["val", "test", "test_proj"])]
        tr = sub[sub["split"] == "train"]
        if ev.empty or tr.empty:
            continue
        e = np.asarray(emb[ev["emb_row"].to_numpy()], dtype=np.float32)
        best = np.full(len(tr), -1.0, dtype=np.float32)
        tr_rows = tr["emb_row"].to_numpy()
        for a in range(0, len(tr_rows), 50_000):
            z = np.asarray(emb[tr_rows[a:a + 50_000]], dtype=np.float32)
            best[a:a + 50_000] = (z @ e.T).max(axis=1)
        df.loc[tr.index[best >= PURGE], "split"] = "purged"

    df["pool"] = False
    for cls, sub in df[df["split"] == "train"].groupby("class_code"):
        if len(sub) <= POOL_CAP:
            df.loc[sub.index, "pool"] = True
            continue
        kept, rest = sub[sub["keep_90"]], sub[~sub["keep_90"]]
        take = kept.sample(min(POOL_CAP, len(kept)), random_state=SEED).index
        if len(take) < POOL_CAP:
            take = take.append(rest.sample(POOL_CAP - len(take), random_state=SEED).index)
        df.loc[take, "pool"] = True

    df.drop(columns=["emb_row"]).to_parquet(out / "split.parquet", index=False)
    summ = df.pivot_table(index="class_code", columns="split", values="zip_name", aggfunc="size", fill_value=0)
    summ["pool"] = df.groupby("class_code")["pool"].sum()
    summ.to_csv(out / "summary_by_class.csv")
    tot = {"snips": len(df), **df["split"].value_counts().to_dict(), "pool": int(df["pool"].sum()),
           "holdout_projects": list(HOLDOUT), "seed": SEED, "purge": PURGE, "pool_cap": POOL_CAP}
    (out / "summary.json").write_text(json.dumps(tot, indent=1))
    print(json.dumps(tot))


if __name__ == "__main__":
    main(*sys.argv[1:4])
