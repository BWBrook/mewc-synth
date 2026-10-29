"""Inventory of the extracted split v2 snips in the layout the generator pipeline reads (prep_stage_a.py,
pseudo_rare.py, translate.py, memorisation.py): INV_DIR/inventory.parquet (relpath, class_code) and
INV_DIR/splits/split_v2/split.parquet (relpath, class_code, group, split), with relpaths under the v2
root (train/<class>/<file>, where train is the capped pool) and groups from the 0.90 components.
Run src/embed.py V2_ROOT INV_DIR afterwards for embeddings, perceptual hashes and chroma.
Usage: python src/inventory_v2.py V2_ROOT CURATION_SPLIT_PARQUET INV_DIR"""
import sys
from pathlib import Path

import pandas as pd


def main(root, split_parquet, inv_dir):
    root, inv = Path(root), Path(inv_dir)
    sp = pd.read_parquet(split_parquet, columns=["zip_name", "class_code", "group_90", "split", "pool"])
    sp = sp[sp["pool"] | sp["split"].isin(["val", "test", "test_proj"])]
    sp = pd.DataFrame({"split": sp["split"].where(~sp["pool"], "train"), "class_code": sp["class_code"],
                       "file": sp["zip_name"].str.rsplit("/", n=1).str[-1], "group_90": sp["group_90"]})
    parts = []
    for part in ("train", "val", "test", "test_proj"):
        df = pd.read_csv(root / f"{part}.csv")
        rel = df["path"].str.split("/").str[-3:].str.join("/")
        parts.append(pd.DataFrame({"relpath": rel, "class_code": df["label"], "split": part,
                                   "file": rel.str.split("/").str[-1]}))
    df = pd.concat(parts, ignore_index=True)
    df = df.merge(sp, on=["split", "class_code", "file"], how="left", validate="one_to_one")
    if df["group_90"].isna().any():
        raise ValueError(f"{df['group_90'].isna().sum()} snips have no group")
    df = df.rename(columns={"group_90": "group"}).drop(columns="file")
    (inv / "splits" / "split_v2").mkdir(parents=True, exist_ok=True)
    df[["relpath", "class_code"]].to_parquet(inv / "inventory.parquet", index=False)
    df[["relpath", "class_code", "group", "split"]].to_parquet(inv / "splits" / "split_v2" / "split.parquet", index=False)
    print(df.groupby("split").size().to_dict(), df["class_code"].nunique(), "classes")


if __name__ == "__main__":
    main(*sys.argv[1:4])
