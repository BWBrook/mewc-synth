"""Build symlinked mewc-train folders for one classifier-screening arm, plus shared val/test folders.
Symlink targets use the container path (CONTAINER_ROOT) so they resolve inside zaandahl/mewc-train
with the data volume mounted there. Only in-model classes are used.
Usage: python src/build_screen.py INV_DIR SPLIT_NAME CLASS_TABLE_CSV OUT_ROOT CONFIG_JSON
CONFIG_JSON example: {"arm": "A1", "train_cap": 4000, "val_cap": 100, "seed": 41,
  "pseudo_rare": {"black_rat": 50}, "synthetic": {"black_rat": "translate_screen"}, "synthetic_n": 200,
  "real_repeat": {"black_rat": 4}}
"synthetic" maps a class to a directory (relative to the data root) holding manifest.parquet.
"real_repeat" links each real snip of a class k times (as rep1_..., rep2_...), which sets the real
share of mewc-train's uniform per-class sampling without changing the synthetic set."""
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA

CONTAINER_ROOT = "/data"


def link(dst_dir, files, repeat=1):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        for j in range(repeat):
            os.symlink(f, dst_dir / (f"rep{j}_{Path(f).name}" if j else Path(f).name))


def container_snip(relpath):
    return f"{CONTAINER_ROOT}/work/snips/{relpath}"


def main(inv_dir, split_name, class_table, out_root, config):
    cfg, inv_dir, out = json.loads(config), Path(inv_dir), Path(out_root)
    classes = sorted(pd.read_csv(class_table).query("in_model")["class"])
    split = pd.read_parquet(inv_dir / "splits" / split_name / "split.parquet")
    split = split[split["class_code"].isin(classes)]
    rng = np.random.default_rng(cfg["seed"])
    shared = out / "shared"
    if not shared.exists():
        for c in classes:
            test = split[(split["class_code"] == c) & (split["split"] == "test")]["relpath"].tolist()
            val = split[(split["class_code"] == c) & (split["split"] == "val")]["relpath"].tolist()
            val = sorted(rng.choice(val, min(cfg["val_cap"], len(val)), replace=False))
            link(shared / "test" / c, [container_snip(r) for r in test])
            link(shared / "val" / c, [container_snip(r) for r in val])
            link(shared / "test_stub" / c, [container_snip(val[0])])
    arm = out / cfg["arm"]
    if arm.exists():
        raise FileExistsError(f"{arm} exists")
    pr = pd.read_parquet(inv_dir / "splits" / split_name / "pseudo_rare.parquet") if cfg.get("pseudo_rare") else None
    counts = []
    for c in classes:
        tr = split[(split["class_code"] == c) & (split["split"] == "train")]["relpath"]
        if c in cfg.get("pseudo_rare", {}):
            keep = set(pr[(pr["class_code"] == c) & (pr["n"] == cfg["pseudo_rare"][c])]["relpath"])
            tr = tr[tr.isin(keep)]
            assert len(tr) == cfg["pseudo_rare"][c], (c, len(tr))
        elif len(tr) > cfg["train_cap"]:
            tr = tr.iloc[sorted(np.random.default_rng(cfg["seed"] + zlib.crc32(c.encode())).choice(len(tr), cfg["train_cap"], replace=False))]
        files = [container_snip(r) for r in tr]
        rep = cfg.get("real_repeat", {}).get(c, 1)
        link(arm / "train" / c, files, rep)
        n_syn = 0
        if c in cfg.get("synthetic", {}):
            sdir = cfg["synthetic"][c]
            man = pd.read_parquet(DATA / sdir / "manifest.parquet")
            syn = man[man["class_code"] == c]["relpath"].head(cfg["synthetic_n"]).tolist()
            assert len(syn) == cfg["synthetic_n"], (c, len(syn))
            link(arm / "train" / c, [f"{CONTAINER_ROOT}/{sdir}/{r}" for r in syn])
            n_syn = len(syn)
        counts.append({"class": c, "real": len(tr), "real_repeat": rep, "synthetic": n_syn})
    pd.DataFrame(counts).to_csv(out / f"{cfg['arm']}_composition.csv", index=False)
    (out / f"{cfg['arm']}_config.json").write_text(json.dumps(cfg, indent=1))
    total = pd.DataFrame(counts)
    print(json.dumps({"arm": cfg["arm"], "classes": len(classes), "real": int(total["real"].sum()),
                      "synthetic": int(total["synthetic"].sum()),
                      "pseudo_rare": {k: v for k, v in cfg.get("pseudo_rare", {}).items()}}))


if __name__ == "__main__":
    main(*sys.argv[1:6])
