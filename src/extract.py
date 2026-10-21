"""Extract a per-class-capped working set from the MEWC snip zip (data_600px/<tier>/<class>/<file>).
Classes from tiers base, sufficient, need_more and too_few are kept; classes above CAP are
sampled with a fixed seed. zipfile verifies each entry's CRC-32 on read. Writes the class
table (code, binomial, tier, in-model flag, totals) and the selection manifest.
Usage: python src/extract.py ZIP CLASS_LIST_96_YAML OUT_ROOT [CAP] [SEED]"""
import json
import sys
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TIERS = ("base", "sufficient", "need_more", "too_few")
ZIP = None


def species_table(text):
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            rows.append({"code": parts[0], "binomial": " ".join(parts[1:-1]).replace("_", " "), "class": parts[-1]})
    return pd.DataFrame(rows)


def _init(path):
    global ZIP
    ZIP = zipfile.ZipFile(path)


def _extract(job):
    out, names = job
    n = 0
    for name in names:
        _, _, cls, fname = name.split("/")
        target = Path(out) / cls / fname
        target.write_bytes(ZIP.read(name))
        n += 1
    return n


def main(zip_path, class_list, out_root, cap=30000, seed=20261021):
    cap, seed, out = int(cap), int(seed), Path(out_root)
    z = zipfile.ZipFile(zip_path)
    species = species_table(z.read("data_600px/species_list.txt").decode(errors="replace"))
    model_codes = {str(k) for k in yaml.safe_load(open(class_list))}
    entries = defaultdict(list)
    tier_of = {}
    for info in z.infolist():
        parts = info.filename.split("/")
        if info.is_dir() or len(parts) != 4 or parts[1] not in TIERS:
            continue
        entries[parts[2]].append(info.filename)
        tier_of[parts[2]] = parts[1]
    rng = np.random.default_rng(seed)
    selected, rows = [], []
    for cls in sorted(entries):
        names = sorted(entries[cls])
        pick = names if len(names) <= cap else sorted(rng.choice(names, cap, replace=False))
        selected.extend(pick)
        rows.append({"class": cls, "tier": tier_of[cls], "n_total": len(names), "n_selected": len(pick)})
    table = pd.DataFrame(rows).merge(species, on="class", how="left")
    missing = table[table["code"].isna()]["class"].tolist()
    if missing:
        raise ValueError(f"Classes without a species_list entry: {missing}")
    table["in_model"] = table["code"].isin(model_codes)
    out.mkdir(parents=True, exist_ok=True)
    for cls in entries:
        (out / cls).mkdir(exist_ok=True)
    chunks = [(str(out), selected[i:i + 2000]) for i in range(0, len(selected), 2000)]
    with Pool(32, initializer=_init, initargs=(zip_path,)) as pool:
        done = sum(pool.imap_unordered(_extract, chunks))
    meta = out.parent / (out.name + "_meta")
    meta.mkdir(exist_ok=True)
    table.to_csv(meta / "class_table.csv", index=False)
    pd.DataFrame({"zip_name": selected}).to_parquet(meta / "selection.parquet", index=False)
    summary = {"zip": str(zip_path), "cap": cap, "seed": seed, "classes": len(table), "extracted": done,
               "in_model_classes": int(table["in_model"].sum()),
               "by_tier": table.groupby("tier")[["n_total", "n_selected"]].sum().to_dict("index")}
    (meta / "extract_summary.json").write_text(json.dumps(summary, indent=1, default=int))
    print(json.dumps(summary, default=int))


if __name__ == "__main__":
    main(*sys.argv[1:6])
