"""Extract a per-class-capped working set from the MEWC snip zip (data_600px/<tier>/<class>/<file>).
Classes from tiers base, sufficient, need_more and too_few are kept; classes above CAP are
sampled with a fixed seed. Writes the class table (code, binomial, tier, in-model flag, totals)
and the selection manifest.
The zip's central directory (4 M entries) costs about 6 GB as Python objects, so only the parent
parses it. Workers read each stored entry directly at its offset and check its CRC-32.
Usage: python src/extract.py ZIP CLASS_LIST_96_YAML OUT_ROOT [CAP] [SEED]"""
import gc
import json
import struct
import sys
import zipfile
import zlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TIERS = ("base", "sufficient", "need_more", "too_few")
ZIP_PATH = None


def species_table(text):
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            rows.append({"code": parts[0], "binomial": " ".join(parts[1:-1]).replace("_", " "), "class": parts[-1]})
    return pd.DataFrame(rows)


def _extract(job):
    out, items = job
    with open(ZIP_PATH, "rb") as f:
        for name, offset, size, crc in items:
            f.seek(offset)
            header = f.read(30)
            if header[:4] != b"PK\x03\x04":
                raise ValueError(f"Bad local header for {name}")
            n, e = struct.unpack("<HH", header[26:30])
            f.seek(offset + 30 + n + e)
            data = f.read(size)
            if zlib.crc32(data) != crc:
                raise ValueError(f"CRC mismatch for {name}")
            _, _, cls, fname = name.split("/")
            (Path(out) / cls / fname).write_bytes(data)
    return len(items)


def main(zip_path, class_list, out_root, cap=30000, seed=20261021):
    global ZIP_PATH
    ZIP_PATH, cap, seed, out = zip_path, int(cap), int(seed), Path(out_root)
    z = zipfile.ZipFile(zip_path)
    species = species_table(z.read("data_600px/species_list.txt").decode(errors="replace"))
    model_codes = {str(k) for k in yaml.safe_load(open(class_list))}
    entries, tier_of = defaultdict(list), {}
    for info in z.infolist():
        parts = info.filename.split("/")
        if info.is_dir() or len(parts) != 4 or parts[1] not in TIERS:
            continue
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"Expected stored entries, got compression {info.compress_type} for {info.filename}")
        entries[parts[2]].append((info.filename, info.header_offset, info.compress_size, info.CRC))
        tier_of[parts[2]] = parts[1]
    del z
    gc.collect()
    rng = np.random.default_rng(seed)
    selected, rows = [], []
    for cls in sorted(entries):
        items = sorted(entries[cls])
        if len(items) > cap:
            keep = sorted(rng.choice(len(items), cap, replace=False))
            items = [items[i] for i in keep]
        selected.extend(items)
        rows.append({"class": cls, "tier": tier_of[cls], "n_total": len(entries[cls]), "n_selected": len(items)})
    del entries
    gc.collect()
    table = pd.DataFrame(rows).merge(species, on="class", how="left")
    missing = table[table["code"].isna()]["class"].tolist()
    if missing:
        raise ValueError(f"Classes without a species_list entry: {missing}")
    table["in_model"] = table["code"].isin(model_codes)
    out.mkdir(parents=True, exist_ok=True)
    for cls in table["class"]:
        (out / cls).mkdir(exist_ok=True)
    chunks = [(str(out), selected[i:i + 2000]) for i in range(0, len(selected), 2000)]
    with Pool(16) as pool:
        done = sum(pool.imap_unordered(_extract, chunks))
    meta = out.parent / (out.name + "_meta")
    meta.mkdir(exist_ok=True)
    table.to_csv(meta / "class_table.csv", index=False)
    pd.DataFrame(selected, columns=["zip_name", "offset", "size", "crc32"]).to_parquet(meta / "selection.parquet", index=False)
    summary = {"zip": str(zip_path), "cap": cap, "seed": seed, "classes": len(table), "extracted": done,
               "in_model_classes": int(table["in_model"].sum()),
               "by_tier": table.groupby("tier")[["n_total", "n_selected"]].sum().to_dict("index")}
    (meta / "extract_summary.json").write_text(json.dumps(summary, indent=1, default=int))
    print(json.dumps(summary, default=int))


if __name__ == "__main__":
    main(*sys.argv[1:6])
