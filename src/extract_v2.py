"""Extract split v2 from the library zip into JPEG folders for training and evaluation:
train (the capped pool), val, test and test_proj, each as OUT/<part>/<class>/<file>, plus OUT/<part>.csv
lists (path, label) and OUT/classes.txt. Reads stored entries by offset with CRC checks, in parallel.
Usage: python src/extract_v2.py ZIP INDEX_PARQUET SPLIT_PARQUET OUT [WORKERS]"""
import struct
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

PARTS = {"train": None, "val": "val", "test": "test", "test_proj": "test_proj"}


def _write(job):
    zip_path, rows = job
    bad = 0
    with open(zip_path, "rb") as f:
        for dest, offset, size, crc in rows:
            f.seek(offset)
            n, e = struct.unpack("<HH", f.read(30)[26:30])
            f.seek(offset + 30 + n + e)
            data = f.read(size)
            if zlib.crc32(data) != crc:
                bad += 1
                continue
            Path(dest).write_bytes(data)
    return bad


def main(zip_path, index, split, out, workers=8):
    out = Path(out)
    sp = pd.read_parquet(split)
    idx = pd.read_parquet(index, columns=["zip_name", "offset"])
    sp = sp.merge(idx, on="zip_name", how="left", validate="one_to_one")
    (out / "classes.txt").parent.mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(sorted(sp["class_code"].unique())) + "\n")
    jobs = []
    for part, tag in PARTS.items():
        sub = sp[sp["pool"]] if tag is None else sp[sp["split"] == tag]
        dest = [str(out / part / c / Path(z).name) for c, z in zip(sub["class_code"], sub["zip_name"])]
        for c in sub["class_code"].unique():
            (out / part / c).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"path": dest, "label": sub["class_code"].to_numpy()}).to_csv(out / f"{part}.csv", index=False)
        rows = list(zip(dest, sub["offset"], sub["size"], sub["crc32"]))
        jobs += [(zip_path, rows[i:i + 2000]) for i in range(0, len(rows), 2000)]
        print(part, len(sub), flush=True)
    with Pool(int(workers)) as pool:
        bad = sum(pool.imap_unordered(_write, jobs))
    print(f"done, {bad} CRC failures")


if __name__ == "__main__":
    main(*sys.argv[1:6])
