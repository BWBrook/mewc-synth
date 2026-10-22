"""Contact sheets for checking full-library grouping by eye. For each chosen class: the largest groups
at cosine 0.90 (up to 8 random members per row, labelled with group size), and snip pairs whose
nearest-neighbour cosine sits just above each threshold (0.90-0.91 and 0.95-0.96), side by side.
Classes default to the six with the largest 0.90 component.
Usage: python src/curate_sheets.py CURATION_DIR ZIP [CLASS ...]"""
import struct
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

THUMB, PER_ROW, ROWS, PAIRS = 160, 8, 6, 12
BANDS = ((0.90, 0.91), (0.95, 0.96))


def read_snip(f, offset, size):
    f.seek(offset)
    n, e = struct.unpack("<HH", f.read(30)[26:30])
    f.seek(offset + 30 + n + e)
    return Image.open(BytesIO(f.read(size))).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)


def sheet(rows, f, idx, path):
    w = max(len(r[1]) for r in rows) * THUMB + 90
    img = Image.new("RGB", (w, len(rows) * THUMB), "white")
    draw = ImageDraw.Draw(img)
    for y, (label, members) in enumerate(rows):
        draw.text((4, y * THUMB + THUMB // 2), label, fill="black")
        for x, i in enumerate(members):
            img.paste(read_snip(f, idx.at[i, "offset"], idx.at[i, "size"]), (90 + x * THUMB, y * THUMB))
    img.save(path, quality=85)


def main(curation_dir, zip_path, *classes):
    d = Path(curation_dir)
    idx = pd.read_parquet(d / "index.parquet")
    g = pd.read_parquet(d / "groups.parquet")
    assert (g["zip_name"].to_numpy() == idx["zip_name"].to_numpy()).all()
    if not classes:
        st = pd.read_csv(d / "redundancy_by_class.csv")
        classes = st.nlargest(6, "largest_90")["class"].tolist()
    shards = sorted((d / "shards").glob("emb_*.npy"))
    emb = np.concatenate([np.load(p, mmap_mode="r") for p in shards])
    out = d / "sheets"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    with open(zip_path, "rb") as f:
        for cls in classes:
            rows_idx = np.flatnonzero(g["class_code"].to_numpy() == cls)
            sizes = g.iloc[rows_idx]["group_90"].value_counts()
            giant_rows = []
            for gid, n in sizes.head(ROWS).items():
                members = rows_idx[g.iloc[rows_idx]["group_90"].to_numpy() == gid]
                giant_rows.append((f"n={n}", rng.choice(members, min(PER_ROW, n), replace=False)))
            sheet(giant_rows, f, idx, out / f"giants_{cls}.jpg")
            z = emb[rows_idx].astype(np.float32)
            nn = g.iloc[rows_idx]["nn_sim"].to_numpy()
            pair_rows = []
            for lo, hi in BANDS:
                cand = np.flatnonzero((nn >= lo) & (nn < hi))
                for q in rng.choice(cand, min(PAIRS // len(BANDS), len(cand)), replace=False):
                    s = z @ z[q]
                    s[q] = -1
                    pair_rows.append((f"{s.max():.3f}", [rows_idx[q], rows_idx[int(s.argmax())]]))
            if pair_rows:
                sheet(pair_rows, f, idx, out / f"margin_{cls}.jpg")
            print(cls, len(giant_rows), "giant rows,", len(pair_rows), "margin pairs", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
