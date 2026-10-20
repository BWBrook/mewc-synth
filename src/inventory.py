"""Inventory a snip library laid out as <root>/<class_code>/*.jpg.
Records size, mode, estimated JPEG quality and SHA-256 per file, plus a per-class summary.
Usage: python src/inventory.py SNIP_ROOT OUT_DIR"""
import hashlib
import io
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image

from common import jpeg_quality


def scan(path):
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as img:
        return {"relpath": f"{path.parent.name}/{path.name}", "class_code": path.parent.name,
                "bytes": len(data), "width": img.width, "height": img.height, "mode": img.mode,
                "format": img.format, "jpeg_quality": jpeg_quality(img),
                "sha256": hashlib.sha256(data).hexdigest()}


def main(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.glob("*/*") if p.suffix.lower() in {".jpg", ".jpeg"})
    with Pool(48) as pool:
        rows = pool.map(scan, files, chunksize=256)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "inventory.parquet", index=False)
    per_class = df.groupby("class_code").agg(n=("relpath", "size"), mb=("bytes", lambda b: round(b.sum() / 1e6, 1)))
    summary = {
        "files": len(df), "classes": df["class_code"].nunique(), "total_gb": round(df["bytes"].sum() / 1e9, 2),
        "sizes": df.groupby(["width", "height"]).size().sort_values(ascending=False).head(5).to_dict(),
        "modes": df["mode"].value_counts().to_dict(),
        "jpeg_quality": df["jpeg_quality"].value_counts().head(5).to_dict(),
        "duplicate_sha256": int(df["sha256"].duplicated().sum()),
        "class_n_min_median_max": [int(per_class["n"].min()), int(per_class["n"].median()), int(per_class["n"].max())],
    }
    summary["sizes"] = {f"{w}x{h}": int(n) for (w, h), n in summary["sizes"].items()}
    per_class.sort_values("n").to_csv(out / "class_counts.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=int))
    print(json.dumps(summary, default=int))


if __name__ == "__main__":
    main(*sys.argv[1:3])
