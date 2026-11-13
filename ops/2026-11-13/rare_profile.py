"""Profile the 25 real low-count classes (in-model need_more and too_few) for synthetic generation: train counts
by source kind (camera trap or eBird/iNaturalist photo), infrared share (chroma < 1, as translate.py), JPEG q95
share, and look-alike candidates. A look-alike is an in-model class with at least 2,000 camera-trap train snips,
ranked by the mean probability that classifiers give it on the low-count class's validation snips, averaged
over every saved run scored on the current validation list that trained on the full real pool for that class
(v2_runs/scores/*_probs.npy).
Usage: python ops/2026-11-13/rare_profile.py MEWC_SYNTH_DIR OUT_CSV"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import source_prefix  # noqa: E402


def quality95(path):
    q = Image.open(path).quantization
    return float(np.mean(q[0])) < 5


def main(root, out):
    root = Path(root)
    ct = pd.read_csv(root / "work/snips_meta/class_table.csv")
    low = ct[ct.in_model & ct.tier.isin(["need_more", "too_few"])]["class"].tolist()
    split = pd.read_parquet(root / "work/audit_v2/splits/split_v2/split.parquet")
    meta = pd.read_parquet(root / "work/audit_v2/embed_meta.parquet")[["relpath", "chroma"]]
    tr = split[split.split == "train"].merge(meta, on="relpath")
    tr["photo"] = tr.relpath.map(source_prefix).isin(["ebird", "inat"])
    cam = tr[~tr.photo].groupby("class_code").size()
    classes = sorted((root / "v2_runs/classes.txt").read_text().split())
    v = pd.read_csv(root / "v2/val.csv")
    files = [f for f in sorted((root / "v2_runs/scores").glob("*_probs.npy"))
             if not f.name.startswith(("rt_", "ss_"))]
    arrays = [a for a in map(np.load, files) if a.shape == (len(v), len(classes))]
    probs = np.mean(arrays, axis=0)
    eligible = [c for c in classes if c not in low and cam.get(c, 0) >= 2000]
    rows = []
    for c in low:
        t = tr[tr.class_code == c]
        camt = t[~t.photo]
        m = probs[(v.label == c).to_numpy()].mean(axis=0)
        cand = sorted(eligible, key=lambda k: -m[classes.index(k)])[:3]
        rows.append({"class": c, "train": len(t), "train_camera": len(camt), "train_photo": int(t.photo.sum()),
                     "ir_share_camera": round(float((camt.chroma < 1.0).mean()), 3) if len(camt) else np.nan,
                     "q95_share": round(float(np.mean([quality95(root / "v2" / r) for r in t.relpath])), 3),
                     "val": int((v.label == c).sum()), "self_prob": round(float(m[classes.index(c)]), 3),
                     **{f"lookalike_{i + 1}": k for i, k in enumerate(cand)},
                     **{f"p_{i + 1}": round(float(m[classes.index(k)]), 4) for i, k in enumerate(cand)}})
    t = pd.DataFrame(rows).sort_values("train")
    t.to_csv(out, index=False)
    print(len(arrays), "runs averaged")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:3])
