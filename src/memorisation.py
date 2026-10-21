"""Memorisation and diversity check for synthetic snips. For each class and setting: DINOv2 cosine of
each synthetic snip to its nearest real snip in the generator's training rows, compared with the
same statistic for real held-out test snips (the natural level for a new event). Diversity is the
mean pairwise cosine within the synthetic set against within real test snips (lower is more diverse).
Anchor share is the fraction of snips whose nearest training snip is the single most common one.
Same-event bursts score about 0.90 or more (2026-10-21 calibration).
Usage: python src/memorisation.py IDENTITY_DIR INV_DIR SPLIT_NAME STAGE_A_RUN"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel

from common import DATA
from embed import MEAN, STD, MODEL


def embed_files(paths, model):
    out = []
    for i in range(0, len(paths), 64):
        xs = []
        for p in paths[i:i + 64]:
            with Image.open(p) as img:
                a = np.asarray(img.convert("RGB").resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255
            xs.append((torch.from_numpy(a).permute(2, 0, 1) - MEAN) / STD)
        with torch.inference_mode():
            z = model(pixel_values=torch.stack(xs).cuda().half()).pooler_output.float()
        out.append(torch.nn.functional.normalize(z, dim=1).cpu())
    return torch.cat(out)


def nn_sim(a, b):
    return (a @ b.T).max(dim=1).values.numpy()


def mean_pairwise(a):
    s = a @ a.T
    n = len(a)
    return float((s.sum() - n) / (n * (n - 1)))


def main(identity_dir, inv_dir, split_name, run):
    d, inv_dir = Path(identity_dir), Path(inv_dir)
    man = pd.read_parquet(d / "manifest.parquet")
    inv = pd.read_parquet(inv_dir / "inventory.parquet")
    emb = np.load(inv_dir / "dinov2_base.npy", mmap_mode="r")
    pos = pd.Series(np.arange(len(inv)), index=inv["relpath"])
    split = pd.read_parquet(inv_dir / "splits" / split_name / "split.parquet")
    rows = pd.read_parquet(DATA / "stage_a" / run / "rows.parquet")
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    gen = embed_files([d / r for r in man["relpath"]], model)
    rng = np.random.default_rng(0)
    out = []
    for (c, tag), g in man.groupby(["class_code", "tag"]):
        train = torch.from_numpy(np.asarray(emb[pos[rows.loc[rows["class_code"] == c, "relpath"]].values], dtype=np.float32))
        test_rel = split.loc[(split["class_code"] == c) & (split["split"] == "test"), "relpath"].values
        test_rel = rng.choice(test_rel, min(200, len(test_rel)), replace=False)
        test = torch.from_numpy(np.asarray(emb[pos[test_rel].values], dtype=np.float32))
        gz = gen[g.index.values]
        gs, ts = nn_sim(gz, train), nn_sim(test, train)
        anchor = float(np.bincount((gz @ train.T).argmax(dim=1).numpy()).max() / len(gz))
        anchor_test = float(np.bincount((test @ train.T).argmax(dim=1).numpy()).max() / len(test))
        out.append({"class": c, "setting": tag, "train_rows": len(train),
                    "gen_nn_median": round(float(np.median(gs)), 3), "test_nn_median": round(float(np.median(ts)), 3),
                    "gen_ge_0.95": round(float((gs >= 0.95).mean()), 3), "test_ge_0.95": round(float((ts >= 0.95).mean()), 3),
                    "gen_ge_0.90": round(float((gs >= 0.90).mean()), 3), "test_ge_0.90": round(float((ts >= 0.90).mean()), 3),
                    "gen_pairwise": round(mean_pairwise(gz), 3), "test_pairwise": round(mean_pairwise(test[:len(gz)]), 3),
                    "gen_anchor_share": round(anchor, 3), "test_anchor_share": round(anchor_test, 3)})
    table = pd.DataFrame(out)
    table.to_csv(d / "memorisation.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:5])
