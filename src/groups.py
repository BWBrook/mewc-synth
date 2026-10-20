"""Group near-duplicate snips (burst frames of one event) within each class.
An edge joins two snips if DINOv2 cosine >= TAU or perceptual-hash Hamming distance <= HAM;
groups are connected components. Also reports nearest-neighbour similarity quantiles for
calibrating TAU. Usage: python src/groups.py INV_DIR [TAU] [HAM]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CHUNK = 4096


def hash_signs(phash):
    bits = np.unpackbits(phash.astype(np.int64).view(np.uint8).reshape(-1, 8), axis=1)
    return torch.from_numpy(bits.astype(np.float16) * 2 - 1)


def class_groups(z, s, tau, ham):
    n = len(z)
    rows, cols, nn = [], [], np.zeros(n, np.float32)
    for a in range(0, n, CHUNK):
        sim = z[a:a + CHUNK] @ z.T
        hd = (64 - s[a:a + CHUNK] @ s.T) / 2
        idx = torch.arange(a, min(a + CHUNK, n), device=z.device)
        sim[torch.arange(len(idx)), idx] = -1
        nn[a:a + CHUNK] = sim.max(dim=1).values.float().cpu().numpy()
        hd[torch.arange(len(idx)), idx] = 65
        r, c = torch.nonzero((sim >= tau) | (hd <= ham), as_tuple=True)
        rows.append((r + a).cpu().numpy())
        cols.append(c.cpu().numpy())
    r, c = np.concatenate(rows), np.concatenate(cols)
    _, labels = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)), directed=False)
    return labels, nn


def main(inv_dir, tau=0.90, ham=8):
    inv_dir, tau, ham = Path(inv_dir), float(tau), int(ham)
    inv = pd.read_parquet(inv_dir / "inventory.parquet")
    meta = pd.read_parquet(inv_dir / "embed_meta.parquet")
    assert (inv["relpath"].values == meta["relpath"].values).all()
    emb = np.load(inv_dir / "dinov2_base.npy", mmap_mode="r")
    group = np.empty(len(inv), dtype=object)
    nn_sim = np.empty(len(inv), np.float32)
    for code, idx in inv.groupby("class_code").indices.items():
        z = torch.from_numpy(np.asarray(emb[idx])).cuda()
        s = hash_signs(meta["phash"].values[idx]).cuda()
        labels, nn = class_groups(z, s, tau, ham)
        group[idx] = [f"{code}:{g}" for g in labels]
        nn_sim[idx] = nn
    out = pd.DataFrame({"relpath": inv["relpath"], "class_code": inv["class_code"], "group": group, "nn_sim": nn_sim})
    out.to_parquet(inv_dir / "groups.parquet", index=False)
    sizes = out.groupby("group").size()
    summary = {"tau": tau, "ham": ham, "snips": len(out), "groups": len(sizes),
               "singletons": int((sizes == 1).sum()), "largest_group": int(sizes.max()),
               "nn_sim_quantiles": {q: round(float(np.quantile(nn_sim, q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}}
    (inv_dir / "groups_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main(*sys.argv[1:4])
