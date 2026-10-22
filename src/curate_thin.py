"""Chaining-free thinning of the full library, per class: visit snips in a seeded random order and keep
one only if no snip already kept has DINOv2 cosine at or above TAU. Unlike connected components this
never merges distinct events through chains, and the kept set covers every dropped snip within TAU.
Blockwise on the GPU: each block is checked against the kept set by matrix product, then thinned
greedily within itself. Writes keep flags per threshold and per-class counts.
Usage: python src/curate_thin.py CURATION_DIR [DEVICE]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BLOCK = 4096
TAUS = (0.90, 0.95)
SEED = 20261022


def thin(z, tau, rng):
    order = rng.permutation(len(z))
    kept = torch.empty(0, z.shape[1], dtype=z.dtype, device=z.device)
    keep = np.zeros(len(z), bool)
    for a in range(0, len(order), BLOCK):
        idx = order[a:a + BLOCK]
        zb = z[idx]
        if len(kept):
            covered = torch.zeros(len(idx), dtype=torch.bool, device=z.device)
            for k in range(0, len(kept), 65536):
                covered |= (zb @ kept[k:k + 65536].T).max(dim=1).values >= tau
            idx, zb = idx[~covered.cpu().numpy()], zb[~covered]
        if not len(idx):
            continue
        near = (zb @ zb.T >= tau).cpu().numpy()
        free = np.ones(len(idx), bool)
        sel = np.zeros(len(idx), bool)
        for i in range(len(idx)):
            if free[i]:
                sel[i] = True
                free &= ~near[i]
        keep[idx[sel]] = True
        kept = torch.cat([kept, zb[torch.from_numpy(sel).to(z.device)]])
    return keep


def main(curation_dir, device="cuda"):
    d = Path(curation_dir)
    g = pd.read_parquet(d / "groups.parquet", columns=["zip_name", "class_code"])
    emb = np.concatenate([np.load(p) for p in sorted((d / "shards").glob("emb_*.npy"))])
    assert len(emb) == len(g)
    flags = {t: np.zeros(len(g), bool) for t in TAUS}
    rows = []
    for cls, idx in g.groupby("class_code").indices.items():
        z = torch.from_numpy(emb[idx]).to(device)
        z = z.half() if device == "cuda" else z.float()
        st = {"class": cls, "snips": len(idx)}
        for t in TAUS:
            keep = thin(z, t, np.random.default_rng(SEED))
            flags[t][idx] = keep
            st[f"kept_{int(100 * t)}"] = int(keep.sum())
        rows.append(st)
        print(json.dumps(st), flush=True)
    out = g[["zip_name"]].copy()
    for t in TAUS:
        out[f"keep_{int(100 * t)}"] = flags[t]
    out.to_parquet(d / "thin.parquet", index=False)
    st = pd.DataFrame(rows).sort_values("snips", ascending=False)
    st.to_csv(d / "thin_by_class.csv", index=False)
    print(json.dumps({"snips": int(st["snips"].sum()), **{f"kept_{int(100 * t)}": int(st[f"kept_{int(100 * t)}"].sum()) for t in TAUS}}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
