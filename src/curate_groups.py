"""Group near-duplicate snips across the full library, per class, from curate_embed.py output.
An edge joins two snips of one class if their DINOv2 cosine is at least TAU; groups are connected
components. Groups are reported at 0.90 (calibrated on 2026-10-21) and at 0.95, with per-class
redundancy statistics: groups, singletons, share of snips in multi-snip groups, snips kept at one or
three per group, largest group, and nearest-neighbour similarity quantiles.
Usage: python src/curate_groups.py CURATION_DIR [DEVICE]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CHUNK = 4096
TAUS = (0.90, 0.95)


def class_edges(z, tau):
    n = len(z)
    rows, cols, sims, nn = [], [], [], np.zeros(n, np.float32)
    for a in range(0, n, CHUNK):
        s = z[a:a + CHUNK] @ z.T
        ar = torch.arange(len(s), device=z.device)
        s[ar, ar + a] = -1
        nn[a:a + CHUNK] = s.max(dim=1).values.float().cpu().numpy()
        r, c = torch.nonzero(s >= tau, as_tuple=True)
        upper = c > r + a
        r, c = r[upper], c[upper]
        rows.append((r + a).int().cpu().numpy()); cols.append(c.int().cpu().numpy())
        sims.append(s[r, c].float().cpu().numpy())
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(sims), nn


def components(n, r, c):
    return connected_components(coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(n, n)), directed=False)[1]


def main(curation_dir, device="cuda"):
    d = Path(curation_dir)
    idx = pd.read_parquet(d / "index.parquet")
    shards = sorted((d / "shards").glob("emb_*.npy"))
    emb = np.concatenate([np.load(p) for p in shards])
    ok = np.concatenate([np.load(d / "shards" / p.name.replace("emb_", "ok_")) for p in shards])
    assert len(emb) == len(idx), (len(emb), len(idx))
    groups = {t: np.empty(len(idx), dtype=object) for t in TAUS}
    nn_all, stats = np.zeros(len(idx), np.float32), []
    for cls, rows_idx in idx.groupby("class_code").indices.items():
        z = torch.from_numpy(emb[rows_idx]).to(device)
        z = z.half() if device == "cuda" else z.float()
        r, c, s, nn = class_edges(z, min(TAUS))
        nn_all[rows_idx] = nn
        st = {"class": cls, "tier": idx["tier"].iat[rows_idx[0]], "snips": len(rows_idx),
              "bad_decode": int((~ok[rows_idx]).sum()),
              "nn_median": round(float(np.median(nn)), 3), "nn_q90": round(float(np.quantile(nn, 0.9)), 3)}
        for t in TAUS:
            keep = s >= t
            lab = components(len(rows_idx), r[keep], c[keep])
            groups[t][rows_idx] = [f"{cls}:{g}" for g in lab]
            sizes = np.bincount(lab)
            k = int(100 * t)
            st.update({f"groups_{k}": int(len(sizes)), f"singletons_{k}": int((sizes == 1).sum()),
                       f"in_multi_{k}": round(float(1 - (sizes == 1).sum() / len(lab)), 3),
                       f"keep1_{k}": int(len(sizes)), f"keep3_{k}": int(np.minimum(sizes, 3).sum()),
                       f"largest_{k}": int(sizes.max())})
        stats.append(st)
        print(json.dumps(st), flush=True)
    out = idx[["zip_name", "tier", "class_code"]].copy()
    out["nn_sim"] = nn_all
    for t in TAUS:
        out[f"group_{int(100 * t)}"] = groups[t]
    out.to_parquet(d / "groups.parquet", index=False)
    st = pd.DataFrame(stats).sort_values("snips", ascending=False)
    st.to_csv(d / "redundancy_by_class.csv", index=False)
    tot = {"snips": int(st["snips"].sum()), "bad_decode": int(st["bad_decode"].sum())}
    for t in TAUS:
        k = int(100 * t)
        tot.update({f"groups_{k}": int(st[f"groups_{k}"].sum()), f"keep3_{k}": int(st[f"keep3_{k}"].sum())})
    (d / "redundancy_summary.json").write_text(json.dumps(tot, indent=1))
    print(json.dumps(tot))


if __name__ == "__main__":
    main(*sys.argv[1:3])
