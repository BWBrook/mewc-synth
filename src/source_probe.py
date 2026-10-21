"""Learn photo (eBird/iNaturalist) versus camera-trap style from DINOv2 embeddings and score every snip.
Labels come from filename prefixes: PHOTO prefixes vs known CAMERA prefixes. Unprefixed (UUID) snips
and unidentified prefixes are scored, not trained on. Training is balanced within classes that have
both sources, so the probe learns style rather than taxon. It is validated by GroupKFold over
classes (held-out species).
Usage: python src/source_probe.py INV_DIR"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import source_prefix

PHOTO = {"ebird", "inat"}
CAMERA = {"iona", "jl", "tlc", "thi", "ink", "nw", "gkvic", "gst", "hr", "lun", "deeca", "delwp",
          "goa", "nret", "kpz", "vish", "evd", "alex", "js", "felix"}
PER_CLASS = 600


def main(inv_dir):
    inv_dir = Path(inv_dir)
    inv = pd.read_parquet(inv_dir / "inventory.parquet")
    emb = np.load(inv_dir / "dinov2_base.npy", mmap_mode="r")
    inv["src"] = inv["relpath"].map(source_prefix)
    inv["label"] = np.where(inv["src"].isin(PHOTO), 1, np.where(inv["src"].isin(CAMERA), 0, -1))
    rng = np.random.default_rng(5)
    picks = []
    for cls, cdf in inv[inv["label"] >= 0].groupby("class_code"):
        if cdf["label"].nunique() < 2:
            continue
        k = min(PER_CLASS, cdf["label"].value_counts().min())
        for lab in (0, 1):
            idx = cdf.index[cdf["label"] == lab]
            picks.extend(rng.choice(idx, k, replace=False))
    tr = inv.loc[sorted(picks)]
    X, y, groups = np.asarray(emb[tr.index], dtype=np.float32), tr["label"].to_numpy(), tr["class_code"].to_numpy()
    model = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    oof = np.zeros(len(y))
    for a, b in GroupKFold(n_splits=5).split(X, y, groups):
        oof[b] = model().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    cv = {"train_snips": int(len(y)), "classes": int(len(set(groups))),
          "heldout_species_auc": round(float(roc_auc_score(y, oof)), 4),
          "heldout_species_acc_at_0.5": round(float(((oof > 0.5) == y).mean()), 4)}
    final = model().fit(X, y)
    p = np.empty(len(inv), np.float32)
    for s in range(0, len(inv), 200_000):
        p[s:s + 200_000] = final.predict_proba(np.asarray(emb[s:s + 200_000], dtype=np.float32))[:, 1]
    inv["p_photo"] = p
    inv[["relpath", "class_code", "src", "label", "p_photo"]].to_parquet(inv_dir / "source_probe.parquet", index=False)
    by_src = inv.groupby("src").agg(n=("relpath", "size"), mean_p=("p_photo", "mean"),
                                    share_gt_0_5=("p_photo", lambda v: (v > 0.5).mean())).round(3)
    by_src = by_src[by_src["n"] >= 100].sort_values("n", ascending=False)
    summary = {"cv": cv, "by_source": by_src.reset_index().to_dict("records")}
    (inv_dir / "source_probe_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(cv))
    print(by_src.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
