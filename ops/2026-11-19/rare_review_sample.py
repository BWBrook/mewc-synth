"""Sample Barry's blind review of the real low-count translations (docs/plan-booking-3.md, Q1; "Yes, send the
100 when ready then", 13 November): per class in the profile, 2 translations plus 1 real camera-trap train snip
of the class and 1 of its look-alike, none of them a translation source; the 31 October audit's method
(ops/2026-10-31/synth_review_sample.py). Every snip gets the same thumbnail path (448 px, JPEG q85).
Usage: python ops/2026-11-19/rare_review_sample.py TRANSLATE_DIR V2_DIR PROFILE_CSV OUT_JSON OUT_CSV SEED"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

N_SYN, N_REAL = 2, 1


def thumb(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((448, 448), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=85)
    return base64.b64encode(b.getvalue()).decode()


def main(tdir, v2, profile, out_json, out_csv, seed):
    PAIRS = dict(pd.read_csv(profile)[["class", "lookalike_1"]].values)
    tdir, rng = Path(tdir), np.random.default_rng(int(seed))
    man = pd.read_parquet(tdir / "manifest.parquet")
    sources = set(man.source_relpath)
    train = pd.read_csv(Path(v2) / "train.csv")
    train["rel"] = train.path.str.split("/").str[-3:].str.join("/")
    camera = ~train.path.str.split("/").str[-1].str.match(r"(ebird|inat)_")
    rows = []
    for block, (target, look) in enumerate(PAIRS.items(), 1):
        syn = man[man.class_code == target].sample(N_SYN, random_state=int(seed))
        rows += [{"kind": "synthetic", "species": target, "path": str(tdir / r), "source": s} for r, s in zip(syn.relpath, syn.source_relpath)]
        for cls, n in [(target, N_REAL), (look, N_REAL)]:
            pool = train[(train.label == cls) & camera & ~train.rel.isin(sources)]
            rows += [{"kind": "real", "species": cls, "path": p, "source": ""} for p in pool.path.sample(n, random_state=int(seed))]
        for r in rows[-(N_SYN + 2 * N_REAL):]:
            r["block"], r["target"], r["lookalike"] = block, target, look
    items = pd.DataFrame(rows)
    items = items.sample(frac=1, random_state=int(seed)).sort_values("block", kind="stable")
    items["id"] = [f"rr{i:03d}" for i in range(1, len(items) + 1)]
    flip = rng.random(len(items)) < 0.5
    items["a"] = np.where(flip, items.lookalike, items.target)
    items["b"] = np.where(flip, items.target, items.lookalike)
    items[["id", "block", "kind", "species", "target", "lookalike", "a", "b", "path", "source"]].to_csv(out_csv, index=False)
    items["img"] = [thumb(p) for p in items.path]
    json.dump(items[["id", "block", "a", "b", "img"]].to_dict("records"), open(out_json, "w"))
    print(len(items), "items;", items.groupby(["kind"]).size().to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:7])
