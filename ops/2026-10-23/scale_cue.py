import sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path.home() / "mewc-synth/src"))
from common import source_prefix
M = Path("/mnt/ml-data/mewc-synth"); snips = M / "work/snips"
split = pd.read_parquet(M / "work/audit/splits/split_v1/split.parquet")
meta = pd.read_parquet(M / "work/audit/embed_meta.parquet")[["relpath", "chroma"]]
s = split[split.split == "train"].merge(meta, on="relpath")
s = s[~s.relpath.map(source_prefix).isin(["ebird", "inat"])]
def feats(rel):
    p = snips / rel
    g = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    F = np.abs(np.fft.rfft2(g - g.mean())) ** 2
    fy = np.fft.fftfreq(g.shape[0])[:, None]; fx = np.fft.rfftfreq(g.shape[1])[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    hf = F[r > 0.25].sum() / F[r > 0.03].sum()
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return hf, lap.var(), p.stat().st_size / 1024
rng = np.random.default_rng(0)
rows = []
for c in ["house_mouse", "black_rat", "swamp_rat", "long_tailed_mouse", "rakali"]:
    sub = s[s.class_code == c]
    sub = sub.iloc[rng.choice(len(sub), min(1500, len(sub)), replace=False)]
    for r in sub.itertuples():
        hf, lv, kb = feats(r.relpath)
        rows.append((c, r.chroma < 1.0, hf, lv, kb))
d = pd.DataFrame(rows, columns=["cls", "ir", "hf_ratio", "lap_var", "jpeg_kb"])
print(d.groupby("cls")[["hf_ratio", "lap_var", "jpeg_kb"]].median().round(4).to_string())
def auc(a, b, ir=None):
    x = d[d.cls.isin([a, b])] if ir is None else d[d.cls.isin([a, b]) & (d.ir == ir)]
    y = (x.cls == b).to_numpy()
    return {m: round(max(roc_auc_score(y, x[m]), 1 - roc_auc_score(y, x[m])), 3) for m in ["hf_ratio", "lap_var", "jpeg_kb"]} | {"n": len(x)}
for a, b in [("house_mouse", "black_rat"), ("house_mouse", "swamp_rat"), ("black_rat", "swamp_rat"), ("house_mouse", "long_tailed_mouse")]:
    print(a, "vs", b, "all", auc(a, b), "| IR only", auc(a, b, True), "| colour only", auc(a, b, False))
print("IR share:", d.groupby("cls").ir.mean().round(2).to_dict())
