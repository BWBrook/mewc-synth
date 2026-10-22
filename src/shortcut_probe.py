"""Shortcut probe (PLAN quality check 4): can a small CNN tell synthetic snips from real ones using only
high-frequency content? Each translated snip is paired with its real source snip (same pose, framing and
lighting), so content is matched and only the generator's processing differs. Both go through the
classifier's input path (stored 600 px JPEG, bilinear resize to 384 without antialiasing, as
tf.image.resize does), then a high-pass residual (image minus a Gaussian blur, sigma 2). Five-fold
cross-validation grouped by pair, training on random 192 px crops; reports held-out AUC on full images
per fold and per class. AUC near 0.5 means no usable fingerprint.
Modes separate content from processing:
  pair    translated snips vs their real source snips (content differs: the species changes);
  within  translated snips vs real train-split snips of the same target class (the shortcut the
          classifier could use within one label);
  vae     real source snips vs the same snips sent through 512 px, the SDXL VAE and finish_snip
          (vae768 and vae1024 decode at those sizes instead);
  resize  real source snips vs the same snips resized 600 -> 512 -> 600 and re-saved as JPEG q75;
  vaedetail  as vae, then detail transfer: the round trip's low frequencies plus the source's high
          frequencies (Gaussian sigma DETAIL_SIGMA at 600 px), re-saved as JPEG q75.
A mode ending in "_full" skips the high-pass filter, so the probe sees every frequency; use it on
content-matched modes (vae, vaedetail, resize).
Usage: python src/shortcut_probe.py SYNTH_DIR [MODE] [EPOCHS]"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from common import DATA, GEN_RES, VAE_FP16, finish_snip

SIZE, CROP, SIGMA, DETAIL_SIGMA = 384, 192, 2.0, 2.0


def load(src):
    a = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255
    x = torch.from_numpy(a).permute(2, 0, 1)[None]
    return F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False, antialias=False)[0]


def blur(x, sigma):
    r = int(3 * sigma)
    g = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).to(x.device)
    c = x.shape[1]
    b = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), g.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(b, (0, 0, r, r), mode="reflect"), g.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def high_pass(x):
    return (x - blur(x, SIGMA)) * 10


def detail_transfer(gen, src):
    """Low frequencies of gen plus high frequencies of src, both PIL images at the same size."""
    t = lambda im: torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)[None]
    g, s = t(gen), t(src)
    out = (blur(g, DETAIL_SIGMA) + s - blur(s, DETAIL_SIGMA)).clamp(0, 1)[0]
    return Image.fromarray((out.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def net():
    return nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                         nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))


def processed(paths, mode):
    """JPEG bytes of real snips after the generation path without diffusion."""
    res = int(mode[3:]) if mode in ("vae768", "vae1024") else GEN_RES
    imgs = [Image.open(p).convert("RGB").resize((res, res), Image.LANCZOS) for p in paths]
    if mode.startswith("vae"):
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(VAE_FP16, torch_dtype=torch.bfloat16).to("cuda")
        out = []
        with torch.no_grad():
            for im in imgs:
                t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
                z = vae.encode(t.to("cuda", torch.bfloat16)).latent_dist.mode()
                r = vae.decode(z).sample[0].float().clamp(-1, 1)
                out.append(Image.fromarray(((r.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)))
        imgs = out
    data = [io.BytesIO(finish_snip(im)) for im in imgs]
    if mode == "vaedetail":
        data = [io.BytesIO(_jpeg(detail_transfer(Image.open(b).convert("RGB"), Image.open(p).convert("RGB"))))
                for b, p in zip(data, paths)]
    return data


def _jpeg(img, quality=75):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def main(synth_dir, mode="pair", epochs=15):
    full = mode.endswith("_full")
    mode = mode.removesuffix("_full")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = Path(synth_dir)
    man = pd.read_parquet(d / "manifest.parquet").reset_index(drop=True)
    snips = DATA / "work" / "snips"
    sources = [snips / r for r in man["source_relpath"]]
    if mode == "pair":
        a, b, pair = sources, [d / r for r in man["relpath"]], np.arange(len(man))
    elif mode == "within":
        split = pd.read_parquet(DATA / "work" / "audit" / "splits" / "split_v1" / "split.parquet")
        tr = split[split["split"] == "train"]
        real = pd.concat([tr[tr["class_code"] == c].sample(n, random_state=0)
                          for c, n in man["class_code"].value_counts().sort_index().items()])
        man = man.sort_values("class_code").reset_index(drop=True)
        a, b, pair = [snips / r for r in real["relpath"]], [d / r for r in man["relpath"]], np.arange(len(man))
    elif mode in ("vae", "vae768", "vae1024", "vaedetail", "resize"):
        a, b, pair = sources, processed(sources, mode), np.arange(len(man))
    else:
        raise ValueError(mode)
    x = torch.stack([load(p) for p in list(a) + list(b)])
    x = torch.cat([(c * 4 if full else high_pass(c.to(dev))).half().cpu() for c in x.split(256)])
    y = torch.cat([torch.zeros(len(a)), torch.ones(len(b))]).to(dev)
    pair = np.concatenate([pair, pair])
    cls = np.concatenate([man["class_code"].to_numpy()] * 2)
    scores = np.zeros(len(y))
    for k, (tr, te) in enumerate(GroupKFold(5).split(np.zeros(len(y)), groups=pair)):
        m = net().to(dev)
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        for _ in range(int(epochs)):
            for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 32)):
                i, j = rng.integers(0, SIZE - CROP, 2)
                xb = x[b][:, :, i:i + CROP, j:j + CROP].to(dev).float()
                if rng.random() < 0.5:
                    xb = xb.flip(3)
                loss = F.binary_cross_entropy_with_logits(m(xb)[:, 0], y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            scores[te] = torch.cat([m(x[te[s:s + 64]].to(dev).float())[:, 0] for s in range(0, len(te), 64)]).cpu().numpy()
        print(json.dumps({"fold": k, "auc": round(roc_auc_score(y[te].cpu().numpy(), scores[te]), 3)}), flush=True)
    yt = y.cpu().numpy()
    mode = mode + ("_full" if full else "")
    out = {"mode": mode, "pairs": len(man), "auc": round(roc_auc_score(yt, scores), 3),
           "auc_by_class": {c: round(roc_auc_score(yt[cls == c], scores[cls == c]), 3) for c in sorted(set(cls))}}
    (d / f"shortcut_probe_{mode}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main(*sys.argv[1:4])
