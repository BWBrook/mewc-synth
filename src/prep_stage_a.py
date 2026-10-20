"""Build the Stage A training cache from the train split: a class-capped sample, captions,
SDXL VAE latents at 512 px (both horizontal orientations) and SDXL text embeddings per caption.
Pseudo-rare classes contribute only their reduced subset (whole groups, drawn with the seed).
Usage: python src/prep_stage_a.py SNIP_ROOT INV_DIR SPLIT_NAME CLASS_MAP_YAML RUN_NAME CONFIG_JSON
CONFIG_JSON example: {"cap": 1500, "seed": 7, "pseudo_rare": {"1": 50, "8": 50}, "ir_chroma": 4.0}"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import DATA, GEN_RES, SDXL, VAE_FP16


def caption(name, infrared):
    light = "infrared night" if infrared else "daylight colour"
    return f"camera trap snip of a {name.replace('_', ' ')}, {light}"


def take_groups(cdf, n, rng):
    """Draw whole groups in seeded order until n snips; trims the last group to hit n exactly."""
    members = cdf.groupby("group").indices
    picked = []
    for g in rng.permutation(sorted(members)):
        picked.extend(cdf.index[members[g]])
        if len(picked) >= n:
            break
    return picked[:n]


class Snips(Dataset):
    def __init__(self, root, relpaths):
        self.root, self.relpaths = Path(root), relpaths

    def __len__(self):
        return len(self.relpaths)

    def __getitem__(self, i):
        with Image.open(self.root / self.relpaths[i]) as img:
            a = np.asarray(img.convert("RGB").resize((GEN_RES, GEN_RES), Image.LANCZOS), dtype=np.float32)
        return torch.from_numpy(a / 127.5 - 1).permute(2, 0, 1)


def main(root, inv_dir, split_name, class_map, run, config):
    inv_dir, cfg = Path(inv_dir), json.loads(config)
    names = {str(k): v.strip() for k, v in yaml.safe_load(open(class_map)).items()}
    split = pd.read_parquet(inv_dir / "splits" / split_name / "split.parquet")
    meta = pd.read_parquet(inv_dir / "embed_meta.parquet")[["relpath", "chroma"]]
    df = split[split["split"] == "train"].merge(meta, on="relpath").reset_index(drop=True)
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for code, cdf in df.groupby("class_code"):
        n = int(cfg["pseudo_rare"].get(code, cfg["cap"]))
        rows.extend(take_groups(cdf, n, rng) if code in cfg["pseudo_rare"] else
                    (cdf.index if len(cdf) <= n else rng.choice(cdf.index, n, replace=False)))
    sel = df.loc[sorted(rows)].reset_index(drop=True)
    sel["infrared"] = sel["chroma"] < cfg["ir_chroma"]
    sel["caption"] = [caption(names[c], ir) for c, ir in zip(sel["class_code"], sel["infrared"])]
    captions = sorted(sel["caption"].unique())
    sel["caption_id"] = sel["caption"].map({c: i for i, c in enumerate(captions)})

    out = DATA / "stage_a" / run
    out.mkdir(parents=True, exist_ok=True)
    sel.drop(columns=["caption"]).to_parquet(out / "rows.parquet", index=False)
    (out / "captions.json").write_text(json.dumps(captions, indent=1))
    (out / "config.json").write_text(json.dumps({**cfg, "split": split_name, "rows": len(sel)}, indent=1))

    pipe = StableDiffusionXLPipeline.from_pretrained(SDXL, vae=None, unet=None, torch_dtype=torch.float16, variant="fp16").to("cuda")
    with torch.inference_mode():
        pe, _, pooled, _ = pipe.encode_prompt(captions, device="cuda", do_classifier_free_guidance=False)
    torch.save({"prompt_embeds": pe.cpu(), "pooled": pooled.cpu()}, out / "text_embeds.pt")
    del pipe

    vae = AutoencoderKL.from_pretrained(VAE_FP16, torch_dtype=torch.float16).cuda().eval()
    sf = vae.config.scaling_factor
    lat = np.lib.format.open_memmap(out / "latents.npy", mode="w+", dtype=np.float16,
                                    shape=(len(sel), 2, 4, GEN_RES // 8, GEN_RES // 8))
    loader = DataLoader(Snips(root, sel["relpath"].tolist()), batch_size=16, num_workers=32, pin_memory=True)
    k = 0
    with torch.inference_mode():
        for x in loader:
            x = x.cuda(non_blocking=True).half()
            both = torch.cat([x, torch.flip(x, dims=[3])])
            z = vae.encode(both).latent_dist.mean * sf
            n = len(x)
            lat[k:k + n] = torch.stack([z[:n], z[n:]], dim=1).cpu().numpy()
            k += n
    lat.flush()
    print(json.dumps({"run": run, "rows": len(sel), "classes": int(sel["class_code"].nunique()),
                      "captions": len(captions), "infrared_share": round(float(sel["infrared"].mean()), 3),
                      "latents": str(out / "latents.npy")}))


if __name__ == "__main__":
    main(*sys.argv[1:7])
