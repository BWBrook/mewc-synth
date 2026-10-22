"""Translate real look-alike snips into a target class with SDXL img2img and a Stage B LoRA (Diff-Mix style):
the real snip's pose, framing, background and lighting carry over and the adapter supplies the target's
features. Sources are drawn from the look-alike's train split (camera-trap sources only); each output is
finished like a real snip and gets a manifest row with its source.
With "source_kind": "photo" the sources are the train split's eBird and iNaturalist photos instead (use
source = target and the domain adapter to restyle a class's own photos as camera-trap snips); captions are
infrared with probability "ir_share" (default 0) since photos carry no infrared state.
If n exceeds the source pool, sources are drawn with replacement (each repeat gets its own seed).
Usage: python src/translate.py LORA_DIR CLASS_TABLE_CSV INV_DIR SPLIT_NAME OUT_DIR CONFIG_JSON
CONFIG_JSON example: {"source": "eastern_quoll", "target": "spotted_tail_quoll", "n": 48,
  "strengths": [0.5, 0.65, 0.8], "guidance": 3.5, "steps": 30, "seed": 31, "ir_chroma": 1.0, "q95_share": 0.09}"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline
from PIL import Image

from common import GEN_RES, SDXL, VAE_FP16, finish_snip, source_prefix
from prep_stage_a import caption, class_names


def main(lora_dir, class_table, inv_dir, split_name, out_dir, config):
    cfg, out, inv_dir = json.loads(config), Path(out_dir), Path(inv_dir)
    names = class_names(class_table)
    split = pd.read_parquet(inv_dir / "splits" / split_name / "split.parquet")
    meta = pd.read_parquet(inv_dir / "embed_meta.parquet")[["relpath", "chroma"]]
    pool = split[(split["class_code"] == cfg["source"]) & (split["split"] == "train")].merge(meta, on="relpath")
    photo = pool["relpath"].map(source_prefix).isin(["ebird", "inat"])
    pool = pool[photo] if cfg.get("source_kind") == "photo" else pool[~photo]
    rng = np.random.default_rng(cfg["seed"])
    src = pool.iloc[rng.choice(len(pool), cfg["n"], replace=cfg["n"] > len(pool))]
    weights = Path(lora_dir) / "pytorch_lora_weights.safetensors"
    lora_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    vae = AutoencoderKL.from_pretrained(VAE_FP16, torch_dtype=torch.bfloat16)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(SDXL, vae=vae, torch_dtype=torch.bfloat16, variant="fp16").to("cuda")
    pipe.load_lora_weights(str(lora_dir))
    pipe.set_progress_bar_config(disable=True)
    snip_root = inv_dir.parent / "snips"
    target = cfg["target"]
    (out / target).mkdir(parents=True, exist_ok=True)
    rows = []
    for s in cfg["strengths"]:
        kind = "photo_" if cfg.get("source_kind") == "photo" else ""
        tag = f"tr_{target}_from_{kind}{cfg['source']}_s{int(s * 100)}"
        for k, r in enumerate(src.itertuples()):
            infrared = (rng.random() < cfg.get("ir_share", 0.0)) if cfg.get("source_kind") == "photo" \
                else r.chroma < cfg["ir_chroma"]
            prompt = caption(*names[target], infrared)
            init = Image.open(snip_root / r.relpath).convert("RGB").resize((GEN_RES, GEN_RES), Image.LANCZOS)
            seed = int(rng.integers(0, 2**31 - 1))
            img = pipe(prompt, image=init, strength=s, guidance_scale=cfg["guidance"], num_inference_steps=cfg["steps"],
                       generator=torch.Generator("cuda").manual_seed(seed)).images[0]
            quality = 95 if rng.random() < cfg.get("q95_share", 0.0) else 75
            data = finish_snip(img, quality)
            name = f"synth_{target}_{tag}_{seed}.jpg"
            (out / target / name).write_bytes(data)
            rows.append({"relpath": f"{target}/{name}", "class_code": target, "tag": tag, "prompt": prompt,
                         "source_relpath": r.relpath, "strength": s, "seed": seed, "guidance": cfg["guidance"],
                         "steps": cfg["steps"], "jpeg_quality": quality, "lora": str(weights),
                         "lora_sha256": lora_sha, "sha256": hashlib.sha256(data).hexdigest()})
    manifest = out / "manifest.parquet"
    df = pd.DataFrame(rows)
    if manifest.exists():
        df = pd.concat([pd.read_parquet(manifest), df], ignore_index=True)
    df.to_parquet(manifest, index=False)
    print(json.dumps({"out": str(out), "translated": len(rows), "source": cfg["source"], "target": target}))


if __name__ == "__main__":
    main(*sys.argv[1:7])
