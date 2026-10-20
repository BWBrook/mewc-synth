"""Generate synthetic snips from SDXL plus a Stage A LoRA, finished exactly like real snips
(512 px -> Lanczos 600 px -> PIL default JPEG), with a manifest row per image.
Usage: python src/generate.py LORA_DIR CLASS_MAP_YAML OUT_DIR JOBS_JSON
JOBS_JSON example: {"seed": 11, "steps": 30, "guidance": 5.0, "batch": 8,
  "jobs": [{"code": "3", "n": 64, "infrared_share": 0.6}]}"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

from common import GEN_RES, SDXL, VAE_FP16, finish_snip
from prep_stage_a import caption


def main(lora_dir, class_map, out_dir, jobs_json):
    cfg, lora_dir, out = json.loads(jobs_json), Path(lora_dir), Path(out_dir)
    names = {str(k): v.strip() for k, v in yaml.safe_load(open(class_map)).items()}
    weights = lora_dir / "pytorch_lora_weights.safetensors"
    lora_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    vae = AutoencoderKL.from_pretrained(VAE_FP16, torch_dtype=torch.bfloat16)
    pipe = StableDiffusionXLPipeline.from_pretrained(SDXL, vae=vae, torch_dtype=torch.bfloat16, variant="fp16").to("cuda")
    pipe.load_lora_weights(str(lora_dir))
    pipe.set_progress_bar_config(disable=True)
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for job in cfg["jobs"]:
        code, n, b = str(job["code"]), int(job["n"]), int(cfg["batch"])
        (out / code).mkdir(parents=True, exist_ok=True)
        k = 0
        while k < n:
            m = min(b, n - k)
            infrared = bool(rng.random() < job["infrared_share"])
            prompt = caption(names[code], infrared)
            seed = int(rng.integers(0, 2**31 - 1))
            images = pipe(prompt, height=GEN_RES, width=GEN_RES, num_inference_steps=cfg["steps"],
                          guidance_scale=cfg["guidance"], num_images_per_prompt=m,
                          generator=torch.Generator("cuda").manual_seed(seed)).images
            for j, img in enumerate(images):
                name = f"synth_{code}_{seed}_{j}.jpg"
                data = finish_snip(img)
                (out / code / name).write_bytes(data)
                rows.append({"relpath": f"{code}/{name}", "class_code": code, "prompt": prompt, "seed": seed,
                             "index_in_batch": j, "steps": cfg["steps"], "guidance": cfg["guidance"],
                             "base_model": SDXL, "vae": VAE_FP16, "lora": str(weights), "lora_sha256": lora_sha,
                             "sha256": hashlib.sha256(data).hexdigest()})
            k += m
    manifest = out / "manifest.parquet"
    df = pd.DataFrame(rows)
    if manifest.exists():
        df = pd.concat([pd.read_parquet(manifest), df], ignore_index=True)
    df.to_parquet(manifest, index=False)
    print(json.dumps({"out": str(out), "generated": len(rows), "lora_sha256": lora_sha[:12]}))


if __name__ == "__main__":
    main(*sys.argv[1:5])
