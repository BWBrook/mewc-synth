"""Smoke test: SDXL base generation at 512 px on the A100, timed.
Usage: python src/gen_smoke.py OUTDIR [N_BATCHES]"""
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPTS = [
    "camera trap snip of a Tasmanian devil (Sarcophilus harrisii), infrared night",
    "camera trap snip of a spotted-tailed quoll (Dasyurus maculatus), infrared night",
    "camera trap snip of a Bennett's wallaby (Notamacropus rufogriseus), daylight colour",
    "camera trap snip of a superb lyrebird (Menura novaehollandiae), daylight colour",
]

out = Path(sys.argv[1])
n_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 4
out.mkdir(parents=True, exist_ok=True)

pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, variant="fp16").to("cuda")
pipe.set_progress_bar_config(disable=True)

batch, steps = 8, 30
pipe(PROMPTS[0], height=512, width=512, num_inference_steps=2, num_images_per_prompt=batch)
torch.cuda.synchronize()

times, k = [], 0
for b in range(n_batches):
    prompt = PROMPTS[b % len(PROMPTS)]
    g = torch.Generator("cuda").manual_seed(1000 + b)
    t = time.perf_counter()
    images = pipe(prompt, height=512, width=512, num_inference_steps=steps,
                  num_images_per_prompt=batch, generator=g).images
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t)
    for img in images:
        img.save(out / f"synth_smoke_{k:04d}.png")
        k += 1

per_img = sum(times[1:] or times) / (len(times[1:] or times) * batch)
print(json.dumps({"model": MODEL, "res": 512, "steps": steps, "batch": batch, "images": k,
                  "seconds_per_image": round(per_img, 3), "images_per_hour": round(3600 / per_img),
                  "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}))
