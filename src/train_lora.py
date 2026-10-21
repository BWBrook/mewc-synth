"""Train one SDXL UNet LoRA on cached latents and text embeddings (Stage A, or Stage B from init_lora).
Default class sampling weight is n_c^-alpha per snip (alpha 0.5 = square-root balancing). With "focus"
(Stage B), only the named classes (each with the given total sampling mass) and prior_n random rows from
other classes (sharing prior_mass) are sampled; "init_lora" resumes from a saved LoRA.
Saves LoRA checkpoints, a loss log and fixed-seed sample grids.
Usage: python src/train_lora.py RUN_NAME OUT_NAME CONFIG_JSON
CONFIG_JSON example: {"rank": 64, "lr": 1e-4, "steps": 6000, "batch": 32, "alpha": 0.5,
  "warmup": 200, "min_snr": 5.0, "save_every": 1000, "sample_every": 1000, "seed": 7,
  "sample_prompts": ["camera trap snip of a tasmanian devil, infrared night"]}
Stage B additions: "init_lora": "<lora dir>", "focus": {"classes": {"scrubtit": 0.45,
  "tasmanian_scrubwren": 0.25}, "prior_mass": 0.3, "prior_n": 3000}"""
import json
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file
from PIL import Image

from common import DATA, GEN_RES, SDXL, VAE_FP16


def save_lora(unet, path):
    layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(path, unet_lora_layers=layers, safe_serialization=True)


def sample_grid(unet, prompts, path, seed):
    vae = AutoencoderKL.from_pretrained(VAE_FP16, torch_dtype=torch.bfloat16)
    pipe = StableDiffusionXLPipeline.from_pretrained(SDXL, unet=unet, vae=vae, torch_dtype=torch.bfloat16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)
    tiles = []
    for p in prompts:
        g = torch.Generator("cuda").manual_seed(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tiles += pipe(p, height=GEN_RES, width=GEN_RES, num_inference_steps=30, num_images_per_prompt=4, generator=g).images
    grid = Image.new("RGB", (4 * 256, len(prompts) * 256))
    for i, t in enumerate(tiles):
        grid.paste(t.resize((256, 256), Image.LANCZOS), ((i % 4) * 256, (i // 4) * 256))
    grid.save(path, quality=90)
    del pipe, vae
    torch.cuda.empty_cache()


def main(run, out_name, config):
    cfg = json.loads(config)
    src, out = DATA / "stage_a" / run, DATA / "stage_a" / run / "lora" / out_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    rows = pd.read_parquet(src / "rows.parquet")
    lat = torch.from_numpy(np.load(src / "latents.npy")).cuda() # about 10 GB for 150k snips: kept on the GPU
    te = torch.load(src / "text_embeds.pt")
    pe_all, pooled_all = te["prompt_embeds"].cuda(), te["pooled"].cuda()
    if "focus" in cfg:
        f = cfg["focus"]
        counts = rows["class_code"].value_counts()
        w = np.array(rows["class_code"].map(lambda c: f["classes"].get(c, 0.0) / counts[c]), dtype=np.float64)
        others = np.flatnonzero(w == 0)
        w[rng.choice(others, min(f["prior_n"], len(others)), replace=False)] = f["prior_mass"] / min(f["prior_n"], len(others))
    else:
        w = np.array(rows["class_code"].map(rows["class_code"].value_counts() ** -cfg["alpha"]), dtype=np.float64)
    w /= w.sum()
    cap_ids = torch.from_numpy(rows["caption_id"].to_numpy()).cuda()

    unet = UNet2DConditionModel.from_pretrained(SDXL, subfolder="unet", torch_dtype=torch.bfloat16, variant="fp16")
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=cfg["rank"], lora_alpha=cfg["rank"], init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    if cfg.get("init_lora"):
        sd = load_file(f"{cfg['init_lora']}/pytorch_lora_weights.safetensors")
        sd = convert_unet_state_dict_to_peft({k.removeprefix("unet."): v for k, v in sd.items() if k.startswith("unet.")})
        res = set_peft_model_state_dict(unet, sd, adapter_name="default")
        if getattr(res, "unexpected_keys", None):
            raise ValueError(f"init_lora has {len(res.unexpected_keys)} unexpected keys, e.g. {res.unexpected_keys[:2]}")
        print(json.dumps({"init_lora": cfg["init_lora"], "tensors": len(sd)}), flush=True)
    unet.enable_gradient_checkpointing()
    unet.to("cuda").train()
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / cfg["warmup"]) *
                                              0.5 * (1 + math.cos(math.pi * min(s, cfg["steps"]) / cfg["steps"])))
    noise_sched = DDPMScheduler.from_pretrained(SDXL, subfolder="scheduler")
    acp = noise_sched.alphas_cumprod.cuda()
    snr_all = acp / (1 - acp)
    b = cfg["batch"]
    time_ids = torch.tensor([[GEN_RES, GEN_RES, 0, 0, GEN_RES, GEN_RES]] * b, device="cuda", dtype=torch.bfloat16)

    log = open(out / "loss.jsonl", "a")
    t0, run_loss = time.time(), 0.0
    for step in range(1, cfg["steps"] + 1):
        idx = torch.from_numpy(rng.choice(len(rows), b, p=w)).cuda()
        flip = torch.from_numpy(rng.integers(0, 2, b)).cuda()
        x0 = lat[idx, flip].to(torch.bfloat16)
        cid = cap_ids[idx]
        noise = torch.randn_like(x0)
        t = torch.randint(0, noise_sched.config.num_train_timesteps, (b,), device="cuda")
        xt = noise_sched.add_noise(x0, noise, t)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = unet(xt, t, encoder_hidden_states=pe_all[cid].to(torch.bfloat16),
                        added_cond_kwargs={"text_embeds": pooled_all[cid].to(torch.bfloat16), "time_ids": time_ids}).sample
        mse = ((pred.float() - noise.float()) ** 2).mean(dim=(1, 2, 3))
        snr = snr_all[t]
        weight = torch.clamp(snr, max=cfg["min_snr"]) / snr if cfg["min_snr"] else torch.ones_like(snr)
        loss = (mse * weight).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        run_loss += loss.item()
        if step % 50 == 0:
            log.write(json.dumps({"step": step, "loss": round(run_loss / 50, 5), "lr": sched.get_last_lr()[0],
                                  "img_per_s": round(50 * b / (time.time() - t0), 1)}) + "\n")
            log.flush()
            run_loss, t0 = 0.0, time.time()
        if step % cfg["save_every"] == 0 or step == cfg["steps"]:
            save_lora(unet, out / f"step_{step:06d}")
        if step % cfg["sample_every"] == 0 or step == cfg["steps"]:
            unet.eval()
            sample_grid(unet, cfg["sample_prompts"], out / f"samples_{step:06d}.jpg", cfg["seed"])
            unet.train()
    print(json.dumps({"out": str(out), "steps": cfg["steps"]}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
