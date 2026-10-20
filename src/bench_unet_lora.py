"""Throughput benchmark: SDXL UNet LoRA training step at 512 px on cached-latent
shapes (random tensors), bf16, gradient checkpointing.
Usage: python src/bench_unet_lora.py [RANK] [BATCH] [STEPS]"""
import json
import sys
import time

import torch
from diffusers import UNet2DConditionModel
from peft import LoraConfig

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
rank = int(sys.argv[1]) if len(sys.argv) > 1 else 64
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 16
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 30

unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=torch.bfloat16, variant="fp16")
unet.requires_grad_(False)
unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                            target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
for p in unet.parameters():
    if p.requires_grad:
        p.data = p.data.float()
unet.enable_gradient_checkpointing()
unet.to("cuda").train()
params = [p for p in unet.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=1e-4)

lat = torch.randn(batch, 4, 64, 64, device="cuda", dtype=torch.bfloat16)
txt = torch.randn(batch, 77, 2048, device="cuda", dtype=torch.bfloat16)
pooled = torch.randn(batch, 1280, device="cuda", dtype=torch.bfloat16)
time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]] * batch, device="cuda", dtype=torch.bfloat16)
cond = {"text_embeds": pooled, "time_ids": time_ids}

def step():
    t = torch.randint(0, 1000, (batch,), device="cuda")
    noise = torch.randn_like(lat)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = unet(lat + noise, t, encoder_hidden_states=txt, added_cond_kwargs=cond).sample
    loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

for _ in range(3):
    step()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(steps):
    step()
torch.cuda.synchronize()
dt = time.perf_counter() - t0
ips = steps * batch / dt
print(json.dumps({"rank": rank, "batch": batch, "steps": steps,
                  "trainable_params_m": round(sum(p.numel() for p in params) / 1e6, 1),
                  "images_per_s": round(ips, 1), "hours_per_100k": round(1e5 / ips / 3600, 2),
                  "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}))
