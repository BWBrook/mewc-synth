"""AddaxAI inference for the mewc-synth Tasmanian classifiers (https://github.com/BWBrook/mewc-synth).

A timm backbone with the MEWC head (average-pooled features, dropout, a compression layer, ELU, dropout, the
class layer), loaded from model.safetensors as described by config.json. Needs torch, timm, safetensors, numpy
and Pillow (AddaxAI's pytorch environment); downloads nothing.

Preprocessing reproduces the training snips exactly: the MegaDetector box is cropped as mewc-snip crops it
(MegaDetector's crop_image), resized to 600 x 600 with LANCZOS and passed through a JPEG round trip at quality
75 (how the snips were stored), then resized to the model's input size by bilinear interpolation without
antialiasing and normalised with the backbone's mean and standard deviation. The model runs in float32.

Follows backend/templates/inference_template.py of AddaxAI (https://github.com/PetervanLunteren/AddaxAI)."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file


class MewcClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.base = timm.create_model(cfg["backbone"], pretrained=False, **cfg["timm_kwargs"])
        n, width = cfg["n_features"], cfg["compression_width"]
        self.drop1, self.compress, self.act = nn.Dropout(0.0), nn.Linear(n, width), nn.ELU()
        self.drop2, self.classify = nn.Dropout(0.0), nn.Linear(width, len(cfg["classes"]))

    def forward(self, x):
        return self.classify(self.drop2(self.act(self.compress(self.drop1(self.base(x))))))


class ModelInference:
    def __init__(self, model_dir: Path, model_path: Path):
        self.model_dir = Path(model_dir)
        self.model_path = Path(model_path)
        self.model = None
        self.cfg = json.loads((self.model_dir / "config.json").read_text())
        self.classes = self.cfg["classes"]
        self.size = self.cfg["img_size"]
        self.snip_size = self.cfg["snip_size"]
        self.jpeg_quality = self.cfg["snip_jpeg_quality"]
        self.mean = torch.tensor(self.cfg["mean"]).view(3, 1, 1) * 255
        self.std = torch.tensor(self.cfg["std"]).view(3, 1, 1) * 255
        self.device = None

    def check_gpu(self) -> bool:
        return torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    def load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        model = MewcClassifier(self.cfg)
        model.load_state_dict(load_file(str(self.model_path)), strict=True)
        self.model = model.eval().to(self.device)

    def get_crop(self, image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image | None:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            print(f"[mewc-synth get_crop] rejecting box with zero or negative size: {bbox}", file=sys.stderr, flush=True)
            return None
        im_w, im_h = image.size
        left, right = min(max(x * im_w, 0), im_w - 1), min(max((x + w) * im_w, 0), im_w - 1)
        top, bottom = min(max(y * im_h, 0), im_h - 1), min(max((y + h) * im_h, 0), im_h - 1)
        if right - left <= 0 or bottom - top <= 0:
            print(f"[mewc-synth get_crop] rejecting box outside the image: {bbox}", file=sys.stderr, flush=True)
            return None
        snip = image.convert("RGB").crop((left, top, right, bottom)).resize((self.snip_size, self.snip_size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        snip.save(buf, format="JPEG", quality=self.jpeg_quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def get_tensor(self, crop: Image.Image):
        x = torch.from_numpy(np.array(crop.convert("RGB"))).permute(2, 0, 1)
        x = F.interpolate(x[None].float(), size=(self.size, self.size), mode="bilinear",
                          align_corners=False, antialias=False)[0].round().clamp(0, 255)
        return ((x - self.mean) / self.std).numpy().astype(np.float32)

    def classify_batch(self, batch):
        if self.model is None:
            raise RuntimeError("Model not loaded - call load_model() first")
        with torch.inference_mode():
            probs = torch.softmax(self.model(torch.from_numpy(np.asarray(batch)).to(self.device)).float(), dim=1).cpu().numpy()
        return [[[self.classes[j], float(p[j])] for j in range(len(p))] for p in probs]

    def get_classification(self, crop: Image.Image) -> list[list[str, float]]:
        if crop is None:
            return []
        return self.classify_batch(self.get_tensor(crop)[None])[0]

    def get_class_names(self) -> dict[str, str]:
        return {str(i + 1): name for i, name in enumerate(self.classes)}
