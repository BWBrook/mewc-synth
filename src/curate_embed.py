"""Embed every snip in the MEWC zip with DINOv2-base (CLS, unit-normalised, float16), reading stored
entries by offset without extracting. Resumable: the index is written once; shards of SHARD entries
are written as separate .npy files and skipped if present. Matches embed.py preprocessing
(RGB, bicubic 224, ImageNet normalisation), so embeddings are comparable with the working set.
Usage: python src/curate_embed.py ZIP OUT_DIR [DEVICE] [WORKERS] [LIMIT]"""
import gc
import json
import struct
import sys
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

MODEL = "facebook/dinov2-base"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SHARD = 50_000


def build_index(zip_path, out):
    path = out / "index.parquet"
    if path.exists():
        return pd.read_parquet(path)
    z = zipfile.ZipFile(zip_path)
    rows = []
    for info in z.infolist():
        parts = info.filename.split("/")
        if info.is_dir() or len(parts) != 4 or not parts[3].lower().endswith((".jpg", ".jpeg")):
            continue
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"not stored: {info.filename}")
        rows.append((info.filename, parts[1], parts[2], info.header_offset, info.compress_size, info.CRC))
    del z
    gc.collect()
    idx = pd.DataFrame(rows, columns=["zip_name", "tier", "class_code", "offset", "size", "crc32"])
    idx = idx.sort_values("zip_name").reset_index(drop=True)
    idx.to_parquet(path, index=False)
    return idx


class ZipSnips(Dataset):
    def __init__(self, zip_path, items):
        self.zip_path, self.items, self.f = zip_path, items, None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.f is None:
            self.f = open(self.zip_path, "rb")
        offset, size, crc = self.items[i]
        self.f.seek(offset)
        n, e = struct.unpack("<HH", self.f.read(30)[26:30])
        self.f.seek(offset + 30 + n + e)
        data = self.f.read(size)
        ok = zlib.crc32(data) == crc
        try:
            from io import BytesIO
            with Image.open(BytesIO(data)) as img:
                a = np.asarray(img.convert("RGB").resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255
            x = (torch.from_numpy(a).permute(2, 0, 1) - MEAN) / STD
        except Exception:
            x, ok = torch.zeros(3, 224, 224), False
        return x, ok


def main(zip_path, out_dir, device="cpu", workers=16, limit=0):
    out, workers, limit = Path(out_dir), int(workers), int(limit)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    idx = build_index(zip_path, out)
    if limit:
        idx = idx.iloc[:limit]
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(MODEL, dtype=dtype).to(device).eval()
    for s in range(0, len(idx), SHARD):
        target = out / "shards" / f"emb_{s // SHARD:04d}.npy"
        if target.exists():
            continue
        part = idx.iloc[s:s + SHARD]
        loader = DataLoader(ZipSnips(zip_path, list(zip(part["offset"], part["size"], part["crc32"]))),
                            batch_size=128, num_workers=workers, pin_memory=device == "cuda")
        embs, oks, t0 = [], [], time.time()
        with torch.inference_mode():
            for x, ok in loader:
                z = model(pixel_values=x.to(device, dtype)).pooler_output.float()
                embs.append(torch.nn.functional.normalize(z, dim=1).cpu().numpy().astype(np.float16))
                oks.append(ok.numpy())
        tmp = target.with_suffix(".tmp.npy")
        np.save(tmp, np.concatenate(embs))
        np.save(out / "shards" / f"ok_{s // SHARD:04d}.npy", np.concatenate(oks))
        tmp.rename(target)
        rate = len(part) / (time.time() - t0)
        print(json.dumps({"shard": s // SHARD, "rows": len(part), "img_per_s": round(rate, 1),
                          "bad": int((~np.concatenate(oks)).sum()), "time": time.strftime("%H:%M:%S")}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:6])
