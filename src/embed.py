"""Embed every inventoried snip: DINOv2-base CLS embedding (unit-normalised, float16),
64-bit perceptual hash and chroma (day colour vs infrared/greyscale).
Usage: python src/embed.py SNIP_ROOT INV_DIR"""
import sys
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

from common import chroma

MODEL = "facebook/dinov2-base"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class Snips(Dataset):
    def __init__(self, root, relpaths):
        self.root, self.relpaths = Path(root), relpaths

    def __len__(self):
        return len(self.relpaths)

    def __getitem__(self, i):
        with Image.open(self.root / self.relpaths[i]) as img:
            img = img.convert("RGB")
            ph = int(str(imagehash.phash(img)), 16)
            ch = chroma(img)
            x = torch.from_numpy(np.asarray(img.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255)
        x = (x.permute(2, 0, 1) - MEAN) / STD
        return x, np.uint64(ph).view(np.int64), ch


def main(root, inv_dir):
    inv_dir = Path(inv_dir)
    inv = pd.read_parquet(inv_dir / "inventory.parquet")
    loader = DataLoader(Snips(root, inv["relpath"].tolist()), batch_size=256, num_workers=48, pin_memory=True)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    emb = np.lib.format.open_memmap(inv_dir / "dinov2_base.npy", mode="w+", dtype=np.float16, shape=(len(inv), 768))
    phash, chro, k = np.empty(len(inv), np.int64), np.empty(len(inv), np.float32), 0
    with torch.inference_mode():
        for x, ph, ch in loader:
            z = model(pixel_values=x.cuda(non_blocking=True).half()).pooler_output.float()
            z = torch.nn.functional.normalize(z, dim=1)
            n = len(z)
            emb[k:k + n] = z.cpu().numpy().astype(np.float16)
            phash[k:k + n], chro[k:k + n] = ph.numpy(), ch.numpy()
            k += n
    emb.flush()
    pd.DataFrame({"relpath": inv["relpath"], "phash": phash, "chroma": chro}).to_parquet(inv_dir / "embed_meta.parquet", index=False)
    print(f"embedded {k} snips -> {inv_dir / 'dinov2_base.npy'}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
