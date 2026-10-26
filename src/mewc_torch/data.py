"""Data for mewc-torch: per-epoch class sampling as mewc-train does it, and a snip dataset that resizes
like tf.image.resize (bilinear, no antialiasing) and applies RandAugment at a given magnitude."""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

MAG_BINS = 31


def read_list(path, classes):
    """A CSV or Parquet list with columns path and label; labels encoded in the sorted class order."""
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    index = {c: i for i, c in enumerate(classes)}
    unknown = set(df["label"]) - set(index)
    if unknown:
        raise ValueError(f"labels not in the class list: {sorted(unknown)[:5]}")
    return df["path"].tolist(), np.array([index[c] for c in df["label"]])


def sample_epoch(labels, n_per_class, seed):
    """n_per_class rows per class (with replacement only when a class has fewer), then shuffled."""
    rng = np.random.default_rng(seed)
    picks = []
    for c in np.unique(labels):
        rows = np.flatnonzero(labels == c)
        picks.append(rng.choice(rows, n_per_class, replace=len(rows) < n_per_class))
    return rng.permutation(np.concatenate(picks))


class Snips(Dataset):
    def __init__(self, paths, labels, size, mean, std, magnitude=0.0, n_ops=0):
        self.paths, self.labels, self.size = paths, labels, size
        self.mean = torch.tensor(mean).view(3, 1, 1) * 255
        self.std = torch.tensor(std).view(3, 1, 1) * 255
        m = int(round(magnitude * (MAG_BINS - 1)))
        self.aug = v2.RandAugment(num_ops=n_ops, magnitude=m, num_magnitude_bins=MAG_BINS) if n_ops and m else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = torch.from_numpy(np.array(im.convert("RGB"))).permute(2, 0, 1)
        x = F.interpolate(x[None].float(), size=(self.size, self.size), mode="bilinear",
                          align_corners=False, antialias=False)[0].round().clamp(0, 255).to(torch.uint8)
        if self.aug is not None:
            x = self.aug(x)
        return (x.float() - self.mean) / self.std, int(self.labels[i])
