"""Score a list of snips with a trained mewc-torch run (best.pt), writing the same CSV and probability
array that vtl_score.py writes, so screen_eval.py and pilot_summary.py read them unchanged.
Usage: python -m mewc_torch.predict RUN_DIR PATHS_TXT OUT_PREFIX [BATCH]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import Snips
from .model import MewcClassifier


def load_run(run_dir, dev):
    run = Path(run_dir)
    cfg = json.loads((run / "config.json").read_text())
    classes = sorted(Path(cfg["classes"]).read_text().split())
    model = MewcClassifier(cfg["backbone"], len(classes), pretrained=False, img_size=cfg.get("img_size"))
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    size = cfg.get("img_size") or model.native_size
    return model.to(dev).eval(), classes, size, cfg


def main(run_dir, paths_txt, out_prefix, batch=128):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, size, cfg = load_run(run_dir, dev)
    paths = [l.strip() for l in open(paths_txt) if l.strip()]
    ds = Snips(paths, np.zeros(len(paths), int), size, model.mean, model.std)
    probs = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
        for x, _ in DataLoader(ds, batch_size=int(batch), num_workers=cfg.get("workers", 16)):
            probs.append(torch.softmax(model(x.to(dev)).float(), dim=1).cpu().numpy())
    p = np.concatenate(probs)
    top = p.argmax(1)
    pd.DataFrame({"path": paths, "top_name": [classes[i] for i in top], "top_prob": p.max(1)}).to_csv(
        f"{out_prefix}.csv", index=False)
    np.save(f"{out_prefix}_probs.npy", p.astype(np.float32))
    print(f"scored {len(paths)} images with {run_dir} ({size}px, {len(classes)} classes)")


if __name__ == "__main__":
    main(*sys.argv[1:5])
