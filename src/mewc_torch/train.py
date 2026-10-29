"""Train a MEWC classifier on a timm backbone, reproducing mewc-train's recipe: a frozen stage on the
initial class sample (no augmentation, early stopping on training loss), then progressive stages with
the top blocks unfrozen, a fresh optimiser, and dropout and RandAugment magnitude stepped up per stage,
each epoch on a fresh per-class sample; categorical focal loss; AdamW with AMSGrad and exponential
learning-rate decay to min_lr_frac; the best epoch by validation loss is kept. Adds bf16, any batch
size and per-epoch checkpoints, so an interrupted run resumes where it stopped.
Usage: python -m mewc_torch.train CONFIG_JSON"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import Snips, read_list, sample_epoch
from .model import MewcClassifier, freeze_base, frozen_norms

DEFAULTS = {"samples_per_class": 4000, "frozen_epochs": 10, "stage_len": 7, "total_epochs": 30,
            "magnitudes": [0.2, 0.4, 0.6, 0.8], "dropouts": [0.1, 0.2, 0.3, 0.4], "n_aug": 3, "blocks": 2,
            "lr": 1e-4, "weight_decay": 1e-4, "min_lr_frac": 0.2, "batch": 16, "focal_alpha": 0.25,
            "focal_gamma": 2.0, "seed": 12345, "val_per_class": 100, "precision": "bf16", "workers": 16,
            "frozen_patience": 5}


def focal_loss(logits, y, alpha, gamma):
    logp = F.log_softmax(logits.float(), dim=1).gather(1, y[:, None])[:, 0]
    return (-alpha * (1 - logp.exp()) ** gamma * logp).mean()


def optimiser(params, cfg, steps_per_epoch):
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"], amsgrad=True)
    total = cfg["total_epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cfg["min_lr_frac"] ** (s / total))
    return opt, sched


def epoch_plan(cfg):
    """(phase, epoch, dropout, magnitude) for every epoch, mirroring fit_frozen and fit_progressive."""
    plan = [("frozen", e, cfg["dropouts"][0], 0.0) for e in range(cfg["frozen_epochs"])]
    stages, n = len(cfg["magnitudes"]), cfg["stage_len"]
    total = max(cfg["total_epochs"], stages * n)
    for s, (d, m) in enumerate(zip(cfg["dropouts"], cfg["magnitudes"])):
        end = (s + 1) * n if s < stages - 1 else total
        plan += [("progressive", e, d, m) for e in range(s * n, end)]
    return plan


def run_epoch(model, loader, cfg, dev, opt=None, sched=None):
    train = opt is not None
    model.train(train)
    if train:
        frozen_norms(model)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["precision"] == "bf16" and dev == "cuda")
    total, n, t0 = 0.0, 0, time.time()
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with amp:
                loss = focal_loss(model(x), y, cfg["focal_alpha"], cfg["focal_gamma"])
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            total += loss.item() * len(y)
            n += len(y)
    return total / n, n / (time.time() - t0)


def main(config_path):
    cfg = {**DEFAULTS, **json.loads(Path(config_path).read_text())}
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    torch.manual_seed(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes = sorted(Path(cfg["classes"]).read_text().split())
    tr_paths, tr_labels = read_list(cfg["train"], classes)
    va_paths, va_labels = read_list(cfg["val"], classes)
    va_rows = np.concatenate([np.flatnonzero(va_labels == c)[:cfg["val_per_class"]] for c in range(len(classes))])
    model = MewcClassifier(cfg["backbone"], len(classes), cfg["dropouts"][0], pretrained=cfg.get("pretrained", True),
                          img_size=cfg.get("img_size")).to(dev)
    size = cfg.get("img_size") or model.native_size
    loader = lambda paths, labels, rows, mag, n_aug, shuffle: DataLoader(
        Snips([paths[i] for i in rows], labels[rows], size, model.mean, model.std, mag, n_aug),
        batch_size=cfg["batch"], shuffle=shuffle, num_workers=cfg["workers"], pin_memory=dev == "cuda",
        persistent_workers=False, drop_last=False)
    val_loader = loader(va_paths, va_labels, va_rows, 0.0, 0, False)
    steps = (len(classes) * cfg["samples_per_class"] + cfg["batch"] - 1) // cfg["batch"]
    plan = epoch_plan(cfg)
    ckpt, state = out / "last.pt", {"next": 0, "best_val": float("inf"), "history": [], "frozen_best": float("inf"), "bad": 0}
    resume, phase = None, None
    if ckpt.exists():
        resume = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(resume["model"])
        state = resume["state"]
    for k in range(state["next"], len(plan)):
        ph, e, drop, mag = plan[k]
        if ph != phase:
            leaving_frozen = phase == "frozen" or (resume is not None and resume["phase"] == "frozen")
            if ph == "progressive" and leaving_frozen and (out / "frozen_best.pt").exists():
                model.load_state_dict(torch.load(out / "frozen_best.pt", map_location=dev))
            freeze_base(model, 0 if ph == "frozen" else cfg["blocks"])
            opt, sched = optimiser([p for p in model.parameters() if p.requires_grad], cfg, steps)
            if resume is not None and resume["phase"] == ph:
                opt.load_state_dict(resume["opt"])
                sched.load_state_dict(resume["sched"])
            resume, phase = None, ph
        if ph == "frozen" and state["bad"] >= cfg["frozen_patience"]:
            continue
        model.set_dropout(drop)
        seed = cfg["seed"] + (0 if ph == "frozen" else e)
        rows = sample_epoch(tr_labels, cfg["samples_per_class"], seed)
        n_aug = cfg["n_aug"] if ph == "progressive" else 0
        tr_loss, tr_ips = run_epoch(model, loader(tr_paths, tr_labels, rows, mag, n_aug, False), cfg, dev, opt, sched)
        va_loss, va_ips = run_epoch(model, val_loader, cfg, dev)
        rec = {"k": k, "phase": ph, "epoch": e, "dropout": drop, "magnitude": mag, "train_loss": tr_loss,
               "val_loss": va_loss, "train_img_s": round(tr_ips, 1), "lr": sched.get_last_lr()[0], "time": time.time()}
        state["history"].append(rec)
        print(json.dumps(rec), flush=True)
        if ph == "frozen":
            if tr_loss < state["frozen_best"] - 1e-3:
                state["frozen_best"], state["bad"] = tr_loss, 0
                torch.save(model.state_dict(), out / "frozen_best.pt")
            else:
                state["bad"] += 1
        elif va_loss < state["best_val"]:
            state["best_val"] = va_loss
            torch.save(model.state_dict(), out / "best.pt")
        state["next"] = k + 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "phase": ph, "state": state}, ckpt)
    summary = {"backbone": cfg["backbone"], "img_size": size, "best_val_loss": state["best_val"],
               "epochs": len(state["history"]), "classes": len(classes)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "history.json").write_text(json.dumps(state["history"], indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main(sys.argv[1])
