"""Assemble a release bundle from a mewc-torch run: model.safetensors (float32 state dict), config.json (what
release/inference.py needs to rebuild the model), classes.txt, taxonomy.csv, inference.py, and the ONNX files
(model.onnx, model_fp16.onnx) when they sit beside the run. Checks that inference.py's model gives the same
logits as mewc_torch's on random input.
Usage: python release/build.py RUN_DIR OUT_DIR NAME"""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mewc_torch.model import MewcClassifier, base_kwargs  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(run_dir, out_dir, name):
    run, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rcfg = json.loads((run / "config.json").read_text())
    classes = (run / "classes.txt").read_text().split()
    assert classes == sorted(classes) == (HERE / "classes.txt").read_text().split()
    model = MewcClassifier(rcfg["backbone"], len(classes), pretrained=False, img_size=rcfg.get("img_size"))
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    model.eval()
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    cfg = {"name": name, "backbone": rcfg["backbone"], "timm_kwargs": base_kwargs(rcfg["backbone"], rcfg.get("img_size")),
           "n_features": model.compress.in_features, "compression_width": model.compress.out_features,
           "img_size": rcfg.get("img_size") or model.native_size, "mean": list(model.mean), "std": list(model.std),
           "snip_size": 600, "snip_jpeg_quality": 75, "classes": classes, "mewc_synth_commit": commit,
           "training": {k: rcfg[k] for k in ("seed", "samples_per_class", "frozen_epochs", "stage_len", "total_epochs",
                                             "batch", "lr", "warmup_steps", "select", "kd_weight") if k in rcfg}
                       | {"train_list": Path(rcfg["train"]).name, "distilled": bool(rcfg.get("soft_targets"))}}
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    (out / "classes.txt").write_text("\n".join(classes) + "\n")
    for f in ("inference.py", "taxonomy.csv"):
        shutil.copy(HERE / f, out / f)
    for f in ("model.onnx", "model_fp16.onnx"):
        if (run / f).exists():
            shutil.copy(run / f, out / f)
    spec = importlib.util.spec_from_file_location("bundle_inference", out / "inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    inf = mod.ModelInference(out, out / "model.safetensors")
    inf.load_model()
    x = torch.randn(2, 3, cfg["img_size"], cfg["img_size"])
    with torch.inference_mode():
        err = float((inf.model.cpu()(x) - model(x)).abs().max())
    print(f"{name}: bundle in {out}, inference.py logits differ from mewc_torch by {err:.2e}")
    if err > 1e-4:
        raise SystemExit("bundle model differs")


if __name__ == "__main__":
    main(*sys.argv[1:4])
