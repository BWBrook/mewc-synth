"""Export a mewc-torch run to ONNX for deployment: input raw RGB 0..255, NCHW float32 at the training size,
dynamic batch; output class probabilities in the order of classes.txt (written beside the model). Branches
that timm can reparameterise (FastViT, MobileOne and similar) are fused first; the ONNX model is checked
against the unfused PyTorch model on random input.
Usage: python -m mewc_torch.export RUN_DIR [OUT_ONNX]"""
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from timm.utils.model import reparameterize_model

from .model import Deployable
from .predict import load_run


def main(run_dir, out=None):
    model, classes, size, cfg = load_run(run_dir, "cpu")
    out = Path(out or Path(run_dir) / "model.onnx")
    wrapped = Deployable(model).eval()
    x = torch.rand(2, 3, size, size) * 255
    torch.onnx.export(Deployable(reparameterize_model(model)).eval(), (x,), str(out), input_names=["image"], output_names=["probs"],
                      dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}}, opset_version=18, dynamo=False)
    (out.parent / "classes.txt").write_text("\n".join(classes) + "\n")
    with torch.inference_mode():
        ref = wrapped(x).numpy()
    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    err = float(np.abs(ref - got).max())
    print(f"exported {out} ({size}px, {len(classes)} classes), max abs difference from PyTorch {err:.2e}")
    if err > 1e-3:
        raise SystemExit("ONNX output differs from PyTorch")


if __name__ == "__main__":
    main(*sys.argv[1:3])
