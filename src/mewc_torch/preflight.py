"""Check architecture-race candidates before GPU time is spent: build each backbone with the MEWC head at its
race size, count its parameters, body groups and the share its unfreeze setting trains, export it to ONNX,
check the export against PyTorch and time it in onnxruntime on CPU threads (bench.py's protocol).
Appends one JSON line per candidate; a candidate that fails records its error and the rest continue.
Usage: python -m mewc_torch.preflight RACE_CSV OUT_JSONL [THREADS] [N_CLASSES]
RACE_CSV columns: name, backbone, img_size, blocks (and any others, carried through)."""
import json
import sys
import tempfile
import traceback

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from . import bench
from .model import BODY, Deployable, MewcClassifier, block_groups, freeze_base


def check(row, threads, n_classes):
    size = int(row["img_size"])
    model = MewcClassifier(row["backbone"], n_classes, img_size=size).eval()
    total = sum(p.numel() for p in model.parameters())
    groups = block_groups(model.base)
    body = sum(any(n.split(".")[0] in BODY for n in g) for g in groups)
    trained = freeze_base(model, int(row["blocks"]))
    wrapped = Deployable(model).eval()
    x = torch.rand(2, 3, size, size) * 255
    with torch.inference_mode():
        ref = wrapped(x).numpy()
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/model.onnx"
        torch.onnx.export(wrapped, (x,), path, input_names=["image"], output_names=["probs"],
                          dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}}, opset_version=18, dynamo=False)
        got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
        speed = bench.main(path, f"{d}/bench.json", "cpu", threads)
    return {"params_M": round(total / 1e6, 1), "body_groups": body, "trained_share": round(trained / sum(
        p.numel() for p in model.base.parameters()), 3), "onnx_max_err": float(np.abs(ref - got).max()),
        "cpu_img_s": speed["img_per_s"], "threads": int(threads)}


def main(race_csv, out, threads=8, n_classes=96):
    torch.set_num_threads(int(threads))
    for row in pd.read_csv(race_csv).to_dict("records"):
        try:
            res = {**row, **check(row, int(threads), int(n_classes)), "ok": True}
        except Exception as e:
            res = {**row, "ok": False, "error": f"{type(e).__name__}: {e}"[:500]}
            traceback.print_exc()
        with open(out, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
