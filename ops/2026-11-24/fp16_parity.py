"""fp16 deployment check (PLAN 2026-11-02, decision 1: fp16 on the RTX 2080 only if its predictions match fp32):
export a run to ONNX with mewc_torch.export (fused where timm can fuse; checked against PyTorch), convert a copy
to fp16 (inputs and outputs kept fp32), and score the same snips with CUDA fp32, CUDA fp16 and CPU fp32.
Reports top-1 agreement and probability differences against CUDA fp32. The snips only need to be real camera-trap
inputs, so validation snips serve even for a refit that trained on them.
Usage: python ops/2026-11-24/fp16_parity.py RUN_DIR SNIPS_CSV OUT_JSON"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn.functional as F
from onnxruntime.transformers.float16 import convert_float_to_float16
from PIL import Image

from mewc_torch import export


def batches(paths, size, n=32):
    for i in range(0, len(paths), n):
        x = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)).permute(2, 0, 1)
                         for p in paths[i:i + n]])
        yield F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=False).numpy()


def main(run_dir, snips_csv, out):
    ort.preload_dlls()
    run = Path(run_dir)
    onnx32 = run / "model.onnx"
    export.main(str(run), str(onnx32))
    size = json.loads((run / "config.json").read_text())["img_size"]
    onnx16 = run / "model_fp16.onnx"
    onnx.save(convert_float_to_float16(onnx.load(onnx32), keep_io_types=True), onnx16)
    sess = {"cuda32": ort.InferenceSession(str(onnx32), providers=["CUDAExecutionProvider"]),
            "cuda16": ort.InferenceSession(str(onnx16), providers=["CUDAExecutionProvider"]),
            "cpu32": ort.InferenceSession(str(onnx32), providers=["CPUExecutionProvider"])}
    for k in ("cuda32", "cuda16"):
        assert sess[k].get_providers()[0] == "CUDAExecutionProvider", k
    paths = pd.read_csv(snips_csv).path.tolist()
    probs = {k: [] for k in sess}
    for x in batches(paths, size):
        for k, s in sess.items():
            probs[k].append(s.run(None, {s.get_inputs()[0].name: x})[0])
    p = {k: np.concatenate(v) for k, v in probs.items()}
    ref = p["cuda32"]
    res = {"run": run.name, "snips": len(paths), "img_size": size}
    for k in ("cuda16", "cpu32"):
        d = np.abs(p[k] - ref)
        res[k] = {"top1_agree": float((p[k].argmax(1) == ref.argmax(1)).mean()),
                  "disagreements": int((p[k].argmax(1) != ref.argmax(1)).sum()),
                  "max_abs_prob_diff": float(d.max()), "mean_abs_top_prob_diff": float(np.abs(p[k].max(1) - ref.max(1)).mean())}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main(*sys.argv[1:4])
