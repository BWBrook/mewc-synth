"""Deployment throughput of the shortlisted models on the decision 1 hardware (Barry, 2 November: an 8-core CPU
and an RTX 2080-class GPU, plus the A100 for reference), measured on alien (i7-9700K, 8 cores; RTX 2080 8 GB).
Each model is built with random weights (throughput does not depend on them), wrapped as export.py does and
exported to ONNX; where timm's reparameterize_model changes the network (FastViT, HGNetV2), the fused form is
timed as well. Timings: onnxruntime CPU, 8 threads, batch 8 (deep's protocol); CUDA fp32 and fp16 (weights
converted, input and output kept fp32) at batch 8 and 32. Appends one JSON line per model.
Usage: python ops/2026-11-02/hw_bench.py MODELS_CSV OUT_JSONL"""
import copy
import json
import sys
import tempfile
import traceback

import onnx
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.transformers.float16 import convert_float_to_float16
from timm.utils.model import reparameterize_model

from mewc_torch import bench
from mewc_torch.model import Deployable, MewcClassifier


def export(model, size, path):
    x = torch.rand(2, 3, size, size) * 255
    torch.onnx.export(Deployable(model).eval(), (x,), path, input_names=["image"], output_names=["probs"],
                      dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}}, opset_version=18, dynamo=False)


def timings(path, d, tag):
    fp16 = f"{d}/{tag}_fp16.onnx"
    onnx.save(convert_float_to_float16(onnx.load(path), keep_io_types=True), fp16)
    res = {f"{tag}_cpu8": bench.main(path, f"{d}/b.json", "cpu", 8, 8, 8)["img_per_s"]}
    for p, prec in [(path, "fp32"), (fp16, "fp16")]:
        for batch in (8, 32):
            r = bench.main(p, f"{d}/b.json", "cuda", 8, batch, 20)
            assert r["provider"] == "CUDAExecutionProvider", r
            res[f"{tag}_cuda_{prec}_b{batch}"] = r["img_per_s"]
    return res


def main(models, out):
    ort.preload_dlls()
    torch.set_num_threads(8)
    for row in pd.read_csv(models).to_dict("records"):
        try:
            size = int(row["img_size"])
            model = MewcClassifier(row["backbone"], 96, pretrained=False, img_size=size).eval()
            fused = reparameterize_model(copy.deepcopy(model))
            res = {**row, "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "ok": True}
            with tempfile.TemporaryDirectory() as d:
                export(model, size, f"{d}/plain.onnx")
                res.update(timings(f"{d}/plain.onnx", d, "plain"))
                if str(fused) != str(model):
                    export(fused, size, f"{d}/fused.onnx")
                    res.update(timings(f"{d}/fused.onnx", d, "fused"))
        except Exception as e:
            res = {**row, "ok": False, "error": f"{type(e).__name__}: {e}"[:500]}
            traceback.print_exc()
        with open(out, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
