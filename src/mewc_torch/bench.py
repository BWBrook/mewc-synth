"""Inference throughput of an exported ONNX model in onnxruntime, the deployment path: CPU with a fixed
thread count (default 8, a desktop stand-in) or CUDA when available. Random input at the model's size.
Usage: python -m mewc_torch.bench MODEL_ONNX OUT_JSON [DEVICE cpu|cuda] [THREADS] [BATCH] [N_BATCHES]"""
import json
import sys
import time

import numpy as np
import onnxruntime as ort


def main(path, out, device="cpu", threads=8, batch=8, n=8):
    threads, batch, n = int(threads), int(batch), int(n)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(path, so, providers=providers)
    size = sess.get_inputs()[0].shape[-1]
    x = (np.random.default_rng(0).random((batch, 3, size, size)) * 255).astype(np.float32)
    sess.run(None, {"image": x})
    t = time.time()
    for _ in range(n):
        sess.run(None, {"image": x})
    res = {"model": path, "device": device, "threads": threads, "batch": batch, "img_size": size,
           "img_per_s": round(batch * n / (time.time() - t), 1), "provider": sess.get_providers()[0]}
    with open(out, "w") as f:
        json.dump(res, f)
    print(json.dumps(res))


if __name__ == "__main__":
    main(*sys.argv[1:7])
