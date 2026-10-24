"""Inference throughput of a trained mewc-train model, run inside the zaandahl/mewc-train container.
Times model.predict on random images at the model's input size after a warm-up call, on whatever
device JAX sees (run with JAX_PLATFORMS=cpu and a CPU set for the CPU figure).
Usage: python bench_infer.py MODEL.keras OUT.json BATCH N_BATCHES"""
import json
import os
import sys
import time

import jax
import numpy as np
from keras import models


def main(path, out, batch=64, n=20):
    batch, n = int(batch), int(n)
    m = models.load_model(path, compile=False)
    size = m.input_shape[1]
    x = (np.random.default_rng(0).random((batch * n, size, size, 3)) * 255).astype("float32")
    m.predict(x[:batch], batch_size=batch, verbose=0)
    t = time.time()
    m.predict(x, batch_size=batch, verbose=0)
    res = {"model": os.path.basename(path), "params": int(m.count_params()), "img_size": size,
           "device": jax.devices()[0].platform, "batch": batch,
           "img_per_s": round(batch * n / (time.time() - t), 1)}
    with open(out, "w") as f:
        json.dump(res, f)
    print(json.dumps(res))


if __name__ == "__main__":
    main(*sys.argv[1:5])
