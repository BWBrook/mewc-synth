"""End-to-end check of a release bundle in an environment like AddaxAI's pytorch one (Python 3.11, torch 2.8,
numpy 1.26, current timm), without mewc_torch: inference.py's batch API on every listed snip and its
single-image API on the first 200 are compared with reference probabilities from mewc_torch.predict on the same
snips (REF_PREFIX.csv and REF_PREFIX_probs.npy). The first 200 snips also go through get_crop with the whole
snip as the box, which adds the 600 px LANCZOS resize and the JPEG round trip of a fresh crop; that is reported
as top-1 agreement, not required to match. Passing needs identical top-1 predictions and probabilities within
5e-3: separate CPU passes of the same model over the same snips differ by up to about 1e-3 (seen 24 November
2026), while inference.py and mewc_torch give identical outputs on identical input tensors.
Usage: python release/parity.py BUNDLE_DIR REF_PREFIX OUT_JSON"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main(bundle, ref_prefix, out):
    bundle = Path(bundle)
    spec = importlib.util.spec_from_file_location("bundle_inference", bundle / "inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    inf = mod.ModelInference(bundle, bundle / "model.safetensors")
    inf.load_model()
    names = inf.get_class_names()
    assert [names[str(i + 1)] for i in range(len(names))] == inf.classes
    paths = pd.read_csv(f"{ref_prefix}.csv").path.tolist()
    ref = np.load(f"{ref_prefix}_probs.npy")
    order = lambda res: np.array([[p for _, p in r] for r in res])
    got = []
    for i in range(0, len(paths), 32):
        got += inf.classify_batch(np.stack([inf.get_tensor(Image.open(p)) for p in paths[i:i + 32]]))
    batch = order(got)
    single = order([inf.get_classification(Image.open(p)) for p in paths[:200]])
    crop = order([inf.get_classification(inf.get_crop(Image.open(p).convert("RGB"), (0.0, 0.0, 1.0, 1.0))) for p in paths[:200]])
    res = {"bundle": bundle.name, "n": len(paths), "device": str(inf.device),
           "batch_max_abs_diff": float(np.abs(batch - ref).max()),
           "batch_top1_agreement": float((batch.argmax(1) == ref.argmax(1)).mean()),
           "single_max_abs_diff_vs_batch": float(np.abs(single - batch[:200]).max()),
           "get_crop_top1_agreement": float((crop.argmax(1) == ref[:200].argmax(1)).mean()),
           "get_crop_max_abs_diff": float(np.abs(crop - ref[:200]).max())}
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    if res["batch_max_abs_diff"] > 5e-3 or res["batch_top1_agreement"] < 1 or res["single_max_abs_diff_vs_batch"] > 1e-4:
        raise SystemExit("PARITY FAILED")


if __name__ == "__main__":
    main(*sys.argv[1:4])
