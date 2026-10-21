"""Shared paths and image helpers for mewc-synth."""
import io
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path("/mnt/ml-data/mewc-synth")
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_FP16 = "madebyollin/sdxl-vae-fp16-fix"
GEN_RES = 512
SNIP_RES = 600

# libjpeg's standard luminance quantisation table (quality 50), in natural order
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])


def _scaled_table(q):
    scale = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip((_STD_LUMA * scale + 50) // 100, 1, 255)


_TABLES = {q: np.sort(_scaled_table(q)) for q in range(1, 101)}


def jpeg_quality(img):
    """Estimate libjpeg quality from the luminance table; None if not a JPEG.
    Compares sorted values, so it does not depend on zigzag or natural order."""
    tables = getattr(img, "quantization", None)
    if not tables or 0 not in tables:
        return None
    luma = np.sort(np.asarray(tables[0], dtype=np.int64))
    return min(_TABLES, key=lambda q: np.abs(_TABLES[q] - luma).sum())


def chroma(img):
    """Mean absolute channel difference on a 64 px thumbnail: near 0 for IR/greyscale."""
    a = np.asarray(img.convert("RGB").resize((64, 64), Image.BILINEAR), dtype=np.float32)
    return float((np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])).mean() / 2)


def finish_snip(img512, quality=75):
    """Finish a generated snip as real snips are finished: Lanczos square resize to 600 px,
    then PIL JPEG with default settings at the given quality (mewc-snip writes 75; one source wrote 95)."""
    buf = io.BytesIO()
    img512.convert("RGB").resize((SNIP_RES, SNIP_RES), Image.LANCZOS).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
