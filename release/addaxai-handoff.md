# AddaxAI handoff

For Peter van Lunteren: two Tasmanian classifiers from mewc-synth, packaged for AddaxAI's custom-classifier
interface (`backend/templates/inference_template.py`). They are meant to sit beside TAS-BB-v1, not replace it,
until an external test compares them.

| | Accurate | Fast |
|---|---|---|
| Hugging Face | `bwbrook/tasmania-mewc-eva02-b` | `bwbrook/tasmania-mewc-hgnetv2-b4` |
| Backbone | EVA-02-B at 392 px | HGNetV2-B4 at 288 px, distilled from EVA-02-B |
| Sealed test, macro-F1 | 0.9725 | see the model card |
| RTX 2080 (fp16 ONNX, batch 32) | 87 images/s | 868 images/s |
| 8 CPU cores | 2.2 images/s | 55 images/s |

## Package

Each repository holds `inference.py` (the `ModelInference` class, including `get_tensor` and `classify_batch`
for batching), `model.safetensors`, `config.json`, `classes.txt`, `taxonomy.csv`, `per_class_test.csv` and two
ONNX files that AddaxAI does not need.

- **Environment:** AddaxAI's `pytorch` environment (torch, timm, safetensors, numpy, Pillow). Nothing is
  downloaded at load time (`pretrained=False`).
- **Checked** in a Python 3.11 environment with torch 2.8.0 (CPU), numpy 1.26.4 and the current timm.
  `inference.py`'s batch and single-image paths reproduce the training code's probabilities on held-back
  snips (`release/parity.py`).
- **`get_crop`** reproduces mewc-snip, so crops match the training snips:
  1. MegaDetector's `crop_image` arithmetic;
  2. a LANCZOS resize to 600 x 600;
  3. a JPEG round trip at quality 75.
- **Taxonomy:** `taxonomy.csv` is TAS-BB-v1's with three class renames: `bait` to `camera_bait`,
  `flame_scarlet_robin` to `flame_robin` (genus level, since it merges both robins) and `maned_goose` to
  `maned_duck`. The 96 classes are otherwise the same as TAS-BB-v1's.

## Draft `models.json` entries

The model IDs, the minimum app version and the wording are Peter's to choose.

```json
{
  "model_id": "TAS-BB-v2",
  "friendly_name": "Tasmania (accurate)",
  "emoji": "🇦🇺",
  "env": "pytorch",
  "model_fname": "model.safetensors",
  "description": "Classifies MegaDetector detections from Tasmanian camera traps into the same 96 classes as the 2024 MEWC Tasmania model: all non-volant terrestrial mammals, native and introduced, over 50 birds commonly seen on camera traps, and general classes such as snake, insect and unknown animal. EVA-02 Base backbone at 392 px, trained with the MEWC recipe in PyTorch on 1.04 million snips. On a sealed test of 98,049 snips, macro-F1 is 0.973 (accuracy 0.988). On two projects held out entirely, accuracy is 0.975 and macro-F1 0.78, lower for rare classes. Slower than the fast version; best with a GPU.",
  "description_short": "96 classes • EVA-02 • 2026",
  "developer": "Barry Brook",
  "license": "https://creativecommons.org/licenses/by-nc/4.0/",
  "info_url": "https://huggingface.co/bwbrook/tasmania-mewc-eva02-b",
  "release_date": "2026-11",
  "region": "oceania",
  "domain": "camera_trap"
}
```

The fast model's entry is the same apart from these fields:

- `"model_id": "TAS-BB-v2-fast"`;
- `"friendly_name": "Tasmania (fast)"`;
- the HGNetV2-B4 description and numbers from its model card;
- `info_url` pointing to `bwbrook/tasmania-mewc-hgnetv2-b4`.
