---
license: cc-by-nc-4.0
library_name: timm
pipeline_tag: image-classification
base_model: timm/hgnetv2_b4.ssld_stage2_ft_in1k
tags:
  - camera-traps
  - wildlife
  - ecology
  - biodiversity
  - tasmania
  - mewc
  - addaxai
---

# Tasmanian camera-trap classifier, HGNetV2-B4 (fast)

This model classifies MegaDetector animal detections from Tasmanian camera traps into 96 classes. The classes
cover every non-volant terrestrial mammal in Tasmania, native and introduced, more than 50 of the birds most
often seen on camera traps, and a few general classes (snake, skink, insect, camera bait, unknown animal).

It is the fast model of a pair, for CPUs and large batches; it learned in part from the accurate model,
[EVA-02-B](https://huggingface.co/bwbrook/tasmania-mewc-eva02-b), which is about ten times slower and a little
more accurate. Both come from the [mewc-synth](https://github.com/BWBrook/mewc-synth) project, which rebuilt the
[MEWC](https://github.com/zaandahl/mewc) Tasmanian classifier on current backbones with registered, seeded
comparisons, and tested synthetic training images for rare classes.

| | |
|---|---|
| Backbone | HGNetV2-B4, SSLD stage-2 ImageNet-1k weights (timm), at 288 px |
| Head | MEWC head: average-pooled features, dropout, a 512-unit layer, ELU, dropout, 96 classes |
| Sealed test, macro-F1 | **0.9676** (accuracy 0.986; 98,049 snips) |
| Camera-trap snips only, macro-F1 | 0.928 (accuracy 0.984; 71,834 snips, 95 classes) |
| Held-out projects, macro-F1 | 0.743 (accuracy 0.967; 72,169 snips, 62 classes) |
| Speed | 868 images/s on an RTX 2080 (ONNX fp16, batch 32); 55 images/s on 8 CPU cores |

## Files

- `model.safetensors`: float32 weights for PyTorch with timm.
- `config.json`: backbone, timm arguments, input size, normalisation and class order.
- `inference.py`: the AddaxAI inference class. It is self-contained and needs torch, timm, safetensors, numpy
  and Pillow.
- `model.onnx` and `model_fp16.onnx`: input is raw RGB in 0 to 255, NCHW float32 at 288 x 288, with a dynamic
  batch; output is class probabilities in the order of `classes.txt`. Use fp16 on GPU only; on an RTX 2080 it
  agreed with fp32 on every one of 2,000 predictions (largest probability difference 0.009).
- `classes.txt` (model output order) and `taxonomy.csv` (class, order, family, genus, species).
- `per_class_test.csv`: F1, precision, recall and support for every class on the sealed test snips.

## Preprocessing

The model was trained on 600 x 600 snips made by [mewc-snip](https://github.com/zaandahl/mewc-snip):

1. Crop the MegaDetector box as `megadetector.visualization.visualization_utils.crop_image` does.
2. Resize the crop to 600 x 600 with LANCZOS, ignoring aspect ratio.
3. Store the snip as a JPEG at quality 75.

At inference, the snip is resized to 288 x 288 by bilinear interpolation without antialiasing and normalised
with the backbone's mean and standard deviation (0.485, 0.456, 0.406; 0.229, 0.224,
0.225). `inference.py` does all of this, including the JPEG round trip. If you feed the ONNX model
yourself, reproduce these steps.

## Use

**AddaxAI:** the files follow AddaxAI's custom-classifier template and run in its `pytorch` environment.

**Python:**

```python
from pathlib import Path
from PIL import Image
from huggingface_hub import snapshot_download
import importlib.util

d = Path(snapshot_download("bwbrook/tasmania-mewc-hgnetv2-b4"))
spec = importlib.util.spec_from_file_location("mewc_inference", d / "inference.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
clf = m.ModelInference(d, d / "model.safetensors"); clf.load_model()
crop = clf.get_crop(Image.open("IMG_0001.JPG"), (0.31, 0.42, 0.18, 0.25))  # MegaDetector box, normalised x, y, w, h
top = max(clf.get_classification(crop), key=lambda r: r[1])
```

## Training

- **Data:** 1,040,627 snips of the 96 classes, from Tasmanian camera-trap projects. For birds that camera traps
  rarely photograph well, eBird and iNaturalist photographs are added (13% of training snips).
  - Classes with more than 30,000 snips were sampled down to 30,000; the smallest class has 65.
  - Near-duplicate snips (frames of one event) were grouped by DINOv2 similarity and perceptual hash, and
    each group went wholly to one split.
  - Two whole projects were held out as a second test set.
  - 135 validation and test labels were corrected after a blind review.
- **Recipe:** mewc-train's, reimplemented in PyTorch (`mewc_torch`):
  - 10 frozen-backbone epochs, then 30 progressive epochs with the top two stages unfrozen, and dropout and
    RandAugment magnitude stepped up in four stages;
  - 4,000 snips per class per epoch;
  - categorical focal loss, AdamW with AMSGrad, learning rate 1e-4 (300 warm-up steps) decaying to 2e-5, bf16,
    batch 64 on one A100.
- **Distillation:** the loss is half focal loss on the labels and half cross-entropy against the class
  probabilities of EVA-02-B (the mean of three EVA-02-B models trained without the validation snips).
  - On validation, over four paired seeds, distillation raised macro-F1 by +0.0029 [+0.0001, +0.0055]
    (registered test).
  - On the sealed test the gain was +0.006: 0.9668 ± 0.0009 over four distilled seeds, against
    0.9605 ± 0.0022 over eight undistilled ones.
- **This release** is a refit on train plus validation with the final epoch kept. Seed 1.
- **Model choice:** HGNetV2-B4 was chosen from 30 backbones by a registered rule. On 8 CPU cores, ties went
  to the cheaper model. EfficientNetV2-S and FastViT-SA24 tied with it at medium scale.

The full record (plan, registered decision rules, every run and evaluation) is in
[mewc-synth](https://github.com/BWBrook/mewc-synth): see `PLAN.md`, `docs/plan-booking-3.md` and
`ops/2026-11-24/test_eval_kd.json`.

## Evaluation

The sealed test snips and the held-out-project snips were never used for training, model choice or tuning, and
were scored once. Macro-F1 is the mean of per-class F1 over the classes present.

| Test set | Labels | Macro-F1 | 25 low-count classes | Accuracy |
|---|---|---|---|---|
| Sealed test (98,049 snips, 96 classes) | corrected | 0.9676 | 0.928 | 0.986 |
| | original | 0.9637 | 0.916 | 0.985 |
| Camera-trap snips of the sealed test (71,834) | corrected | 0.928 | | 0.984 |
| Held-out projects (72,169 snips, 62 classes) | corrected | 0.743 | 0.500 | 0.967 |
| | original | 0.767 | 0.538 | 0.967 |

- **Four distilled train-only seeds** (validation excluded from training) score 0.9668 ± 0.0009 on the
  sealed test, against 0.9689 on validation. So repeated model choice on validation biased the estimate little.
- **The weakest classes** on the sealed test are small mammals:
  - house mouse, F1 0.77;
  - white-footed dunnart, 0.77;
  - antechinus, 0.81;
  - long-tailed mouse, 0.83.
- **Held-out projects** test generalisation to new sites and camera set-ups.
  - Accuracy holds at 0.967, but macro-F1 falls to 0.74, and to 0.50 for the low-count classes.
  - Many classes are rare there: 24 of the 62 have fewer than 10 snips, so per-class F1 is noisy.
  - Expect lower per-class performance on new deployments than on the sealed test, mostly for rare classes.
  - Where rare classes matter and time allows, use EVA-02-B (0.779 and 0.601).

### Confidence thresholds

Share of detections kept, and accuracy of those kept, when the top probability must reach a threshold.

| Threshold | Kept, validation (4 train-only seeds) | Accuracy kept | Kept, held-out projects | Accuracy kept |
|---|---|---|---|---|
| none | 1.000 | 0.982 | 1.000 | 0.967 |
| 0.5 | 0.990 | 0.988 | 0.975 | 0.981 |
| 0.7 | 0.972 | 0.993 | 0.943 | 0.991 |
| 0.8 | 0.959 | 0.995 | 0.923 | 0.995 |
| 0.9 | 0.933 | 0.998 | 0.887 | 0.998 |
| 0.95 | 0.900 | 0.999 | 0.845 | 0.999 |

A threshold of 0.8 to 0.9 keeps about 89 to 96% of detections at 99.5% accuracy or better, and sends the rest
for review.

## Limitations

- **No external test yet.** The plan made a test on recent, independently labelled snips the release gate.
  That test has not been run. The held-out projects are the closest substitute.
- **No comparison with the earlier MEWC Tasmanian model** (TAS-BB-v1 in AddaxAI). That model was trained on
  all of the library's snips, including these test snips, so only an external test can compare the two
  fairly. The two models' published accuracy figures come from different splits and are not comparable.
- **Closed set.** Species outside the 96 classes are assigned to the nearest class, often with high
  confidence. The `unknown_animal` class catches only some of them.
- **Sites can leak across splits.** Snips carry no site or date metadata, so near-duplicate grouping stands in
  for site grouping. Snips from one site on different days can fall in both training and test, which is why
  the held-out-project scores are lower.
- **`flame_robin`** merges flame and scarlet robins.

## Licence

- **Weights:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), as for the earlier MEWC
  Tasmanian model. The training images include photographs from eBird and iNaturalist, as well as camera-trap
  images from several projects.
- **`inference.py`:** MIT.
- **Backbone:** the pretrained HGNetV2 weights are Apache 2.0 (timm).

## Authors and AI roles

- **Barry W. Brook** (University of Tasmania) directed the project and set its questions and decision
  rules. He reviewed labels and synthetic images blind, and reviews every release.
- **Claude (Anthropic; Opus 5.5, in Claude Code)** contributed:
  - the experimental design;
  - the `mewc_torch` reimplementation and the release code;
  - every training run and evaluation;
  - this card.
- **Compute:** the ARDC Nectar Research Cloud (A100 vGPU).
- **Prior work:** the training recipe follows MEWC, the crop code follows MegaDetector, and the inference
  class follows AddaxAI's template (Peter van Lunteren).

## Citation

Until the mewc-synth paper is out, cite the repository and the MEWC preprint:

- mewc-synth: https://github.com/BWBrook/mewc-synth
- MEWC (EcoEvoRxiv): https://ecoevorxiv.org/repository/view/6405/
