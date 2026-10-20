# mewc-synth

Synthetic camera-trap snips to strengthen low-count classes in the MEWC
classifier (Tasmania, 96 classes). Started 20 October 2026 by Barry W. Brook
with Claude (Opus 5.5). `PLAN.md` is the written test plan: follow it, and
record any departure in its change log with the reason before results are
seen.

## Context

- MEWC training: `zaandahl/mewc-train` (2.0.9 image), base `mewc-flow`
  2.0.4, Keras 3 on JAX, `kimm` backbones. The production Tasmanian model is
  ViT-L/16 at 384 (`VTL`). Snipping: `zaandahl/mewc-snip`, using MegaDetector's
  Lanczos square resize to 600 px and PIL default JPEG (quality 75).
- Compute: Nectar A100 instance `ngx-gpu` (tailnet), boot volume
  `ngx-boot-1`, scratch `/mnt/scratch`, persistent `/mnt/ml-data`. Operating
  rules live in `~/git/nectar-gpu` and `~/work/nectar`.
- On the instance, the Hugging Face cache is `/mnt/scratch/hf`, and project
  data go under `/mnt/ml-data/mewc-synth/` (inputs, splits, checkpoints,
  outputs).

## Guardrails

- Images, weights and checkpoints never go into Git; split lists and hashes do.
- Test and validation snips must never reach the generator or a classifier's
  training set. For pseudo-rare classes this includes the domain LoRA.
- Synthetic snips are always marked (`synth_` prefix plus a manifest row) and
  must never be mixed into records or public datasets unmarked.
- Synthetic and real snips go through identical finishing: 512 px, then
  Lanczos 600 x 600, then PIL default JPEG.
