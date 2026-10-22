# Roadmap

## Later: MEWC model selection and tuning study (proposed 20 October 2026)

Barry: "a good time for us to (later) really go to town on model selection and
parameter tuning for the various CNN and ViT that mewc exposes, along with a
rigorous hyperparameter [search]: looking ultimately for a 'pareto optimal'
model that balances performance vs inference time." Context: Barry holds
about 200,000 Nectar service units for 2026. The A100 reservation costs
3.596 SU/h, so compute is effectively unconstrained; wall-clock time and GPU
availability are the limits.

Prerequisites, most already under way in `mewc-synth`:

- Fixed, grouped train, validation and test splits (near-duplicate groups),
  with the test set touched once, at the end.
- The `mewc-train` fixes (real dropout schedule, best-model save,
  reproducible sampling) merged and rebuilt.
- A modernised, faster training stack. Parallel decode gave no gain on the
  A100 (compute-bound), so the levers are bf16 mixed precision, larger
  batches and newer Keras/JAX/kimm. Numerical parity on real data must be
  checked before adopting any of them.

Design sketch:

- Objectives: macro-F1 (and tail-class F1, or F1 weighted by management
  priority) against measured inference latency and throughput **on the
  deployment hardware** (field laptop CPU, workstation GPU or server). The
  Pareto front depends on the hardware, so measure there, at the batch sizes
  `mewc-predict` uses.
- Search space:
  - Backbones: every `kimm` family exposed (EfficientNetV2 B0 to XL,
    ConvNeXt Pico to Large, ViT Tiny to Large).
  - Input size.
  - Fine-tuning: blocks unfrozen.
  - Optimiser: learning rate and schedule, optimiser, weight decay.
  - Data: class samples, RandAugment depth and magnitude, dropout ramp.
  - Loss: focal-loss settings.
  - Training budget: batch size, epochs.
  - Also newer backbones not in kimm 0.2.5, such as DINOv2/v3 ViTs, EVA-02,
    ConvNeXt V2 and BioCLIP 2 image encoders.
- Method: multi-objective, multi-fidelity search (Optuna NSGA-II with ASHA or
  Hyperband pruning on reduced class samples and epochs), with the best
  candidates promoted to full fidelity and several seeds.
- Inference-side levers belong on the same front: knowledge distillation from
  a large teacher (VTL) into small students, int8 quantisation, and
  ONNX/TensorRT export.
- Also report calibration (ECE) and open-set behaviour (`unknown_animal`),
  since MEWC relies on confidence thresholds.
- Ordering with `mewc-synth`: run the synthetic-data study on a fixed
  screening recipe first. Then include the synthetic share as one factor in
  the full search.

## Later: diversity-aware curation of the real snip library (proposed 22 October 2026)

Barry: "A lot of my snips on the common classes likely have very high
similarity scores (statistically), and so are contributing little to
classifier training fidelity in practice. A lot of wasted compute perhaps,
and also, misleading test statistics. I wonder if we should do a pass over
all snips at some point, and create a cleaner training, validation and test
pool that tries to improve diversity in those real images too."

What the 1.07 M working set already shows (DINOv2 groups at tau 0.90):

- 30.7% of snips sit in multi-snip near-duplicate groups. One per group
  keeps 800,755 snips (75%); three per group keeps 881,553 (83%).
- By class: forester kangaroo 40%, unknown animal 40%, forest raven 37% and
  Cape Barren goose 37% of their (sampled) snips are in bursts; pademelon 8%,
  wombat 14%.
- These are underestimates for abundant classes. They were randomly
  subsampled to 30,000, which dilutes bursts (for example 30k of 689k
  pademelons). Full-library redundancy will be much higher.

Proposed study:

1. Embed all 4.07 M snips (DINOv2-base, about 40 min on the A100) and group
   near-duplicates within classes. Use approximate nearest neighbours (for
   example FAISS) for the largest classes. Calibrate tau visually, as on
   2026-10-21.
2. Build production train, validation and test splits by group (no burst or
   repeat-visit cluster straddles splits), with a small group-disjoint
   validation set for model selection.
3. Quantify how optimistic the existing random-split accuracy is. Score the
   current production model on group-held-out test snips, against its
   reported figures.
4. Make sampling in `mewc-train` group-aware (draw a group, then a member),
   so `CLASS_SAMPLES` per epoch buys diversity rather than repeats. Also
   consider pruning to k per group to save storage and compute. Measure
   whether accuracy holds at the same or lower compute.
5. Fold in the related data issues found on 2026-10-21:
   - 114 cross-class duplicate images to adjudicate.
   - Class-correlated JPEG quality (q95 source).
   - Photo-dominated bird classes (eBird) and how they are balanced against
     camera-trap snips.

This pairs naturally with the model-selection and Pareto study above: both
need the same group-disjoint splits.
