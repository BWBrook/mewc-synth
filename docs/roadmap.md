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
