# Plan: MEWC model study and production refit, in phases

Drafted 25 October 2026 by Claude (Opus 5.5) at Barry's request, revised the same day after an
independent critical review by a Claude Sonnet 5.5 subagent (findings at the end), and re-cut into
phases on 26 October after Barry booked two simultaneous 11-day A100 slots (29 October 00:00 to
8 November 23:59 UTC; 22 of his 30 GPU-days a month). Barry, 26 October: "We can stage the plan in
phases, with this block being phase 1. It's no drama, we're not in a rush: plenty of future bookings
as/when we need them", and "We don't need to wed to `kimm` either ... Happy to move to a more
general / updatable format for model API." Each stage is registered in `PLAN.md` before its runs.

## Goal

Two production Tasmanian classifiers, each better than what Barry has now and each backed by
experiments: an **accurate** model (GPU users) and a **deployable** model (CPU and small-GPU users);
real evidence on synthetic snips for genuinely rare classes; a modern, updatable training stack for
MEWC; and two write-ups (synthetic share; accuracy against cost).

- **Phase 1** (this booking): evaluation set, training stack, shortlist, recipe, synthetic re-test.
- **Phase 2** (a later booking): production refit, external test, release.
- **Phase 3** (later): new classes, scale metadata, distillation and other extensions.

## Starting point (26 October)

- **Infrastructure.** Boot volumes `ngx-boot-1` and `ngx-boot-2` carry the environment; `ml-data`
  (550 GiB, holds the 183 GB library zip) and `ml-data-2` (250 GiB) carry data at identical paths.
  The parked instances resize into the new reservation flavours in about a minute each. deep-server
  merges and evaluates unattended.
- **Synthetic snips.** Look-alike translations at a 20% share of draws raised F1 for four classes cut
  to 50 real snips by +0.029 [+0.004, +0.053] on VTL (eight seeds) and on VTB (seven); 50% and 80%
  help less or harm, probably through a VAE fingerprint. Scored on split v1's test set throughout,
  with the 20% share chosen after seeing it, so phase 1 re-tests on sealed v2 data.
- **Pilot** (mewc-train defaults, screen scale, `logs/2026-10-25-pilot-summary.csv`): VTL 0.949,
  ConvNeXt-B 0.945 at twice VTL's A100 throughput and four to five times its CPU throughput,
  ConvNeXt-S and VTB 0.937, EfficientNet B0 0.876.
- **Split v2, built 26 October** (`deep-server:/data/mewc/curation/v2/`, `src/split_v2.py`): 3.99 M
  snips of the 96 classes; training 3.59 M, validation 29k, test 98k, held-out projects (`mtwt`,
  `hr`) 72k; 154k training snips purged as near neighbours of validation or test; training pool
  capped at 40,000 per class, 1.01 M snips (about 47 GB). Details in `logs/2026-10-26-split-v2.md`.

## Model stack: PyTorch and timm for training, ONNX for deployment

kimm 0.2.5 (October 2024) is the newest kimm and its zoo is fixed. **timm** is the largest maintained
image-model library: every family tried so far plus DINOv2 and DINOv3, EVA-02, ConvNeXt V2,
MobileNetV4, FastViT and EfficientViT, and BioCLIP 2 through `open_clip`. Models export to **ONNX**,
which `onnxruntime` runs on CPU and GPU on any platform, so deployment stops depending on one
framework. Phase 1 builds `mewc-torch`, a trainer that reproduces the MEWC recipe (per-class sampling,
staged RandAugment and dropout, frozen then partial fine-tuning, focal loss) with bf16, larger
batches and checkpoint resume. **Parity gate on day 1**: ViT-B and ConvNeXt-B trained by both stacks
at screen scale (the mewc-train side already exists from the pilot); `mewc-torch` replaces mewc-train
for phase 1 only if macro-F1 agrees within seed noise. mewc-train stays the default-recipe anchor.
Moving `mewc-predict` and AddaxAI to ONNX is phase 2 work, for Barry and Zach to steer.

## Evaluation protocol

- **Split v2** as above. The 0.90 groups keep near-duplicate pairs on one side; the purge removes
  training snips within cosine 0.88 of validation or test (at 0.85 the purge band was mostly different
  animals in similar poses, and it would have removed 59 to 79% of some photo-heavy classes); two
  whole source projects form a project-disjoint test stratum.
- **Label check before sealing.** On day 1 a strong model scores validation and test; confident
  disagreements go to Barry (decision 4).
- **Sealed test.** Phase 1 runs score validation only; test logits are computed once, in phase 2.
  Validation during training uses 100 snips per class per epoch; full validation once per run.
- **Metrics.** Macro-F1 (all, camera-trap, rare-class subset), per-class F1, calibration (ECE),
  open-set behaviour on the library's out-of-model classes (`not_in_model`, Gippsland-only and
  candidate classes) with risk-coverage curves, and throughput of the ONNX model in `onnxruntime` on
  the A100, 8 CPU cores and decision 1's hardware.
- **Statistics.** Paired seeds, two-level bootstrap, the amended no-harm rule, and a default-recipe
  anchor at every scale.

## Phase 1: 26 October to 8 November

### Stage 0, before the booking (26 to 28 October, CPU)

- Done: split v2, purge and pool lists.
- `mewc-torch` trainer, ONNX export and `onnxruntime` benchmark; per-GPU job runner; run registry
  (config, seed, metrics, timings) in one Parquet file; deep-server evaluation for the new runs.
- The registration of phase 1's stages in `PLAN.md`.

### Booking (29 October to 8 November), GPU-hours at mewc-train speeds

| Days | GPU 1 | GPU 2 | GPU-h |
|---|---|---|---|
| 1 | pool extraction; parity runs (ViT-B, ConvNeXt-B, two seeds) | pool extraction; label-check scoring; Stage A retrain on v2 | 20 |
| 2 to 4 | architecture race: about 25 timm backbones, one seed, screen scale, including the modern families; frozen-feature probes for the largest | synthetic pipeline for the 25 low-count classes (adapters, translations, identity and shortcut probes); Barry's blind review on days 3 to 4 | 60 |
| 4 to 5 | per-family mini-search, top four families, about 10 trials each | synthetic re-test: four pseudo-rare classes on v2, eight paired seeds | 70 |
| 5 to 9 | recipe tuning for the shortlisted accurate and deployable models: Sobol 40, fANOVA, top 8 at three seeds | synthetic re-test: real low-count classes, eight paired seeds; fidelity check (four configurations, screen against medium) | 190 |
| 9 to 11 | medium-scale confirmation of the best two recipes per model, two seeds | one production-scale run of the accurate model on the interim recipe, if it fits | 120 |
| | | Total | about 460 of 528 |

The shortlist rule is stated in advance: the accurate model has the highest validation macro-F1 at any
cost; the deployable model has the highest macro-F1 above 15 images/s on 8 CPU cores in
`onnxruntime` (to be revised once decision 1 names the target hardware). Ties within the seed interval
go to the cheaper model. bf16 and larger batches in `mewc-torch` should roughly halve these times;
the slack goes to more seeds and the production run.

**Phase 1 delivers:** the shortlist with its frontier, tuned recipes confirmed at medium scale, the
sealed synthetic re-test for pseudo-rare and real rare classes, `mewc-torch` with its parity record,
and the fidelity evidence that sets phase 2's run lengths.

## Phase 2: production refit and release (next booking)

- Production-scale runs: accurate model two seeds, deployable model two seeds, default-recipe anchor;
  synthetic snips for rare classes if phase 1 supports them (distinct synthetic snips scaled with
  budget, about 20 views each at most).
- Test logits once; the external test as the release gate; comparison with Barry's current VTL on the
  external test only (it saw every library snip).
- Refit on all data with the fixed recipe and epoch count; ONNX export; `mewc-predict` and AddaxAI
  integration; model card with open-set and confidence-threshold advice; upstream release.
- Stretch: distillation of the accurate model into the deployable one.
- Write-ups: synthetic minority data (share curve, mechanism, sealed re-test, real rare classes) and
  accuracy against cost in MEWC.

## Phase 3: later

New classes (the candidate `too_few` classes, the flame and scarlet robin split) with synthetic
support; crop size as metadata from `mewc-snip` (the scale cue, `docs/roadmap.md`); the paper's
future directions (hierarchical classifiers for sex, disease and morph; active-learning loops).

## Risks

- **`mewc-torch` parity**: if it fails on day 1, phase 1 continues on mewc-train with its kimm zoo,
  and the modern families are probed on frozen features only.
- **Rank transfer across scales**: the fidelity check.
- **Rare-class power**: a pooled endpoint, eight seeds, the minimum detectable effect stated in
  advance; per-class results reported, not used for decisions.
- **Storage**: the pool (47 GB) fits both volumes.
- **Lease end**: instances are deleted at the end; park by `before_end_lease`, with the deep-server
  guard as a backstop.

## Decisions for Barry

1. **Deployment hardware** for the throughput axis. Recommendation: an 8-core CPU and a consumer GPU
   (RTX 2080 class, as in the MEWC paper), plus the A100.
2. **External test** (needed for phase 2, not phase 1): labelled snips from recent services not in
   `data_600px.zip`, a few thousand across cameras including rare species. Recommendation: yes.
3. **Synthetic review**, days 3 to 4 (31 October to 1 November): a blind review of synthetic snips for
   the rare classes, about 30 minutes. Recommendation: yes.
4. **Label check**, days 1 to 2: confidently disagreed validation and test snips, likely a few hundred,
   an hour or two. Recommendation: yes, rare classes at least.
5. **Class list**: the 96 classes for phases 1 and 2; new classes in phase 3. Recommendation: as stated.
6. **Training stack**: `mewc-torch` (PyTorch, timm, ONNX) as above, subject to the parity gate.
   Recommendation: yes; it is also the route to the modern architectures.

## Independent review (25 October) and what changed

A Claude Sonnet 5.5 subagent reviewed the first draft against the logs, the MEWC paper and the
`mewc-train` code. All findings were accepted:

- **Leakage and test sealing**: group-disjoint split with a purge buffer, a project-disjoint stratum,
  validation-only selection, test sealed until phase 2.
- **Synthetic transfer**: views per synthetic snip grow about 25 times from screen to production
  scale; rare-class power is low; generators must be retrained on v2 training data. Now: synthetic
  count scales with budget, Stage A retrained, eight paired seeds, pooled endpoint.
- **Recipe search**: single-seed successive halving ranks noise and is biased by ramped
  regularisation. Now: Sobol, fANOVA, paired seeds, fixed compute.
- **Fidelity**: hyperparameter rank transfer is the risk; tested directly.
- **Compute**: medium scale is 20% of production; VTL production is 60 to 75 h; resume and early
  production runs.
- **Baselines**: default-recipe anchor at every scale; separate accurate and deployable criteria;
  external test as gate.
- **Data**: pools extracted as JPEG folders with a 40,000 cap.
- **Shortlist bias and benchmarking**: per-family mini-search before shortlisting; throughput measured
  on the deployed format.
- **Missing pieces added**: open-set behaviour, label review, refit on all data, early synthetic
  generation.
