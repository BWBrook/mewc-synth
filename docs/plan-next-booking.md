# Plan for the next block booking: two A100s, about four weeks

Drafted 25 October 2026 by Claude (Opus 5.5) at Barry's request ("lay out a clear plan for
post-Sunday night work, when I block book 2 x A100 for the next extended period"), then revised after
an independent critical review by a Claude Sonnet 5.5 subagent the same day (its findings and what
changed are listed at the end). It turns the design sketch in `docs/roadmap.md` into a schedule. Each
stage is registered in `PLAN.md` before its runs.

## Goal

Two production Tasmanian classifiers, each better than what Barry has now and each backed by
experiments: an **accurate** model (GPU users) and a **deployable** model (CPU and small-GPU users).
Plus real evidence on synthetic snips for genuinely rare classes, `mewc-train` improvements upstream,
and two write-ups (synthetic share; accuracy against cost). Success criteria are fixed per model
before stage 5, with the external test as the gate (decision 3).

## Starting point (25 October)

- **Infrastructure.** Boot volumes `ngx-boot-1` and `ngx-boot-2` carry the environment; `ml-data`
  (550 GiB, holds the 183 GB library zip) and `ml-data-2` (250 GiB) carry data at identical paths.
  Relaunch takes about two minutes per machine. deep-server merges and evaluates unattended.
  `zaandahl/mewc-train:2.0.10` has working progressive dropout.
- **Synthetic snips.** Look-alike translations at a 20% share of draws raised F1 for four classes cut
  to 50 real snips by +0.029 [+0.004, +0.053] on VTL (eight seeds) and on VTB (seven), with no harm
  elsewhere; 50% and 80% help less or harm, probably through a VAE fingerprint. These results were
  scored on split v1's test set throughout, and the 20% share was chosen after seeing it, so they
  are re-confirmed on sealed v2 data in stage 4.
- **Curation.** 4.07 M snips embedded; greedy thinning at cosine 0.90 keeps 75%.
- **Recipe.** Every model before 2.0.10 trained with dropout fixed at 0.1; the shipped recipe has
  never been tuned with its regularisation ramp working.
- **Pilot (screen scale, defaults).** ViTs lead (VTL 0.949, VTB 0.937, ViT-S 0.930 macro-F1);
  ConvNeXt-T 0.925 at ViT-S's GPU speed and faster on CPU; EfficientNet B0/B2 0.87 to 0.89. With
  only two blocks unfrozen and a short schedule the convolutional models are probably underfit, so
  defaults favour ViTs and shortlisting must come after a fair per-family search.
- **Cost per run, A100 at batch 16** (draws = classes × samples per class × epochs):

  | Scale | Draws | VTB | VTL |
  |---|---|---|---|
  | screen (400 × 12) | 0.46 M | 1.3 h | 3 h |
  | medium (1,600 × 20) | 3.1 M | about 8 h | 15 to 20 h |
  | production (4,000 × 40) | 15.4 M | about 30 h | 60 to 75 h (88 h on the slower GPU) |

## Evaluation protocol (fixed in stage 0)

- **Split v2, built to resist leakage.** Same-event groups from the 0.90 embedding graph; validation
  and test drawn per class by whole groups, at most 20 snips per test group; then **every training
  snip within cosine 0.85 of any validation or test snip is purged** from training, so neighbouring
  burst frames cannot straddle the boundary. One or two whole source projects (filename prefixes
  with broad species coverage) are held out as a project-disjoint test stratum, the nearest thing to
  new cameras the metadata allows.
- **Label check before sealing.** A strong model scores validation and test; confident disagreements
  (top-1 against the label with probability above 0.9) go to Barry for review (decision 6). One wrong
  label moves a 19-snip class's F1 by about 0.05.
- **Sealed test.** Stages 1 to 4 use validation only: runs score validation, never test. Test logits
  are computed once, in stage 5. Validation during training uses 100 snips per class per epoch; the
  full validation set is scored once per run.
- **Metrics.** Macro-F1 (all, camera-trap, rare-class subset), per-class F1, calibration (ECE), and
  **open-set behaviour**: the library already holds an out-of-distribution set (the 53
  `not_in_model` classes, the Gippsland-only classes and the candidate new classes), so report how
  often they receive confident in-model labels and risk-coverage curves for the confidence
  thresholds MEWC users apply.
- **Statistics.** Paired seeds, two-level bootstrap (test snips and seeds), the amended no-harm rule.
  A default-recipe **anchor** runs at every scale, so each change is measured against the shipped
  recipe on identical data.
- **Throughput** measured end to end in the `mewc-predict` container (its own backend and
  preprocessing) on the A100, an 8-core CPU and the hardware of decision 2; `bench_infer.py`
  (JAX, random tensors) is only a quick proxy.

## Stage 0: before the booking (CPU, this week)

1. Split v2, purge, project-disjoint stratum; out-of-distribution set; label-check list for Barry.
2. Training pool: the purged v2 training split, capped at 40,000 snips per class by greedy thinning
   (screen and medium runs draw at most 4,800 and 32,000 per class), extracted as JPEG folders to both
   volumes. A larger pool (up to 120,000 per class) is extracted on `ml-data` only for stage 5.
3. `mewc-train`: checkpoint resume for long runs (a 70 h run cannot restart from zero), bf16 option,
   per-family defaults for blocks to unfreeze, validation subsampling per epoch.
4. Automation: per-GPU job lists, a run registry (config, seed, metrics, timings) in one Parquet file,
   deep-server evaluation; the `mewc-predict` throughput harness.
5. Synthetic pipeline on v2: retrain the Stage A domain adapter on the v2 training split only (the
   current one saw split v1 training snips that v2 may put in test).
6. Register stages 1 and 2 in `PLAN.md`.

## Stage 1: speed levers and fidelity (days 1 to 4)

- **Levers, tested on VTL** (it dominates cost) at screen scale, two seeds: batch 16 against 64 with
  scaled learning rate; bf16 against float32. Adopt a lever only if macro-F1 holds within ±0.003 and
  throughput rises. Budgets below assume no speed-up; any gain becomes slack.
- **Fidelity of hyperparameter rankings** (the real risk, not architecture rankings): six to eight
  recipe configurations on VTB at screen and medium scale, two seeds each. Rank agreement decides how
  much stage 3 can trust screen scale. One production-scale pair (default against the best screen
  configuration) runs later, in stage 3.
- In parallel (GPU 2): synthetic generation for stage 4 starts now, off the critical path: Stage B
  adapters for the 25 low-count classes on v2 training data, translations, identity and shortcut
  probes; Barry's blind review of a sample (decision 5) in days 3 to 5.

## Stage 2: fair architecture shortlist (days 4 to 8)

- **Race:** all 16 kimm backbones plus EfficientNet input-size variants, one seed, reusing the 11
  pilot models where the recipe matches.
- **Per-family mini-search:** the best four families get about 10 trials each over blocks to
  unfreeze, learning rate and schedule length, so convolutional models are not judged on ViT
  defaults.
- **Shortlist rule, stated in advance:** the accurate model is the highest validation macro-F1 at
  any cost; the deployable model is the highest macro-F1 above 15 images/s on 8 CPU cores in
  `mewc-predict`. Ties within the seed interval go to the cheaper model.

## Stage 3: recipe tuning (days 8 to 16)

For each shortlisted model, at fixed compute per trial (samples per class and stage lengths are budget,
not hyperparameters):

1. **Sobol sweep**, 40 trials, one seed, over learning rate, weight decay, blocks to unfreeze, dropout
   ramp end points, RandAugment layers and magnitudes, focal-loss settings. Pruning only at stage
   boundaries (early epochs favour weak regularisation under a ramped schedule).
2. **fANOVA** to find the dimensions that matter; freeze the rest at defaults.
3. **Top eight** configurations at three paired seeds, screen scale.
4. **Top three** at medium scale, three seeds, against the default anchor; plus the one production-scale
   pair on VTB (default against best), which tests whether screen-scale optima under-regularise at
   production scale.

## Stage 4: data factors (days 8 to 16, on whichever GPU stage 3 leaves free)

1. **Synthetic snips, sealed re-test and real rare classes.** VTB at screen scale, eight paired seeds,
   one pooled pre-registered endpoint (macro-F1 over the low-count classes on v2 validation) with its
   minimum detectable effect stated in advance. The four pseudo-rare classes are re-run on v2 as the
   replication of the first lease's result. Distinct synthetic snips scale with the training
   budget: at most about 20 views per synthetic snip, which at production scale means about 1,200 per
   class at a 20% share (otherwise the share is lowered).
2. **Camera-trap weighting** for photo-dominated birds as a recipe option (it helped two of nine birds
   in the first lease's bird arm).
3. **Thinned against random pool**, one VTB screen pair only: thinning saves no GPU time (cost is
   samples per class), so it matters only if it changes accuracy.

## Stage 5: production refit and release (days 14 to 28)

- **Day 14:** one VTL production seed starts on the interim recipe (stage 3 top configuration), so
  the longest run is not on the critical path at the end.
- **Days 17 to 24:** the final accurate-model seed on the final recipe and data; the deployable
  model, two seeds; the default-recipe anchor at production scale for the accurate architecture.
- **Day 24:** test logits computed once; the external test as the gate; comparison with Barry's current
  VTL on the external test only (it saw every library snip, so the v2 test would flatter it).
- **Then:** refit the chosen models on all data (training, validation and test) with the fixed recipe
  and epoch count, which recovers the 20 to 30% of rare-class groups withheld for evaluation; the
  deployed model is that refit, documented as such.
- **Stretch, only if on schedule:** distillation of the accurate model into the deployable one.
- **Deliver:** `.keras` models and class maps, a `mewc-predict` and AddaxAI load check, throughput, a
  model card (data, recipe, per-class and open-set results, confidence-threshold advice); `mewc-train`
  2.1 upstream (resume, bf16, per-family defaults) by the 2.0.10 PR route.

## Schedule and compute

| Days | GPU 1 | GPU 2 |
|---|---|---|
| 1 to 4 | lever tests on VTL; fidelity configs | synthetic pipeline (Stage A, adapters, translations, probes) |
| 4 to 8 | architecture race and mini-searches | architecture race and mini-searches |
| 8 to 14 | stage 3 sweeps (accurate model) | stage 3 sweeps (deployable model), stage 4 |
| 14 to 17 | VTL production, interim recipe | stage 3 top-3 at medium; production pair |
| 17 to 24 | VTL production, final recipe | deployable model ×2; default anchor at production |
| 24 to 28 | test, refit, release | refit, distillation stretch |

| Stage | GPU-hours, no speed-up |
|---|---|
| 1 Levers and fidelity | 110 |
| 2 Shortlist | 90 |
| 3 Recipe tuning (incl. production pair) | 330 |
| 4 Data factors (incl. generation) | 120 |
| 5 Production, anchor, refit | 300 |
| Total | 950 of about 1,340 (2 × 28 days) |

At a 1.5 times speed-up the total falls to about 650. The margin is for reruns and the slower GPU.
Service units: about 4,800 for the reservation, against Barry's 200,000.

## Risks

- **Production runs are long and fragile**: checkpoint resume (stage 0) and the early interim seed.
- **Rare-class power**: pooled endpoint, eight seeds, stated minimum detectable effect; per-class
  results reported with intervals, not used for decisions.
- **Storage**: the 40,000-per-class pool fits both volumes; the 120,000 pool fits `ml-data` only.
- **Lease end**: instances are deleted; volumes persist. Park by `before_end_lease`, with the
  deep-server guard as a backstop.

## Decisions for Barry

1. **Dates.** Booking start and length; the plan assumes 28 days and two A100s.
2. **Deployment hardware** for the throughput axis. Recommendation: an 8-core CPU and a consumer GPU
   (RTX 2080 class, as in the MEWC paper), plus the A100.
3. **External test.** Labelled snips from recent services not in `data_600px.zip`, ideally a few
   thousand across cameras and including rare species. Recommendation: yes; it is the gate for the
   release and the only fair comparison with the current model.
4. **Class list.** The 96 classes for the refit; candidate new classes and the flame and scarlet robin
   split as a separate experiment. Recommendation: as stated.
5. **Synthetic review.** A blind review of synthetic snips for the rare classes in days 3 to 5 (about 30
   minutes). Recommendation: yes.
6. **Label check.** Review of confidently disagreed validation and test snips before sealing (likely a
   few hundred, an hour or two). Recommendation: yes, for the rare classes at least.

## Independent review (25 October) and what changed

A Claude Sonnet 5.5 subagent reviewed the first draft against the logs, the MEWC paper and the
`mewc-train` code. Its findings, all accepted:

- **Leakage and test sealing**: nearest-representative groups can split burst frames; the screen
  scored test on every run. Now: purge within cosine 0.85, cap per test group, project-disjoint
  stratum, validation-only selection, test sealed until stage 5.
- **Synthetic transfer**: views per synthetic snip grow about 25 times from screen to production
  scale; rare-class power is low; the generators saw v1 training snips that v2 may place in test.
  Now: synthetic count scales with budget, Stage A retrained on v2, eight paired seeds, pooled endpoint.
- **Recipe search**: single-seed successive halving ranks noise and is biased by ramped
  regularisation; samples and stage lengths are budget. Now: Sobol, fANOVA, paired seeds, fixed
  compute, pruning only at stage boundaries.
- **Fidelity**: the risk is hyperparameter rank transfer, not architecture rank. Now tested directly,
  with one production-scale pair.
- **Compute**: medium scale is 20% of production, not 10%; VTL production is 60 to 75 h. Budgets are
  now at 1×, with resume and an early production seed.
- **Baselines**: a default-recipe anchor at every scale; accurate and deployable criteria stated
  separately; the external test as gate.
- **Data**: `mewc-train` needs JPEG folders, so pools are extracted with a 40,000 cap; thinning saves no
  GPU time and is cut to one check.
- **Shortlist bias and benchmarking**: per-family mini-search before shortlisting; throughput in
  `mewc-predict`, not JAX on random tensors.
- **Missing**: open-set behaviour and risk-coverage, label review before sealing, refit on all data,
  early synthetic generation.
