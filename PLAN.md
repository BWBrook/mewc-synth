# Test plan: synthetic snips for low-count MEWC classes

Written 20 October 2026, before any generation or classifier result. Changes
after this date are recorded in the change log at the end with reasons. This
plan is the reference for what counts as success.

## Question

Can synthetic camera-trap snips from a diffusion model, first adapted to the
Tasmanian MEWC domain, improve classification of low-count classes compared
with the current practice of oversampling the few real images? A secondary
question, deferred, is whether the same approach makes it feasible to add new
classes that have only tens of real images.

Success means better classifier performance on **held-out real snips**.
Image realism, FID or visual appeal are diagnostics only.

## The pipeline synthetic snips must match

Established by reading `mewc-snip` (v1.0.4 to current), MegaDetector's
`visualization_utils` and `mewc-train` 2.0.9 / `mewc-flow` 2.0.4:

1. A real snip is the exact MegaDetector box crop, with no expansion. It is
   squared to 600 x 600 with PIL Lanczos (aspect ratio not preserved) and saved
   by PIL with default JPEG settings (quality 75, 4:2:0). Most boxes are
   smaller than 600 px, so most snips are upsampled.
2. `mewc-train` decodes each JPEG and resizes it to the model size (384 for
   ViT) with `tf.image.resize`: bilinear, no antialiasing.
3. Each epoch samples exactly `CLASS_SAMPLES` images per class, with
   replacement for small classes, re-drawn per epoch. Training uses categorical
   focal loss and RandAugment with rising magnitude.

Synthetic snips are generated at 512 x 512, Lanczos-resized to 600 x 600 and
saved with PIL defaults into the class folders, so steps 2 and 3 apply to real
and synthetic snips identically.

## Data and splits

- Input: labelled 600 px snips, one folder per class, 96 classes, counts from
  about 50 to about 200,000. No camera, site or date metadata survives.
- Near-duplicates (burst frames of one event) are grouped by DINOv2
  embedding similarity plus perceptual hash. Every group goes wholly to one
  split. This stands in for site grouping and is a known, recorded
  limitation: snips from one site on different days can still straddle
  splits.
- Fixed, seeded splits per class: test, validation (separate from test), and
  train. File lists and SHA-256 hashes are committed; images are not.
- **Pseudo-rare benchmark:** about three abundant classes with different body
  plans (for example a macropod, a bird and a small mammal; chosen after the
  data audit) are cut to n = 50 and n = 20 training images. Their thousands of
  remaining held-out real snips give statistical power that true tail classes
  cannot. For these classes, neither the generator nor the classifier backbone
  may see any image beyond the cut set.
- **True tail classes:** every class with fewer than a threshold of real
  images (set after the audit, roughly under 500) uses its own grouped test
  split.

## Generator

- Base model: SDXL 1.0 base (`stabilityai/stable-diffusion-xl-base-1.0`,
  OpenRAIL++), 512 x 512 training and generation.
- Stage A, domain adaptation: one LoRA (rank 64 to 128) trained on a
  class-balanced sample of training snips across all 96 classes. Each class is
  capped (for example 1,000 to 2,000 snips) and small classes contribute all
  their training snips. Captions: `camera trap snip of a {common name}
  ({binomial}), {daylight colour | infrared night}`, with day and night set by
  a chroma test.
- Stage B, used only where Stage A is inadequate for a class: a per-class LoRA
  on top of Stage A for the rarest and new classes. Alternatively, translate
  real snips of an abundant look-alike class into the target class (Diff-Mix
  style).
- Forgetting check: generate from held-out prompts (species the domain stage
  never saw) with and without the domain LoRA and compare anatomy plausibility
  by blind review.

## Filters applied before any synthetic snip is used

1. Memorisation: reject snips whose nearest training image (DINOv2 cosine) is
   above a threshold calibrated on real near-duplicate pairs.
2. Identity: reject snips whose embedding is closer to another class's
   centroid than to the target class (BioCLIP 2 or DINOv2 features).
3. Human audit: Barry blind-labels a random sample of real and synthetic snips
   for (a) real or synthetic and (b) species. This gives an error rate per
   class.
4. Shortcut probe: a small classifier trained to separate real from synthetic
   snips on high-pass-filtered images only, after the identical
   600 px / JPEG / 384 px path. Accuracy well above chance means a processing
   fingerprint, to be fixed before any classifier comparison.

## Classifier comparisons

- Recipe: `mewc-train` 2.0.9 unchanged except paths, sample counts and epochs.
  Separate validation and test folders.
- Screening model: ViT-B/16 at 384 (`VTB`) from ImageNet weights, with a
  reduced `CLASS_SAMPLES` and epoch count, run identically for every option
  with three seeds. Confirmation: production-like `VTL` for the best option
  against baseline.
- Options, each filling the same per-class sample slots:
  - **A0** real only (current practice: oversampling plus RandAugment).
  - **A1** real plus copy-paste snips (segmented real animals on background
    crops), as the non-generative competitor.
  - **A2** real plus synthetic, at synthetic shares of 50% and 80% of the
    slots.
- Primary metric: per-class F1 on held-out real test snips for pseudo-rare
  and tail classes, with recall and precision reported separately. Secondary:
  confusion with look-alike classes, macro-F1 over all classes, and accuracy
  on the abundant classes (no-harm check).
- Uncertainty: bootstrap 95% intervals over test snips, plus seed variation.
- Decision rule: synthetic snips help if, for the pseudo-rare classes, A2
  beats A0 on F1 with the 95% interval of the difference excluding zero, and
  no abundant class loses more than 1 percentage point of F1. A2 must also be
  at least as good as A1 to justify generation over copy-paste.

## Governance

- Synthetic snips are always marked: a `synth_` filename prefix, a PNG/JPEG
  comment, and a manifest row recording generator version, LoRA hash, seed,
  prompt and filter results. They never enter species records, sighting
  databases or public datasets unmarked.
- Snips, generated images and model weights stay out of Git. Rare-species
  data carries location sensitivity even without metadata.
- Every result records the AI systems involved and their roles.

## Change log

- 2026-10-20: initial plan.
- 2026-10-21, before any data were processed. These are the data facts and
  the resulting decisions.
  - **Source:** `deep-server:/data/data_600px.zip`, 196 GB stored (no
    compression), 4,069,864 JPEGs, in folders
    `data_600px/<tier>/<common_name>/`. The tiers are `base` (22 classes,
    3.16 M), `sufficient` (61, 894 k), `need_more` (23, 11.5 k), `too_few`
    (14, 988) and `not_in_model` (53 tiny classes, 663; excluded).
    `species_list.txt` in the zip gives code, binomial and common name for
    all 120 used classes.
  - **The Tasmanian 96-class model** (`class_list_96sp.yaml`) is 22 `base`,
    49 `sufficient`, 22 `need_more` and 3 `too_few` classes (chestnut teal,
    brown goshawk, European goldfinch). Three name aliases apply: `bait` =
    `camera_bait`, `flame_scarlet_robin` = `flame_robin`, `maned_goose` =
    `maned_duck`. The 12 other `sufficient` classes and `bush_rat` are
    Gippsland (Victorian) species from Barry's separate model. The 11 other
    `too_few` classes are candidate new classes.
  - **Working set:** every snip for classes up to 30,000, and a seeded
    random 30,000 above that (seed 20261021), about 1.07 M snips. Class
    identity is the folder name.
  - **Scope:** Stage A trains on the train split of all 120 classes,
    Victorian included, because it is a generator and more camera-trap
    variety helps. Classifier comparisons use only the 96 Tasmanian model
    classes. "Low-count" targets are the in-model `need_more` and in-model
    `too_few` classes (94 to 1,016 snips).
  - **Captions** carry binomials: `camera trap snip of a {common}
    ({binomial}), {daylight colour | infrared night}`.
  - **Proposed pseudo-rare classes** (Barry to confirm), chosen as abundant
    in-model classes with a close look-alike, so identity errors in synthetic
    snips will show:
    - spotted-tailed quoll (24 k; look-alike eastern quoll)
    - forester kangaroo (131 k; look-alikes Bennett's wallaby, pademelon)
    - black currawong (43 k; look-alikes forest raven, grey currawong)
    - black rat (22 k; look-alikes swamp rat, house mouse, long-tailed
      mouse)

    Each is cut to n = 50 and n = 20 training snips by whole groups. Their
    test sets may be enlarged beyond the 2,000 cap for power.
