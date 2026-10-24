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
  - **Barry's difficulty notes (2026-10-21, chat, paraphrased).** Spotted-tailed
    and eastern quolls are easy to tell apart. Forester kangaroo is easy
    except for partial views (foot, tail, close fur). Black and grey
    currawongs are easy if the beak or rump shows, otherwise nearly
    impossible. Black rat versus house mouse is very hard because snips lack
    scale, especially in IR. Swamp rat and long-tailed mouse are relatively
    easy. The four pseudo-rare classes are therefore kept as a **difficulty
    gradient**:
    - quoll: easy, a ceiling and no-harm control;
    - kangaroo: easy except partial views;
    - black currawong: partial-view-limited, where generation should help
      most;
    - black rat: scale-limited, a negative control where generation cannot
      add the missing cue.

    Results are interpreted per class.
  - `maned_duck` (Australian wood duck, *Chenonetta jubata*) is one species
    under several names. `flame_robin` is the merged flame and scarlet robin
    class (females are very hard to separate), so its caption is overridden
    in `configs/caption_overrides.csv`.
  - **Later candidate (Barry's suggestion):** splitting flame and scarlet
    robins with synthetic support. This needs an expert-labelled seed set and
    test set for each species, including females.
- 2026-10-22, before the classifier screen. Synthetic snips for the pseudo-rare
  classes come from look-alike translation (SDXL img2img at strength 0.65 from
  real camera-trap snips of the look-alike, with a per-class Stage B adapter),
  because generation from noise gave too little diversity. A reference arm
  **R** (pseudo-rare classes at full real data) was added to measure the
  headroom. Seeds: 3 for A0 and the synthetic arm, 2 for R.
- 2026-10-22, after round 1 results (`logs/2026-10-22-screen-r1.md`).
  - **Naming.** The round 1 synthetic arm was run under the name `A1`; it is
    this plan's **A2 at the 80% share** and is reported as **S80**. The 50%
    share is **S50** (real snips linked four times so that real and synthetic
    each fill half the draws, with the same 200 synthetic snips). The plan's
    A1 (copy-paste) keeps its name and has not been run.
  - **Deviation.** Quality check 4 (shortcut probe) ran after the comparison,
    not before. It found a VAE fingerprint (AUC 0.69 for a VAE round trip of
    real snips, 0.50 for resizing and JPEG alone); later synthetic sets are
    probed before any classifier run.
  - **No-harm rule, amended after seeing results.** As written, the rule
    flagged 11 of 92 non-focus classes for R, which cannot harm them, so it
    measures seed noise. Proposed replacement: a non-focus class is harmed if
    its F1 difference has a two-level (test snips and seeds) 95% interval
    entirely below -0.01, and the arm fails if more classes are harmed than
    under R. This change is post hoc and is reported as such.
- 2026-10-22, before any result for it: **bird arm (B)**, registered here.
  - Question: do camera-trap-style synthetic snips raise camera-trap
    performance for photo-dominated birds, on top of full real data?
  - Targets: the in-model classes with at least 80% eBird/iNaturalist
    training snips and camera-trap recall below 0.9 under R: Pacific black
    duck, flame robin, yellow-throated, strong-billed and crescent
    honeyeaters, pink robin, wedge-tailed eagle, grey shrike-thrush, maned
    duck (63 to 1,059 camera-trap train snips each).
  - A pilot restyling eBird photos as camera-trap snips kept identity but not
    the camera-trap viewpoint or setting, so it is not used. Synthetic snips
    are instead translations of each target's own camera-trap train snips
    (source = target, SDXL img2img with the Stage A domain adapter
    `a_v2/r64_s10k/step_010000`, strength 0.55, guidance 3.5, 30 steps,
    infrared captions from source chroma), 400 per target, sources drawn
    with replacement.
  - Arms, all on R's composition (every class at its real data, cap
    4,000): **R** (seeds 1 to 3); **B-syn**, R plus the 400 translations per
    target (seeds 1 to 3); **B-rep**, R plus 400 extra links cycling through
    the same real camera-trap train snips (seeds 1 to 3), the non-generative
    competitor using identical real information.
  - Primary metric: macro-F1 over the nine targets on camera-trap test snips
    (about 450), with two-level bootstrap intervals; per-class F1 and recall
    reported. B-syn helps if it beats both R and B-rep with intervals
    excluding zero; no-harm by the amended rule above. The shortcut probe
    (within mode) runs on the translations before the classifier runs.
- 2026-10-22, before its results: **S20** added to complete the share axis
  (same 200 translated snips per pseudo-rare class, real snips linked 16
  times, so 20% of draws are synthetic; seeds 1 to 3), queued behind the bird
  arm. Judged by the same rule as S80 and S50.
- 2026-10-23, after round 2 (`logs/2026-10-23-screen-r2.md`), before any of
  these results:
  - **Confirmation of S20**: seeds 4 and 5 for A0 and S20. S20 is confirmed
    if, over seeds 1 to 5, its focus macro-F1 gain over A0 keeps a two-level
    interval excluding zero with no class harmed under the amended rule.
  - **S10**: the same synthetic set at a 10% share (real snips linked 36
    times), seeds 1 to 3, to locate the optimum share.
  - **Production-model check** (this plan's "Confirmation: production-like
    VTL for the best option against baseline"): VTL A0 against VTL S20,
    seeds 1 and 2, same recipe as the screen. Read as confirming if the
    focus macro-F1 difference has the same sign as under VTB.
  - The bird arm (B-syn) failed its rule and is not pursued further in this
    lease.
- 2026-10-23, before any of these results, queued behind the confirmation
  runs:
  - **Second synthetic set (S20b)**: a fresh translation draw (seed 52, new
    look-alike sources sampled from the same pool, same adapters, strength
    0.65), 200 per pseudo-rare class at the 20% share, seeds 1 to 3. Tests
    whether the S20 gain depends on one particular batch of synthetic snips.
  - **Seeds 6 and 7** for A0 and S20 (VTB), so the confirmation rests on
    seven seeds per arm.
- 2026-10-24, after round 3 (seven VTB seeds, S10, S20b and two VTL seeds),
  before these results: the two-seed VTL check gave +0.004 [-0.036, +0.045]
  in focus macro-F1, which meets the registered "same sign" reading only
  nominally, with VTL's seed-to-seed spread larger than the effect. The VTL
  check is extended to **eight seeds per arm** (A0 and S20, seeds 3 to 8).
  S20's gain is taken to transfer to the production model if, over the eight
  seeds, the focus macro-F1 difference has a two-level interval excluding
  zero with no class harmed under the amended rule; otherwise the conclusion
  is that the VTB gain does not transfer to VTL at this sample size.
