# Plan: third GPU booking (17 November to 1 December 2026)

Two A100 vGPU reservations, 17 November 00:00 UTC to 1 December 23:59 UTC (10:59 AEDT on 2 December), about
700 GPU-hours. Barry: "They're all yours to do with as you think best!" (13 November). This booking is phase 2
of `docs/plan-next-booking.md` (production refit, test, release), rescoped by what phase 1 found
(`docs/phase1-summary.md`). Every decision rule below is registered in `PLAN.md` before its runs.

## What phase 1 changed

1. **Screen-scale choices do not transfer.** Tuned recipes lost their gain at medium scale and reversed at
   production scale. The shortlist was decided at screen scale too, among tie sets four models wide, so it
   is re-checked at medium scale before anything is refit.
2. **EVA-02-B overtrains on the production schedule.** Its best validation loss comes at epoch 12 of 40
   (two epochs into the progressive phase) and its macro-F1 is lower than at medium scale (0.9689 against
   0.9722). HGNetV2-B4 improves to the end at both scales. Each model needs its own schedule.
3. **Synthetic snips help more as training grows** (+0.023, +0.029, +0.041 for the pseudo-rare classes at
   screen, medium and production scale), because without them the rare classes overfit their few real
   snips. The real low-count classes are the case that matters for the released models and are untested.

## Questions for this booking

| # | Question | Design | GPU-h |
|---|---|---|---|
| Q1 | Do synthetic snips help the 25 real low-count classes? | HGNetV2-B4, production, A0 (full real) against S20 (20% of each low-count class's draws synthetic), 8 paired seeds | 80 + 37 generation |
| Q2 | Does the shortlist hold at medium scale? | the screen-scale tie sets at medium scale, 2 seeds each: DINOv3-L, EVA-02-L, ConvNeXt V2-L against EVA-02-B; EfficientNetV2-S, FastViT-SA24, MobileNetV4-HL against HGNetV2-B4 | 90 |
| Q3 | What schedule suits the accurate model? | EVA-02-B at 2,400 per class (6 frozen + 24 progressive), 2 seeds, plus a third medium seed; per-epoch validation macro-F1 is now logged | 25 |
| Q4 | How many real snips is a synthetic snip worth? | pseudo-rare classes, HGNetV2-B4, production: real-only arms with 100 and 200 real snips per class (nested supersets of the 50), 6 seeds each, read against A0 (50), S20 and full real | 60 |
| Q5 | Final models | train-only seeds of the chosen models and data (2 each), a refit on train + validation, test logits once, ONNX and fp16 checks, model card | 60 to 100 |
| Q6 | Stretch: does distillation from the accurate model improve the deployable one? | HGNetV2-B4 production with EVA-02-B soft targets, 2 seeds | 20 |

That is about 400 of the 700 GPU-hours; the slack covers retries, a fourth question if one of these opens
it, and seeds where an interval stays wide.

## Q1: real low-count classes

- **Classes:** the 25 in-model `need_more` and `too_few` classes, 65 to 680 training snips, 8 to 99
  validation snips (1,125 in all), 18 to 196 test snips. All are camera-trap snips; none is q95 JPEG.
- **Generator:** the split-v2 domain adapter `a_sv2` (`a_v2` was built on split v1, whose training set can
  hold v2 validation or test snips, so it cannot be used). Its cached latents were cleared on 8 November and
  are re-encoded first; the script checks that the re-encoded rows match the saved ones.
- **Class adapters:** Stage B as in phase 1 (rank 64, lr 5e-5, batch 16, 400 steps, or 200 below 100
  training snips), focus mass 0.45 on the class, 0.25 on its look-alike, 0.3 spread over 3,000 others.
- **Look-alike:** the in-model class with at least 2,000 camera-trap training snips that classifiers most
  confuse with the class: the highest mean probability on its validation snips, averaged over 127 saved runs
  (`ops/2026-11-13/rare_profile.csv`). This is a fixed rule, not hand-picked; for classes with almost no
  confusion (swamp harrier, skink, owlet-nightjar) it still names the nearest class.
- **Translations:** 1,600 per class, the S20 recipe (strength 0.65, guidance 3.5, 30 steps), sources from
  the look-alike's camera-trap training snips. That keeps each synthetic snip at about 20 views in a
  production run (4,000 × 40 × 20% / 1,600), as `plan-next-booking.md` asks.
- **Checks:** identity by the production VTL, memorisation against the class's training snips, the
  real-against-synthetic shortcut probe. A blind review by Barry is offered but not waited on.
- **Arm S20:** the training list plus, per low-count class, its real rows repeated k times and the 1,600
  translations, with k = round(4 × 1,600 / n_real), so synthetic snips are 20% of the class's draws.
- **Endpoint:** macro-F1 over the 25 classes on the validation snips, S20 − A0, two-level bootstrap
  (seeds, then snips). S20 helps if the interval excludes zero and no other class is harmed (an interval
  entirely below −0.01). With 8 to 99 validation snips per class, per-class results are reported, not
  tested.

## Q2: shortlist at medium scale

- Medium scale as in phase 1 (1,600 per class, 4 frozen + 16 progressive), default recipes as in the
  shortlist configs, seeds 1 and 2. EVA-02-B and HGNetV2-B4 already have two medium seeds.
- **Rule:** the same as the shortlist (PLAN 2026-11-02): on each track the leader is the highest two-seed
  mean, and any model whose two-level interval against the leader contains zero ties, with the tie going
  to the cheaper model on that track's hardware (2080 fp16 for accurate, 8-core CPU for deployable).

## Q3: accurate-model schedule

- **Lengths:** EVA-02-B at screen (have 3 seeds), medium (have 2, add a third), 2,400 per class with 6 + 24
  epochs (new, 2 seeds) and production (have 1).
- **Rule:** the accurate model trains at the length with the highest mean validation macro-F1.
- If Q2 replaces EVA-02-B, the new model runs at medium length unless one seed at 2,400 per class beats
  it.

## Q4: worth of a synthetic snip

Real-only arms with 100 and 200 real snips per pseudo-rare class, nested on the 50-snip subsets, HGNetV2-B4
at production, seeds 1 to 6 (pairing with phase 1's A0 and S20 production seeds). The focus macro-F1 curve
over 50, 100, 200 and the full pool gives S20's real-snip equivalent by interpolation, with a bootstrap
interval. This is descriptive and drives no decision.

## Q5: final models

- **Choices:** each track's model from Q2, the schedule from Q3 (production for the deployable model, whose
  validation loss is still falling at the end), and S20 for the low-count classes if Q1 says it helps.
- **Train-only seeds:** two per track on the final configuration (Q1's S20 or A0 runs are the deployable
  ones if HGNetV2-B4 stays).
- **Test, once:** each final train-only model scores the sealed test and held-out-project test snips once,
  on both the corrected labels and the original ones (`v2/*.pre_lc.csv`). Nothing is tuned afterwards.
- **Release candidates:** one refit per track on train + validation at the same schedule (`"select":
  "last"`), exported to ONNX (fused), with fp16 parity checked on alien's RTX 2080.
- **Not in this booking:** upstream release (MEWC, AddaxAI) and the external test, which needs labelled
  snips from recent services that Barry would supply.

## Placement

Only one CUDA process per GPU (vGPU). gpu-5 holds `a_sv2`, so it generates.

| Hours from start | ngx-gpu-4 (`ml-data`) | ngx-gpu-5 (`ml-data-2`) |
|---|---|---|
| 0 to 34 | Q1 A0 production, seeds 2 to 8 | `a_sv2` latents, 25 adapters, 40,000 translations, checks |
| 34 to 66 | Q3 EVA-02-B lengths; Q2 deployable track | Q1 S20 production, seeds 1 to 8 |
| 66 to 115 | Q2 accurate track, seed 1 | Q2 accurate track, seed 2 |
| 115 to 175 | Q4 real-equivalence | Q5 accurate-model runs |
| 175 onwards | Q5, Q6 | Q5, Q6 |

The first two rows need no decision and are queued at lease start. Everything after Q1, Q2 and Q3 report is
queued as their rules resolve. Both instances park on `c3.small` by 1 December 23:00 UTC.
