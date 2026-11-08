# Phase 1 summary (lease ending 9 November 2026)

Phase 1 of `docs/plan-next-booking.md` ran on two Nectar A100 vGPU instances from 29 October to 8 November
2026. Every decision rule was registered in `PLAN.md` before its runs; this page collects the outcomes. All
scores are macro-F1 on the v2c validation snips (29,011). The test snips (98,060, plus 72,175 from held-out
projects) were never used for training or selection, and stay sealed for phase 2.

## Results

**Models** (PLAN 2026-11-03). Deployment hardware is an 8-core CPU and an RTX 2080 (Barry, decision 1).
- Accurate track: **EVA-02-B** at 392 px (2.2 images/s on 8 CPU cores, 87 on the 2080 in fp16).
- Deployable track: **HGNetV2-B4** at 288 px (54.6 images/s on 8 CPU cores, 868 on the 2080 in fp16).
- Both won under the registered tie rule with two-level intervals: EVA-02-B on four seeds (CAFormer-S36 led on
  three and fell back on its fourth), HGNetV2-B4 on three.

**Recipe tuning does not transfer from screen scale** (PLAN 2026-11-04 to 2026-11-08). Forty Sobol trials and
three-seed confirmation at screen scale (5% of production compute) found recipes beating the default by
+0.0037 (EVA-02-B) and +0.0102 (HGNetV2-B4), with intervals clear of zero. Each tuned recipe's difference from
the default shrank or reversed as the scale grew:

| Model | Screen (3 seeds) | Medium, 20% (2 seeds) | Production (1 seed) |
|---|---|---|---|
| EVA-02-B t21 − default | +0.0037 | +0.0003 | −0.0056 |
| HGNetV2-B4 t23 − default | +0.0102 | −0.0010 | not run (rule kept the default) |

Every winning recipe decays the learning rate further than the default, which suits a short budget. The
default recipes stay for phase 2. Any future recipe search has to run at medium scale or above.

**Synthetic snips help more as training grows** (PLAN 2026-11-05 to 2026-11-08). HGNetV2-B4, default
recipe; four pseudo-rare classes cut to 50 real snips each; S20 adds 200 look-alike translations per class
(20% of draws). Focus score = macro-F1 over those four classes. Intervals are two-level bootstraps (seeds,
then snips).

| Scale | Seed pairs | Focus, real only | Focus, S20 | S20 − real only | Overall, real only | Overall, S20 |
|---|---|---|---|---|---|---|
| screen | 8 | 0.841 | 0.864 | +0.023 [+0.014, +0.032] | 0.929 | 0.932 |
| medium | 8 | 0.814 | 0.843 | +0.029 [+0.003, +0.055] | 0.944 | 0.946 |
| production | 6 | 0.784 | 0.825 | +0.041 [+0.024, +0.056] | 0.942 | 0.947 |

- No non-focus class was harmed at any scale.
- The production gain exceeds the screen gain by +0.018 [−0.001, +0.034].
- With real snips only, the focus classes get worse as training grows: their 50 snips are drawn more often
  and the model overfits them. The synthetic snips offset this.
- The screen result replicates the ViT-B re-test of 1 November (+0.028) on a second architecture.

**Production-scale anchors** (one seed each): EVA-02-B 0.9689 and HGNetV2-B4 0.9670.

- HGNetV2-B4 improves at every step up in scale (0.943, 0.964, 0.967).
- EVA-02-B scores lower at production than at medium (0.972). One seed cannot separate that from noise.

**Other outcomes:**
- **Focal loss:** a NaN bug when γ < 1 was found and fixed (PLAN 2026-11-04).
- **Labels:** 135 validation and test labels were corrected after Barry's blind review (30 October). The
  original labels are kept in `v2/*.pre_lc.csv`.

## Where things are

- **Instances:** `ngx-gpu-4` and `ngx-gpu-5`, parked on `c3.small` at 22:03 AEDT on 8 November.
- **Environments:** kept on the boot volumes `ngx-boot-1` and `ngx-boot-2`.
- **Data volumes:** `ml-data` (on ngx-gpu-4) and `ml-data-2` (on ngx-gpu-5) hold the data, every run's
  `best.pt` and scores, and `v2_runs/registry.jsonl`. Each instance has its own registry.
- **Disk:** `ml-data-2` was cleaned on 8 November with Barry's approval, from 2.4 GB to 145 GB free
  (`ops/2026-11-08`; `cleanup_record/removed.tsv` lists every removed file). It removed 107 GB of material
  that a config, a download or a short computation rebuilds: resume checkpoints; the weights of screen-scale
  runs, whose predictions are all kept; cached VAE latents; intermediate adapter steps; download caches; and
  the mewc-train Docker images. A further 30 GB was saved by hard-linking v2's snips to their identical copies
  in `work/snips`, which removed nothing. Kept: all data and labels, every results file, the medium,
  production and production-synthetic weights, and every adapter used downstream. `ml-data` on ngx-gpu-4 has
  not been cleaned.

## For the next booking

1. **Phase 2 production refit** on the default recipes: EVA-02-B and HGNetV2-B4, at least two seeds each,
   with S20-style synthetic snips for the classes they help. Then the external test, reported on both the
   corrected and original labels.
2. **EVA-02-B at production:** check whether its production dip is real, with a second seed, or by
   keeping the final checkpoint against the best one.
3. **Real low-count classes:** build the synthetic pipeline (adapters, translations, Barry's blind audit),
   then the re-test.
4. **fp16 on the RTX 2080:** on alien, check that fp16 predictions match fp32 for the production
   HGNetV2-B4 before adopting fp16.
5. **Booking:** avoid the Nectar Tasmania maintenance window, 7 to 11 December 2026.
