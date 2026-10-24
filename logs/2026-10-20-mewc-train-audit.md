# 2026-10-20 mewc-train audit, fixes and speed

Barry asked for the `lowest_loss` bug to be fixed and for other pipeline
problems and speed-ups to be found, tested and eventually packaged as a PR
to `zaandahl/mewc-train` plus a Docker Hub rebuild. The fixes are on the local
branch `fix/progressive-training-and-speed` in `~/git/mewc-train` (commit
`02d2975`). Released on 24 October 2026 at Barry's go-ahead ("Go ahead and merge
that PR and rebuild the Docker Hub image, definitely."): PR
[zaandahl/mewc-train#6](https://github.com/zaandahl/mewc-train/pull/6) merged
as `35ea469`, tagged `v2.0.10`, and published by the repository's CI as
`zaandahl/mewc-train:2.0.10` (also `:2.0` and `:latest`, digest
`sha256:25ef68931cf2d...`). The published image's `lib_model.py`,
`lib_data.py` and `mewc_train.py` are byte-identical to the merged branch, and
the 31 tests pass inside it.

## Findings

| # | Finding | Evidence | Effect | Fix |
|---|---|---|---|---|
| 1 | Best-model tracking starts at `lowest_loss = 1` | ViT-B and ViT-L benchmarks on noise data: val loss stayed above 1, so no model was saved and `calc_class_metrics` raised `File not found` | Harmless while focal loss falls below 1; fatal otherwise | Start at `float('inf')` |
| 2 | **Progressive dropout has never been applied** | In-container test (Keras 3.3.3, JAX): (a) `dropout_layer=[-2,-4],` is the tuple `([-2,-4],)`, so `len > 1` is false; (b) index -4 is `compression_bottleneck`, and the Dropout layers are at -5 and -2; (c) `.rate` changed after the first fit kept loss at 10.9 (rate 0.1 behaviour), and only after recompiling did it reach 89.6 (theory about 100) | Every model, including production, trained with constant dropout `DROPOUTS[0]` = 0.1 instead of the 0.1 to 0.4 ramp. The RandAugment ramp does apply, because the dataset is rebuilt each epoch | `set_dropout_rate()`: set every top-level Dropout, then `make_train_function(force=True)` |
| 3 | Written class map uses shuffled first-appearance order | `classes = train_df['Label'].unique()` after shuffling. `LabelEncoder` orders the output axis by sorted string codes (`'0','1','10','11',...`) | Not consumed wrongly today: old `mewc-predict` re-sorts `class_list.yaml` codes as strings, and the September 2026 `mewc-predict` demands explicit `class_ids`. A trap for anyone using the file | Sort classes |
| 4 | Sample order depends on `PYTHONHASHSEED` | `for label in set(ds_labels)` before the seeded shuffle | Runs with the same `SEED` are not reproducible across processes | Iterate `sorted(set(...))` |
| 5 | Labels are encoded per dataset | `LabelEncoder().fit_transform` on each data frame | A validation or test set missing a class would silently misalign one-hot columns | Fail fast unless class sets match |
| 6 | JPEG decode and resize run single-threaded | First `.map` in `create_tensorset` has no `num_parallel_calls` | Throughput limit, measured below | `num_parallel_calls=tf.data.AUTOTUNE`; order preserved (deterministic by default) |
| 7 | Pre-existing test error | `test_find_unfreeze_points_vit` passed the alias `'vitt'`; the code tests `mname[:2] == 'vt'`. Real runs use `VTx` codes and kimm names `blocks_N_attn`, so production unfreezing is unaffected | Test only | Pass `'vtt'` |

Minor and not changed: training images are cast to uint8 (truncated) before
RandAugment, while validation and test images stay float (at most one grey
level). In Keras 3.3.3, a `Dropout` constructed at rate 0 has no seed
generator, so raising its rate later crashes. mewc starts at 0.1, so it is
unaffected.

Behaviour change to note before any production retrain: fix 2 makes the
documented dropout ramp real. Its effect on accuracy has to be measured, not
assumed.

## Tests

Baseline 2.0.9: 29 tests, 1 error (#7). Branch: 31 tests, all OK. This
includes `test_set_dropout_rate_takes_effect_after_first_fit` and
`test_fit_progressive_saves_when_val_loss_above_one`. Run in the
`zaandahl/mewc-train:2.0.9` container with the branch sources bind-mounted.

## Speed (A100 40 GiB vGPU, dummy 600 px JPEGs, 96 classes x 100 samples, batch 16, BUF 9)

| Model | Phase | 2.0.9 | Branch |
|---|---|---|---|
| ViT-B/16 384 | frozen | 65 ms/step (about 246 images/s) | see below |
| ViT-B/16 384 | progressive | 117 to 120 ms/step (about 135 images/s) | see below |
| ViT-L/16 384 | frozen | 156 ms/step (about 103 images/s) | |
| ViT-L/16 384 | progressive | 227 ms/step (about 70 images/s) | |

At those rates a production-scale run (96 classes x 4,000 x 40 epochs) takes
about 30 h for ViT-B and about 2.5 days for ViT-L. `nvidia-smi` reports
utilisation correctly on this vGPU (99% under a PyTorch matmul).
