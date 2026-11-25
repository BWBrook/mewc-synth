# mewc-synth

Can a diffusion model, first adapted to Tasmanian camera-trap snips, generate
synthetic training examples that improve the MEWC classifier on rare
classes? This repo holds the test plan (`PLAN.md`), the code and the dated
logs. Snips, generated images and model weights live on the compute volumes,
not here.

Status: phase 2 under way (third GPU booking, 17 November to 1 December 2026; `docs/plan-booking-3.md`). The two final classifiers are released on Hugging Face (CC BY-NC 4.0): [EVA-02-B, accurate](https://huggingface.co/bwbrook/tasmania-mewc-eva02-b) and [HGNetV2-B4, fast](https://huggingface.co/bwbrook/tasmania-mewc-hgnetv2-b4). Phase 1 is summarised in `docs/phase1-summary.md`.

## Licence

Code is released under the [MIT Licence](LICENSE). Text, figures and data created for this project are released under [CC BY-SA 4.0](LICENSE-CONTENT). Third-party material keeps its own terms where stated.
