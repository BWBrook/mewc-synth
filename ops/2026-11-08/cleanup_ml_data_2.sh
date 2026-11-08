#!/usr/bin/env bash
# Clean-up of ml-data-2 after phase 1 (Barry, 8 November 2026: "lose no primary data or material that is costly
# to regenerate, but neither do we hang onto anything that is easily regenerated"). Removes only material that
# a config in Git, a download or a short computation rebuilds; every results file stays (registry, scores,
# probabilities, histories, configs, logs). Writes a manifest of every removed file first.
# Usage: [DRY=1] bash ops/2026-11-08/cleanup_ml_data_2.sh   (on ngx-gpu-5; DRY writes the manifest only)
set -euo pipefail
M=/mnt/ml-data/mewc-synth; S=/mnt/ml-data/scratch; OUT=$M/cleanup_2026-11-08
grep -q "^ml-data-2 " /mnt/ml-data/VOLUME || { echo "not ml-data-2"; exit 1; }
mkdir -p $OUT; man=$OUT/removed.tsv; : > $man
df -B1 /mnt/ml-data | tail -1 > $OUT/df_before.txt
add() { find "$@" -printf "%s\t%p\n" >> $man; }

# Resume and frozen-phase checkpoints of every finished run.
add $M/v2_runs/runs -type f \( -name last.pt -o -name frozen_best.pt \)
# Final weights of screen-scale runs (each retrains from its config in 15 min to 2 h; predictions are kept).
for p in race mini rt tn sl pt smoke; do add $M/v2_runs/runs -path "*/${p}_*/best.pt" -type f; done
# Keras weights of the October screen runs (screen scale, about 1 to 2 h each; scores and logs are kept).
add $M/screen/runs -type f -name "*.keras"
# Cached VAE latents for adapter training (re-encoded in minutes) and the stand-in smoke adapters.
add $M/stage_a -maxdepth 2 -type f -name latents.npy
add $M/stage_a/standin_a1/lora -type f
# Intermediate steps of the base adapters; step_010000, the one used downstream, stays.
add $M/stage_a -path "*/lora/r64_s10k/step_00[2468]000/*" -type f
# Download caches.
add $S/keras-cache $S/uv-cache $S/hf -type f
# The production VTL archive, if its single member equals vtl_mewc.keras byte for byte.
if [ "$(unzip -p $M/models/vtl_mewc.zip vtl_mewc.keras | sha256sum | cut -d' ' -f1)" = "$(sha256sum < $M/models/vtl_mewc.keras | cut -d' ' -f1)" ]; then
  add $M/models/vtl_mewc.zip
fi
awk -F'\t' '{s+=$1; n++} END {printf "manifest: %d files, %.1f GB\n", n, s/1e9}' $man
[ -n "${DRY:-}" ] && exit 0

cut -f2- $man | tr "\n" "\0" | sudo xargs -0 rm -f   # the October screen runs were written as root by Docker
for d in $M/stage_a/standin_a1/lora $M/stage_a/*/lora/r64_s10k/step_00[2468]000 $S/keras-cache $S/uv-cache $S/hf; do
  [ -d "$d" ] && find "$d" -depth -type d -empty -delete
done

# Docker images (public on Docker Hub, and also on ngx-gpu-4).
sudo docker image rm zaandahl/mewc-train:2.0.10 zaandahl/mewc-train:2.0.9 || true
sudo docker system prune -af || true

$HOME/mewc-synth/.venv/bin/python $HOME/mewc-synth/ops/2026-11-08/dedupe_snips.py $M | tee $OUT/dedupe.txt
df -B1 /mnt/ml-data | tail -1 > $OUT/df_after.txt
echo CLEANUP DONE
