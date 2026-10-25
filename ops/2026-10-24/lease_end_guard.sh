#!/usr/bin/env bash
# deep-server safety net for the 25 October lease end (before_end_lease 22:59 UTC): at 21:30 UTC, on any
# ngx GPU instance still on its GPU flavour (nvidia-smi works), stop Docker, sync and unmount the scratch
# and data volumes so the lease-end hard delete cannot catch them mounted. Parked instances are skipped.
set -uo pipefail
K="ssh -i $HOME/.ssh/id_ed25519_nectar -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30"
stamp() { echo "[$(date -u +%H:%M:%S) UTC] $*"; }
target=$(date -u -d "2026-10-25 21:30" +%s)
while [ "$(date +%s)" -lt "$target" ]; do sleep 300; done
for h in ubuntu@100.119.231.51 ubuntu@100.70.41.43; do
  $K $h 'nvidia-smi -L >/dev/null 2>&1 || { echo "$(hostname): no GPU (parked), skipped"; exit 0; }
    if pgrep -f "run_screen|bench_infer|translate.py" >/dev/null; then echo "$(hostname): jobs still running, unmounting anyway"; fi
    sudo systemctl stop docker docker.socket containerd 2>/dev/null; sync
    for m in /mnt/scratch /mnt/ml-data; do mountpoint -q $m && sudo umount $m && echo "$(hostname): unmounted $m"; done; true' 2>&1 | sed "s/^/  /"
done
stamp "LEASE GUARD DONE"
