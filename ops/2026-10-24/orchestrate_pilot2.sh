#!/usr/bin/env bash
# deep-server: when the CNN pilot and CPU benchmarks finish, merge ngx-gpu-5's scores onto ngx-gpu-4 and run the
# screening evaluations there; copies of the outputs land here. 2026-10-23.
set -uo pipefail
K="ssh -i $HOME/.ssh/id_ed25519_nectar -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30"
G4=ubuntu@100.119.231.51; G5=ubuntu@100.70.41.43; D=/data/mewc/screen_orch; R=/mnt/ml-data/mewc-synth/screen
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
done_on() { $K $1 "grep -q '$2' /mnt/scratch/logs/$3" 2>/dev/null; }
merge() {
  rsync -a -e "$K" $G5:$R/scores/ $D/scores/ && rsync -a --ignore-existing -e "$K" $D/scores/ $G4:$R/scores/
}
evaluate() {
  rsync -a -e "$K" $D/specs/ $G4:/mnt/scratch/specs/
  $K $G4 "cd ~/mewc-synth/src && nice ../.venv/bin/python screen_eval.py $R \"\$(cat /mnt/scratch/specs/$1.json)\" > $R/eval_$1.txt 2>&1 && cp $R/eval_report.json $R/eval_$1.json" \
    && stamp "eval $1 done" || stamp "eval $1 FAILED"
  rsync -a -e "$K" "$G4:$R/eval_$1.*" $D/out/ 2>/dev/null
}
mkdir -p $D/scores $D/out $D/bench
until done_on $G4 "CPUBENCH g4 DONE" cpubench_g4.log && done_on $G5 "CPUBENCH g5 DONE" cpubench_g5.log; do sleep 300; done
stamp "pilot2: merge"; merge
rsync -a -e "$K" $G5:$R/bench/ $D/bench/ && rsync -a -e "$K" $G4:$R/bench/ $D/bench/ && rsync -a -e "$K" $D/bench/ $G4:$R/bench/
$K $G4 "cd ~/mewc-synth/src && nice ../.venv/bin/python pilot_summary.py $R > $R/pilot_summary.txt 2>&1" && stamp "pilot summary done" || stamp "pilot summary FAILED"
rsync -a -e "$K" "$G4:$R/pilot_summary.*" $D/out/
stamp "PILOT2 ORCHESTRATION DONE"
