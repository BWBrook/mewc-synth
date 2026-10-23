# Queue scripts, 22 to 23 October 2026

Copies of the one-off queues run on ngx-gpu-4 (`*_g4.sh`) and ngx-gpu-5 (`*_g5*.sh`), launched with
`setsid nohup`, logs in each machine's `/mnt/scratch/logs/`; and the deep-server orchestrator
(`orchestrate.sh`) that merges ngx-gpu-5's scores onto ngx-gpu-4 and runs the evaluations, with the
arm definitions in `specs/`. `scale_cue.py` is the sharpness check recorded in `docs/roadmap.md`.
