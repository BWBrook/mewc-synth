"""Recompute registry rows on the corrected validation labels for every run on this instance that has
scores and no v2c row yet, carrying over the run's stack, backbone, seed, buffer and training time.
Usage: python ops/2026-10-30/rescore_v2c.py RUNS_DIR CLASS_TABLE_CSV"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import v2_eval

d, ct = Path(sys.argv[1]), sys.argv[2]
rows = [json.loads(line) for line in open(d / "registry.jsonl")]
done = {r["name"] for r in rows if r.get("labels") == "v2c"}
for r in rows:
    if r["name"] in done or r["name"].startswith("smoke") or not (d / "scores" / f"{r['name']}.csv").exists():
        continue
    v2_eval.main(d, r["name"], ct, json.dumps({k: r[k] for k in ("stack", "backbone", "seed", "buf", "train_s") if k in r}))
    done.add(r["name"])
