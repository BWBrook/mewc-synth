"""Score Barry's blind review of the real low-count translations (ops/2026-11-19): species answers against the
key, for translations (did they read as the intended class or as the look-alike source?) and for real snips (a
check on the reviewer and on labels); and real-or-synthetic answers against the truth. Barry gave the species
answers first and the real-or-synthetic answers in a second pass the same day.
Usage: python ops/2026-11-19/rare_review_eval.py KEY_CSV ANSWERS_DIR OUT_CSV"""
import json
import sys
from pathlib import Path

import pandas as pd


def main(key, answers, out):
    k = pd.read_csv(key)
    a = pd.DataFrame([{"id": f.stem, **json.loads(f.read_text())} for f in Path(answers).glob("rr*.json")])
    d = k.merge(a[["id", "real", "species", "note"]].rename(columns={"species": "answer", "real": "said"}), on="id", how="left")
    d["called"] = d.apply(lambda r: "target" if r.answer == r.target else "lookalike" if r.answer == r.lookalike
                          else (r.answer or "none"), axis=1)
    d["correct"] = d.answer == d.species
    d.drop(columns=["path", "source"]).to_csv(out, index=False)
    print("answers", d.answer.notna().sum(), "of", len(d))
    for kind, g in d.groupby("kind"):
        print(kind, "species correct", int(g.correct.sum()), "of", len(g), g.called.value_counts().to_dict(),
              "| said", g.said.value_counts().to_dict())
    print("real-or-synthetic, decided answers only:", pd.crosstab(d.kind, d.said).to_dict("index"))
    syn = d[d.kind == "synthetic"].assign(c=lambda x: x.called + "(" + x.said.str[0] + ")").groupby("target").c.agg(lambda s: " ".join(s))
    print("\ntranslations by class (2 each):"); print(syn.to_string())
    print("\nreal snips answered wrongly:"); print(d[(d.kind == "real") & ~d.correct][["id", "species", "answer", "note"]].to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
