"""Training lists for the phase 1 synthetic re-test on split v2 (labels corrected, PLAN 2026-11-01). The four
pseudo-rare classes keep only their 50-snip subsets (Stage A run a_sv2); every other class keeps the full
corrected pool. Arms:
  A0    the 50 real snips per pseudo-rare class;
  S20   the 50 real snips listed 16 times plus the 200 translations of translate_v2 (20% of draws synthetic);
  S20f  as S20, but the 200 synthetic rows cycle through only the translations the production VTL assigns to
        the target class (identity filter, PLAN filter 2), so share and real exposure match S20;
  R     the pseudo-rare classes at their full corrected real pool (headroom reference).
Usage: python ops/2026-11-01/retest_build.py V2_DIR TRANSLATE_DIR STAGE_A_ROWS OUT_DIR"""
import sys
from pathlib import Path

import pandas as pd

PR = ["spotted_tail_quoll", "forester_kangaroo", "black_currawong", "black_rat"]


def main(v2, tdir, rows, out):
    v2, tdir, out = Path(v2), Path(tdir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(v2 / "train.csv")
    sub = pd.read_parquet(rows)
    sub = sub[sub.class_code.isin(PR)].assign(path=lambda x: str(v2) + "/" + x.relpath)[["path", "class_code"]]
    sub = sub.rename(columns={"class_code": "label"}).merge(train, on=["path", "label"])
    rest = train[~train.label.isin(PR)]
    man = pd.read_parquet(tdir / "manifest.parquet").assign(path=lambda x: str(tdir) + "/" + x.relpath)
    vtl = pd.read_csv(tdir / "vtl.csv").assign(path=lambda x: x.path.str.replace("/data/", "/mnt/ml-data/mewc-synth/", n=1))
    man = man.merge(vtl[["path", "top_name"]], on="path")
    synth = man[["path", "class_code"]].rename(columns={"class_code": "label"})
    passed = man[man.top_name == man.class_code]
    filt = pd.concat([g.iloc[[i % len(g) for i in range(200)]] for _, g in passed.groupby("class_code")])
    filt = filt[["path", "class_code"]].rename(columns={"class_code": "label"})
    arms = {"A0": [rest, sub], "S20": [rest, *[sub] * 16, synth], "S20f": [rest, *[sub] * 16, filt],
            "R": [train]}
    for arm, parts in arms.items():
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(out / f"{arm}.csv", index=False)
        pr = df[df.label.isin(PR)]
        print(arm, len(df), pr.groupby("label").agg(rows=("path", "size"), distinct=("path", "nunique"),
                                                    synth_share=("path", lambda p: p.str.contains("/synth_").mean())).round(3).to_dict("index"))
    print("50-snip subsets present:", sub.groupby("label").size().to_dict())
    print("translations passing the identity filter:", passed.groupby("class_code").size().to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:5])
