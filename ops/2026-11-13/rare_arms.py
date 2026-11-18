"""S20 training list for the real low-count classes (docs/plan-booking-3.md, Q1): the full training list plus,
per class in the profile, its real rows repeated k = round(4 × m / n_real) times and its m translations, so
synthetic snips are about 20% of that class's draws. Translations come from the manifest, one row per file
that exists (a rerun of a class writes the same names).
With --filtered (S20f, registered 18 November): the synthetic rows cycle through only the translations the
production VTL assigns to the class (TRANSLATE_DIR/vtl.csv), keeping the share and real exposure of S20; a
class with no passing translation gets no synthetic rows.
Usage: python ops/2026-11-13/rare_arms.py TRAIN_CSV TRANSLATE_DIR PROFILE_CSV OUT_CSV [--filtered]"""
import sys
from pathlib import Path

import pandas as pd


def main(train_csv, tdir, profile, out, filtered=False):
    train, tdir = pd.read_csv(train_csv), Path(tdir)
    classes = pd.read_csv(profile)["class"].tolist()
    man = pd.read_parquet(tdir / "manifest.parquet")
    man = man.assign(path=str(tdir) + "/" + man.relpath).drop_duplicates("path")
    man = man[man.path.map(lambda p: Path(p).exists())]
    if filtered:
        vtl = pd.read_csv(tdir / "vtl.csv").assign(path=lambda x: x.path.str.replace("/data/", "/mnt/ml-data/mewc-synth/", n=1))
        man = man.merge(vtl[["path", "top_name"]], on="path")
    parts = [train]
    for c in classes:
        real = train[train.label == c]
        syn = man[man.class_code == c][["path"]].assign(label=c)
        m = len(syn)
        if filtered:
            ok = man[(man.class_code == c) & (man.top_name == c)][["path"]].assign(label=c)
            syn = ok.iloc[[i % len(ok) for i in range(m)]] if len(ok) else ok
            print(f"{c}: {len(ok)} of {m} translations pass the identity filter")
        k = max(1, round(4 * m / len(real)))
        parts += [real] * (k - 1) + [syn]
        print(f"{c}: real {len(real)} x {k}, synthetic rows {len(syn)}, share {len(syn) / (k * len(real) + len(syn)):.3f}")
    pd.concat(parts, ignore_index=True)[["path", "label"]].to_csv(out, index=False)


if __name__ == "__main__":
    main(*sys.argv[1:5], filtered="--filtered" in sys.argv[5:])
