"""Apply a label-corrections table to the v2 evaluation lists: relabel or drop each listed snip in val.csv,
test.csv and test_proj.csv, and rewrite val_paths.txt from the corrected validation list. The uncorrected
lists are kept as PART.pre_lc.csv and are always the starting point, so a rerun with a revised table is
safe. Snip files stay where they are: the lists carry the labels.
Usage: python src/apply_label_corrections.py V2_DIR CORRECTIONS_CSV
CORRECTIONS_CSV columns: part, path, old_label, new_label, action (relabel or exclude)."""
import shutil
import sys
from pathlib import Path

import pandas as pd


def main(v2, corrections):
    v2, c = Path(v2), pd.read_csv(corrections)
    classes = set((v2 / "classes.txt").read_text().split())
    bad = set(c.loc[c.action == "relabel", "new_label"]) - classes
    if bad:
        raise ValueError(f"new labels not in the class list: {sorted(bad)}")
    for part in ("val", "test", "test_proj"):
        orig = v2 / f"{part}.pre_lc.csv"
        if not orig.exists():
            shutil.copy(v2 / f"{part}.csv", orig)
        df, cp = pd.read_csv(orig), c[c.part == part]
        m = df.merge(cp[["path", "old_label"]], on="path", how="inner")
        if len(m) != len(cp) or (m.label != m.old_label).any():
            raise ValueError(f"{part}: {len(cp) - len(m)} listed paths missing or labels differ from the list")
        new = dict(zip(cp.loc[cp.action == "relabel", "path"], cp.loc[cp.action == "relabel", "new_label"]))
        df["label"] = [new.get(p, lab) for p, lab in zip(df.path, df.label)]
        df = df[~df.path.isin(cp.loc[cp.action == "exclude", "path"])]
        df.to_csv(v2 / f"{part}.csv", index=False)
        print(f"{part}: {len(new)} relabelled, {(cp.action == 'exclude').sum()} excluded, {len(df)} rows")
        if part == "val":
            (v2 / "val_paths.txt").write_text("\n".join(df.path) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
