"""Split the training-pool label-check flags by Barry's rule (30 October 2026, chat: "Auto-correct those",
the option to auto-correct unanimous same-first-name misclicks in the training pool and send the other
confident flags, or a sample if there are many, for review). A flag whose label and prediction share a first
name and on which every run agrees becomes a relabel in the auto-corrections table; every other flag goes to
the review list.
Usage: python ops/2026-10-30/train_flags_split.py FLAGGED_CSV N_RUNS OUT_DIR"""
import sys
from pathlib import Path

import pandas as pd


def main(flagged, n_runs, out_dir):
    f, out = pd.read_csv(flagged), Path(out_dir)
    misclick = (f.label.str.split("_").str[0] == f.pred.str.split("_").str[0]) & (f.agree == int(n_runs))
    auto = f[misclick].rename(columns={"label": "old_label", "pred": "new_label"}).assign(
        action="relabel", reason="same-first-name misclick, unanimous ensemble")
    auto[["part", "path", "old_label", "new_label", "action", "reason", "p_pred"]].to_csv(out / "train_corrections_auto.csv", index=False)
    f[~misclick].to_csv(out / "train_review.csv", index=False)
    print(f"{len(f)} flagged: {len(auto)} auto-corrected, {(~misclick).sum()} for review")
    print(auto.groupby(["old_label", "new_label"]).size().sort_values(ascending=False).to_string())
    print(f[~misclick].groupby(["label", "pred"]).size().sort_values(ascending=False).head(20).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
