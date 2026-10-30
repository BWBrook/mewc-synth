"""Label check: average the class probabilities of several runs over the listed parts of split v2 (val,
test, test_proj, train) and list the snips where the ensemble confidently disagrees with the label, for a
person to adjudicate. Only the disagreement list is written; no metrics are computed on the test parts.
Validation probabilities come from RUNS_DIR/scores, the others from RUNS_DIR/labelcheck/RUN_PART_probs.npy.
Usage: python src/label_check.py RUNS_DIR V2_DIR OUT_CSV MIN_PROB PARTS RUN [RUN ...]
PARTS is comma-separated, for example val,test,test_proj."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(runs_dir, v2, out, min_prob, parts, *runs):
    d, v2, min_prob = Path(runs_dir), Path(v2), float(min_prob)
    classes = sorted((v2 / "classes.txt").read_text().split())
    ci = {c: i for i, c in enumerate(classes)}
    flagged = []
    for part in parts.split(","):
        df = pd.read_csv(v2 / f"{part}.csv")
        probs = np.stack([np.load(d / "scores" / f"{r}_probs.npy" if part == "val"
                                  else d / "labelcheck" / f"{r}_{part}_probs.npy") for r in runs])
        if probs.shape[1] != len(df):
            raise ValueError(f"{part}: {probs.shape[1]} scored rows against {len(df)} listed")
        mean, y = probs.mean(0), df["label"].map(ci).to_numpy()
        pred = mean.argmax(1)
        keep = (pred != y) & (mean.max(1) >= min_prob)
        flagged.append(pd.DataFrame({
            "part": part, "path": df["path"].to_numpy()[keep], "label": df["label"].to_numpy()[keep],
            "pred": np.array(classes)[pred[keep]], "p_pred": mean.max(1)[keep].round(4),
            "p_label": mean[np.flatnonzero(keep), y[keep]].round(4),
            "agree": (probs.argmax(2)[:, keep] == pred[keep]).sum(0)}))
        print(f"{part}: {keep.sum()} of {len(df)} flagged")
    res = pd.concat(flagged, ignore_index=True).sort_values("p_pred", ascending=False)
    res.to_csv(out, index=False)
    pairs = res.groupby(["label", "pred"]).size().sort_values(ascending=False)
    print(f"{len(res)} flagged with {len(runs)} runs at p >= {min_prob}; most frequent label -> prediction:")
    print(pairs.head(25).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:6], *sys.argv[6:])
