"""Build the blind label-review page from the label-check thumbnails: group the flagged snips by species
pair, give each an id and a fixed random order for its two candidates (current label and ensemble
prediction), write the id key to items.csv and the page to label_review.html. The page holds no record of
which candidate is the label.
Usage: python build.py ITEMS_JSON CLASSES_TXT OUT_DIR"""
import json
import random
import sys
from pathlib import Path

import pandas as pd


def main(items_json, classes_txt, out_dir):
    out, here = Path(out_dir), Path(__file__).parent
    items = pd.DataFrame(json.load(open(items_json)))
    items["pair"] = [" | ".join(sorted(p)) for p in zip(items.label, items.pred)]
    size = items.groupby("pair").size().rename("n")
    items = items.join(size, on="pair").sort_values(["n", "pair", "p_pred"], ascending=[False, True, False])
    items["group"] = items.pair.map({p: i + 1 for i, p in enumerate(items.pair.unique())})
    items["id"] = [f"lc{i:03d}" for i in range(1, len(items) + 1)]
    flip = [random.Random(i).random() < 0.5 for i in items.id]
    items["a"] = [p if f else lab for f, lab, p in zip(flip, items.label, items.pred)]
    items["b"] = [lab if f else p for f, lab, p in zip(flip, items.label, items.pred)]
    items[["id", "group", "part", "path", "label", "pred", "p_pred", "agree", "a", "b"]].to_csv(out / "items.csv", index=False)
    page = (here / "template.html").read_text()
    page = page.replace("__CLASSES__", json.dumps(sorted(Path(classes_txt).read_text().split())))
    page = page.replace("__ITEMS__", json.dumps(items[["id", "group", "a", "b", "img"]].to_dict("records")))
    (out / "label_review.html").write_text(page)
    print(len(items), "items in", items.group.max(), "pairs;", round(len(page) / 1e6, 1), "MB")


if __name__ == "__main__":
    main(*sys.argv[1:4])
