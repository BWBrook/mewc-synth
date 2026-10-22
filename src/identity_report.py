"""Summarise classifier verdicts on synthetic snips: per class and generation setting, the share
whose top-1 class is the intended class, the share taken by named look-alikes, and the mean
probability of the intended class. Also checks the classifier's output order on real snips.
Usage: python src/identity_report.py IDENTITY_DIR CLASS_LIST_YAML SANITY_PREFIX SANITY_TRUTH_CSV"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ALIASES = {"camera_bait": "bait", "flame_robin": "flame_scarlet_robin", "maned_duck": "maned_goose"}
LOOKALIKES = {"spotted_tail_quoll": ["eastern_quoll"],
              "black_currawong": ["grey_currawong", "forest_raven"],
              "forester_kangaroo": ["bennetts_wallaby", "tasmanian_pademelon"],
              "black_rat": ["house_mouse", "swamp_rat", "long_tailed_mouse"],
              "eastern_quoll": ["spotted_tail_quoll"],
              "grey_currawong": ["black_currawong", "forest_raven"],
              "tasmanian_devil": [],
              "scrubtit": ["tasmanian_scrubwren"],
              "tasmanian_scrubwren": ["scrubtit"],
              "strong_billed_honeyeater": ["yellow_throated_honeyeater", "crescent_honeyeater"],
              "yellow_throated_honeyeater": ["strong_billed_honeyeater"]}


def main(identity_dir, class_yaml, sanity_prefix, sanity_truth):
    d = Path(identity_dir)
    cmap = {str(k): str(v).strip() for k, v in yaml.safe_load(open(class_yaml)).items()}
    codes = sorted(cmap)
    col = {cmap[c]: i for i, c in enumerate(codes)}

    san = pd.read_csv(sanity_prefix + ".csv")
    truth = pd.read_csv(sanity_truth)
    san["truth"] = truth["class_code"].map(lambda c: ALIASES.get(c, c)).values
    sanity = {"snips": len(san), "top1_accuracy": round(float((san["top_name"] == san["truth"]).mean()), 4),
              "classes_below_50pct": san.assign(ok=san["top_name"] == san["truth"]).groupby("truth")["ok"].mean()
              .loc[lambda s: s < 0.5].round(2).to_dict()}

    man = pd.read_parquet(d / "manifest.parquet")
    vtl = pd.read_csv(d / "vtl.csv")
    probs = np.load(d / "vtl_probs.npy")
    vtl["relpath"] = vtl["path"].str.split("/").str[-2:].str.join("/")
    df = man.merge(vtl.reset_index(), on="relpath")
    df["model_name"] = df["class_code"].map(lambda c: ALIASES.get(c, c))
    df["p_intended"] = [probs[i, col[c]] for i, c in zip(df["index"], df["model_name"])]
    df["intended"] = df["top_name"] == df["model_name"]
    df["lookalike"] = [t in LOOKALIKES.get(c, []) for t, c in zip(df["top_name"], df["class_code"])]
    df["infrared"] = df["prompt"].str.contains("infrared")
    rows = []
    for (c, tag), g in df.groupby(["class_code", "tag"]):
        others = g.loc[~g["intended"], "top_name"].value_counts().head(3)
        rows.append({"class": c, "setting": tag, "n": len(g), "intended": round(float(g["intended"].mean()), 3),
                     "lookalike": round(float(g["lookalike"].mean()), 3),
                     "mean_p_intended": round(float(g["p_intended"].mean()), 3),
                     "intended_ir": round(float(g.loc[g["infrared"], "intended"].mean()), 3) if g["infrared"].any() else None,
                     "intended_day": round(float(g.loc[~g["infrared"], "intended"].mean()), 3) if (~g["infrared"]).any() else None,
                     "top_wrong": "; ".join(f"{k} {v}" for k, v in others.items())})
    table = pd.DataFrame(rows)
    table.to_csv(d / "identity_report.csv", index=False)
    (d / "identity_report.json").write_text(json.dumps({"sanity": sanity, "identity": rows}, indent=1))
    print(json.dumps(sanity))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:5])
