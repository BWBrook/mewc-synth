"""Contact sheet for a translation run: for each target class, a row of source snips, then one row per
strength with the translated snips from the same sources, labelled with the VTL judge's top call.
Usage: python src/translate_sheet.py TRANSLATE_DIR [PER_ROW]"""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from common import DATA

T = 120


def main(out_dir, per_row=8):
    d, per_row = Path(out_dir), int(per_row)
    man = pd.read_parquet(d / "manifest.parquet")
    v = pd.read_csv(d / "vtl.csv")
    v["relpath"] = v["path"].str.split("/").str[-2:].str.join("/")
    man = man.merge(v[["relpath", "top_name", "top_prob"]], on="relpath")
    panels = []
    for tgt, sub in man.groupby("class_code"):
        srcs = sub["source_relpath"].drop_duplicates().head(per_row).tolist()
        panels.append((f"{tgt[:16]} src", [DATA / "work" / "snips" / p for p in srcs], [""] * len(srcs)))
        for s in sorted(sub["strength"].unique()):
            rows = [sub[(sub["strength"] == s) & (sub["source_relpath"] == p)].iloc[0] for p in srcs]
            ok = sum(r.top_name == tgt for r in sub[sub["strength"] == s].itertuples())
            panels.append((f"s={s} {ok}/{(sub['strength'] == s).sum()}", [d / r.relpath for r in rows],
                           [f"{r.top_name[:13]} {r.top_prob:.2f}" for r in rows]))
    img = Image.new("RGB", (per_row * T + 130, len(panels) * (T + 12)), "white")
    draw = ImageDraw.Draw(img)
    for i, (label, files, notes) in enumerate(panels):
        draw.text((4, i * (T + 12) + T // 2), label, fill="black")
        for k, (f, n) in enumerate(zip(files, notes)):
            img.paste(Image.open(f).convert("RGB").resize((T, T)), (130 + k * T, i * (T + 12) + 12))
            draw.text((132 + k * T, i * (T + 12)), n, fill="black")
    img.save(d / "translate_sheet.jpg", quality=88)
    print(d / "translate_sheet.jpg")


if __name__ == "__main__":
    main(*sys.argv[1:3])
