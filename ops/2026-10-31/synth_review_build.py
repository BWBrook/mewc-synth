"""Assemble the synthetic-audit page: the head and stylesheet of the label-review template, then
synth_review_body.html with the sampled items.
Usage: python ops/2026-10-31/synth_review_build.py ITEMS_JSON OUT_HTML"""
import sys
from pathlib import Path

here = Path(__file__).parent
head = (here.parent / "2026-10-30/label_review/template.html").read_text().split("</style>")[0] + "</style>\n"
head = head.replace("__TITLE__", "Synthetic Snip Audit")
body = (here / "synth_review_body.html").read_text().replace("__ITEMS__", Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_text(head + body)
