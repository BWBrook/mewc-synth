"""Replace byte-identical copies under v2/ with hard links to the same file under work/snips/ (split v2 copied
its snips out of the extracted pool). Nothing is removed: both paths stay, and each now points at one inode.
A v2 file is linked only if a work/snips file with the same name and size has identical bytes.
Usage: python ops/2026-11-08/dedupe_snips.py MEWC_SYNTH_DIR"""
import filecmp
import os
import sys
from collections import defaultdict
from pathlib import Path


def main(root):
    root = Path(root)
    index = defaultdict(list)
    for d, _, files in os.walk(root / "work" / "snips"):
        for f in files:
            index[f].append(os.path.join(d, f))
    linked = same = missing = differ = 0
    saved = 0
    for part in ("train", "val", "test", "test_proj"):
        for d, _, files in os.walk(root / "v2" / part):
            for f in files:
                p = os.path.join(d, f)
                st = os.stat(p)
                match = None
                for c in index.get(f, []):
                    cs = os.stat(c)
                    if cs.st_ino == st.st_ino:
                        match = "same"
                        break
                    if cs.st_size == st.st_size and filecmp.cmp(c, p, shallow=False):
                        match = c
                        break
                if match == "same":
                    same += 1
                elif match is None:
                    missing += 1 if not index.get(f) else 0
                    differ += 1 if index.get(f) else 0
                else:
                    tmp = p + ".dedupe_tmp"
                    os.link(match, tmp)
                    os.replace(tmp, p)
                    linked += 1
                    saved += st.st_size
        print(part, "linked", linked, "already", same, "no source", missing, "differ", differ,
              f"saved {saved / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
