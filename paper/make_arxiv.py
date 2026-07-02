#!/usr/bin/env python3
"""Rebuild the arXiv bundle from pathfinder_draft.tex (run from paper/):
flattens figure paths, ASCII-fies the 6 unicode chars, copies figures, tars."""
import re, shutil, glob, os, subprocess
t = open("pathfinder_draft.tex", encoding="utf-8").read().replace("../figures/", "")
for k, v in {"§": r"\S{}", "·": r"\textperiodcentered{}", "†": r"\textdagger{}",
             "–": "--", "²": r"\textsuperscript{2}", "ö": r'\"o'}.items():
    t = t.replace(k, v)
bad = [c for c in set(t) if ord(c) > 127]
assert not bad, f"non-ASCII left: {bad}"
os.makedirs("arxiv", exist_ok=True)
open("arxiv/pathfinder.tex", "w").write(t)
for f in glob.glob("../figures/fig[0-9][0-9]_*.png"):
    shutil.copy(f, "arxiv/")
import datetime
tag = datetime.date.today().isoformat()
subprocess.run(["tar", "czf", f"pathfinder-arxiv-{tag}.tar.gz", "-C", "arxiv", "."], check=True)
print(f"wrote pathfinder-arxiv-{tag}.tar.gz")
