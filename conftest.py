import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, os.path.join(ROOT, "qldpc")):
    if p not in sys.path:
        sys.path.insert(0, p)
