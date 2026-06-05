"""Enumerate small bivariate-bicycle codes and width-check exact code-capacity coset-ML.
BBCode(l, m, A_terms, B_terms): A/B are 3 monomials each, term=(var,power), var in {'x','y'}.
A BB code on (l,m) has n = 2*l*m physical qubits."""
import itertools
import json
import os
import sys

import numpy as np

# bb_code.py lives in the qldpc/ package dir (one level up from foundation/).
# pytest's conftest puts that dir on sys.path; when run via `python3 -m`, it is not,
# so add it here to make both invocation paths work.
_QLDPC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _QLDPC_DIR not in sys.path:
    sys.path.insert(0, _QLDPC_DIR)

from bb_code import BBCode
from qldpc.foundation.tn_width import contraction_width

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enumerate_bb.json")


def _monomial_sets(l, m, k=3):
    mons = [("x", a) for a in range(l)] + [("y", b) for b in range(m)]
    return list(itertools.combinations(mons, k))


def candidate_bb_params(max_lm=5, max_n=80, per_lm=6):
    cands = []
    for l in range(2, max_lm + 1):
        for m in range(2, max_lm + 1):
            n = 2 * l * m
            if n > max_n:
                continue
            sets = _monomial_sets(l, m)
            for A in sets[:per_lm]:
                for B in sets[:per_lm]:
                    cands.append({"l": l, "m": m, "A_terms": A, "B_terms": B, "n": n})
    return cands


def _css_valid(bb):
    HX = np.asarray(bb.HX) % 2
    HZ = np.asarray(bb.HZ) % 2
    return bool(np.all((HX @ HZ.T) % 2 == 0))


def evaluate_candidate(c, width_time=60, width_repeats=200):
    try:
        bb = BBCode(l=c["l"], m=c["m"], A_terms=c["A_terms"], B_terms=c["B_terms"])
    except Exception as e:
        return {**c, "css_valid": False, "k": 0, "width": float("inf"), "error": repr(e)}
    valid = _css_valid(bb)
    k = int(getattr(bb, "k", 0))
    w = float("inf")
    if valid and k >= 1:
        w = contraction_width(np.asarray(bb.HZ) % 2, np.asarray(bb.logicals_X()) % 2,
                              max_time=width_time, max_repeats=width_repeats)
    return {**c, "css_valid": valid, "k": k, "width": w}


def run(width_budget=32, width_time=15, width_repeats=40, per_lm=4, **kw):
    cands = candidate_bb_params(per_lm=per_lm, **kw)
    results = [evaluate_candidate(c, width_time=width_time, width_repeats=width_repeats) for c in cands]
    viable = sorted([r for r in results if r["css_valid"] and r["k"] >= 1 and r["width"] <= width_budget],
                    key=lambda r: (-r["n"], r["width"]))
    out = {"width_budget": width_budget, "n_evaluated": len(results),
           "n_viable": len(viable), "viable": viable, "all": results}
    json.dump(out, open(_OUT, "w"), indent=2)
    return out


if __name__ == "__main__":
    o = run()
    print(f"evaluated {o['n_evaluated']}, viable (width<=32) {o['n_viable']}")
    for r in o["viable"][:12]:
        print(f"  l={r['l']} m={r['m']} n={r['n']} k={r['k']} width={r['width']:.0f}  A={r['A_terms']} B={r['B_terms']}")
