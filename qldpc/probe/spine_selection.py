"""Derive the genuinely-bivariate BB spine by ENUMERATION (not curation).

The degeneracy probe's trend lane needs BB points whose MLE-vs-coset-ML gap is
*discriminable* and *bit-reproducible*. That requires TWO distinct gates (the
first alone is insufficient -- it admits A==B codes that force exact coset-ML
ties on every nonzero syndrome, where both decoders guess and the gap is
undefined / not bit-reproducible):

  1. STRONG bivariate: A and B EACH contain >=1 x-monomial AND >=1 y-monomial
     (so neither block is a degenerate quasi-1D all-x / all-y circulant).
  2. A != B as monomial sets: no exact coset-ML ties forced by the A==B
     symmetry. (Added 2026-06-06 after the original n=18 A==B point was found to
     force a 2-fold tie on every nonzero syndrome -- see CODA_PROBE_RESULT.)

Plus the standard tractability/validity gates:
  3. CSS valid: (HX @ HZ.T) % 2 == 0.
  4. k == 4 (16 coset classes -> cheap exact contraction).
  5. contraction_width(HZ, logicals_X) <= 28 (exact coset-ML tractable).

SELECTION is DETERMINISTIC and DISCRIMINABILITY-MAXIMIZING: among all gated
candidates pick the one with the SMALLEST exact-coset-ML-tie fraction (lex order
of (A, B) as tiebreak). The ``A != B`` gate removes only the *universal* ties
forced by the A==B symmetry, but exact coset-ML ties are partly *inherent* to
small degenerate codes -- the lex-smallest A!=B (3,3) code still ties 68% of
nonzero syndromes, whereas the least-tied gated (3,3) code ties only ~12%. On a
tied syndrome both coset-ML and MLE are forced to guess, diluting the measurable
gap toward 1.0, so minimizing the tie fraction maximizes discriminable signal.
Tie fraction is measured on a fixed-seed code-capacity syndrome sample
(deterministic); the well-separated fractions make the choice robust. (We do NOT
select by "lowest width": the cotengra width search is randomized, so width is a
comfortably-margined GATE, not a selector.) Residual ties (~7-36% even for the
best small codes) are made bit-reproducible downstream by the deterministic
tolerance tie-break in ``tn_mld._argmax_deterministic``.

Deriving the spine here -- rather than hand-picking it -- removes any
"cherry-picked the cleanest-ratio code" concern: the choice is a pure function
of (l, m, weight) and the gates.

Run: ``python3 -m qldpc.probe.spine_selection``  (prints the derived spine).
"""
import itertools
import os
import sys

import numpy as np

_QLDPC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _QLDPC_DIR not in sys.path:
    sys.path.insert(0, _QLDPC_DIR)

from bb_code import BBCode
from qldpc.foundation.tn_width import contraction_width

# (l, m) for the three spine scales; n = 2*l*m = 12, 18, 24. m>=2 and l>=2 avoid
# degenerate 1-D; (3,*) is the only non-1-D factorization at these n.
SPINE_LM = [(3, 2), (3, 3), (3, 4)]


def _monomials(l, m):
    """Distinct monomial alphabet for an (l, m) BB code: x^0..x^{l-1}, y^0..y^{m-1}."""
    return [("x", i) for i in range(l)] + [("y", j) for j in range(m)]


def _strong_bivariate(terms):
    """A block is STRONG bivariate iff it has >=1 x-monomial AND >=1 y-monomial."""
    return any(v == "x" for v, _ in terms) and any(v == "y" for v, _ in terms)


def gated_candidates(l, m, weight=3, width_budget=28, width_time=20, width_repeats=60):
    """Yield (A, B, k, width) for every (A, B) passing all five gates, in lex order.

    A, B are weight-``weight`` subsets of the (l, m) monomial alphabet. Cheap gates
    (STRONG bivariate, A!=B, CSS valid, k==4) are checked before the expensive width
    gate so the width search runs only on survivors.
    """
    mons = _monomials(l, m)
    subs = [tuple(sorted(c)) for c in itertools.combinations(mons, weight)]
    strong = [s for s in subs if _strong_bivariate(s)]
    strong.sort()  # lex order -> deterministic iteration / selection
    for A in strong:
        for B in strong:
            if A == B:                      # gate 2: A != B
                continue
            try:
                bb = BBCode(l=l, m=m, A_terms=A, B_terms=B)
            except Exception:
                continue
            HX = np.asarray(bb.HX) % 2
            HZ = np.asarray(bb.HZ) % 2
            if HX.shape[0] == 0 or HZ.shape[0] == 0:
                continue
            if ((HX @ HZ.T) % 2).any():     # gate 3: CSS valid
                continue
            if int(bb.k) != 4:              # gate 4: k == 4
                continue
            LX = np.asarray(bb.logicals_X()) % 2
            w = contraction_width(HZ, LX, max_time=width_time, max_repeats=width_repeats)
            if w > width_budget:            # gate 5: tractable
                continue
            yield A, B, int(bb.k), float(w)


def tie_fraction(l, m, A_terms, B_terms, p=0.05, shots=2000, seed=0):
    """Fraction of UNIQUE nonzero code-capacity syndromes (fixed-seed sample) on which
    coset-ML has an EXACT tie (top-2 coset probabilities equal up to contraction
    roundoff). Deterministic given (p, shots, seed). Lower = more discriminable."""
    from qldpc.probe.degeneracy import bb_HZ_LX, build_codecap_circuit
    from qldpc.foundation.tn_mld import _coset_prob

    HZ, LX, n, _k = bb_HZ_LX(l, m, A_terms, B_terms)
    priors = np.full(n, float(p))
    circ = build_codecap_circuit(HZ, LX, p)
    dets, _obs = circ.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    dets = np.asarray(dets, dtype=bool)
    uniq = {tuple(d.tolist()) for d in dets if d.any()}
    if not uniq:
        return 0.0
    tied = 0
    for s in uniq:
        pv = np.asarray(_coset_prob(HZ, LX, priors, np.array(s, dtype=bool))).reshape(-1)
        top = np.sort(pv)[::-1]
        if top[0] - top[1] < 1e-9 * max(top[0], 1e-30):
            tied += 1
    return tied / len(uniq)


def derive_point(l, m, **kw):
    """Deterministically derive ONE spine point at (l, m): the gated (A, B) with the
    SMALLEST exact-tie fraction (lex order of (A, B) as tiebreak)."""
    best_key = None
    best = None
    for A, B, k, w in gated_candidates(l, m, **kw):
        tf = tie_fraction(l, m, A, B)
        key = (round(tf, 6), A, B)  # minimize tie fraction, then lex
        if best_key is None or key < best_key:
            best_key = key
            best = dict(l=l, m=m, n=2 * l * m, A_terms=A, B_terms=B, k=k,
                        width=w, tie_frac=tf)
    if best is None:
        raise RuntimeError(f"no gated spine point for (l={l}, m={m})")
    return best


def derive_spine(lm=SPINE_LM, **kw):
    """Derive all spine points (one per (l, m)), deterministically."""
    return [derive_point(l, m, **kw) for (l, m) in lm]


if __name__ == "__main__":
    print("Deriving genuinely-bivariate BB spine (STRONG bivariate + A!=B + CSS + k=4 + width<=28),")
    print("selection = MINIMUM exact-tie fraction among gated candidates (lex tiebreak):\n")
    for pt in derive_spine():
        print(f"  n={pt['n']:>2}  l={pt['l']} m={pt['m']}  "
              f"A={pt['A_terms']}  B={pt['B_terms']}  k={pt['k']}  width={pt['width']:.1f}  "
              f"tie_frac={pt['tie_frac']*100:.0f}%")
