"""Tests for the derived BB spine: discriminability gates + tie-break determinism."""
import numpy as np
from bb_code import BBCode

from qldpc.probe.spine_selection import _strong_bivariate, tie_fraction
from qldpc.probe.degeneracy import _BB_SPINE, bb_HZ_LX
from qldpc.foundation.tn_mld import _argmax_deterministic


def _a_neq_b(e):
    return tuple(sorted(e["A_terms"])) != tuple(sorted(e["B_terms"]))


def test_spine_passes_discriminability_gates():
    """All three spine codes: STRONG bivariate, A!=B, CSS valid, k==4, strictly increasing n."""
    assert len(_BB_SPINE) == 3
    ns = []
    for e in _BB_SPINE:
        HZ, LX, n, k = bb_HZ_LX(**e)
        ns.append(n)
        assert _strong_bivariate(e["A_terms"]), f"{e} A not strong bivariate"
        assert _strong_bivariate(e["B_terms"]), f"{e} B not strong bivariate"
        assert _a_neq_b(e), f"{e} has A==B (forces universal coset-ML ties)"
        bb = BBCode(**e)
        HX = np.asarray(bb.HX) % 2
        HZc = np.asarray(bb.HZ) % 2
        assert not ((HX @ HZc.T) % 2).any(), f"{e} not CSS-valid"
        assert k == 4, f"{e} k={k} != 4"
    assert ns == [12, 18, 24], f"spine n's not strictly increasing 12/18/24: {ns}"


def test_spine_is_discriminable_low_tie_fraction():
    """Min-tie selection: every spine code ties < 20% of sampled nonzero syndromes
    (the rejected A==B n=18 code tied 100%). Low ties = a measurable degeneracy gap."""
    for e in _BB_SPINE:
        tf = tie_fraction(e["l"], e["m"], e["A_terms"], e["B_terms"], shots=800)
        assert tf < 0.20, f"{e} tie_frac={tf:.2f} too high -> not discriminable"


def test_argmax_deterministic_lowest_index_on_ties():
    """_argmax_deterministic returns the LOWEST index among tolerance-tied maxima, so
    coset-ML is bit-reproducible despite contraction-roundoff flipping exact ties."""
    assert _argmax_deterministic(np.array([0.5, 0.5, 0.0])) == 0          # exact 2-fold tie
    assert _argmax_deterministic(np.array([0.5, 0.5 + 1e-15, 0.1])) == 0  # within rtol -> lowest
    assert _argmax_deterministic(np.array([0.1, 0.9, 0.2])) == 1          # clear unique max
    assert _argmax_deterministic(np.array([0.3, 0.3, 0.3])) == 0          # 3-fold tie -> lowest
    assert _argmax_deterministic(np.array([0.2, 0.8, 0.8])) == 1          # tie not at index 0
