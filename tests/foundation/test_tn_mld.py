import itertools

import numpy as np
from qldpc.foundation.tn_mld import (
    coset_ml_ler,
    _brute_coset_ml_single,
    _coset_prob,
)
from qldpc.bb_code import BBCode


def _brute_coset_probs_multi(H, L, priors, syndrome):
    """Reference brute-force coset-probability VECTOR for general n_log.

    Enumerate all 2^n_mech error vectors; for those consistent with the
    syndrome (H e == s mod 2) accumulate prod(priors^e (1-priors)^(1-e)) into
    bucket ``class = sum_o ((L e) % 2)[o] * 2^o`` (little-endian, MATCHING the
    contractor's _coset_prob convention). Returns the (unnormalized) length
    2^n_log vector.
    """
    H = np.asarray(H) % 2
    L = np.asarray(L) % 2
    priors = np.asarray(priors, dtype=float)
    syndrome = np.asarray(syndrome) % 2
    n = H.shape[1]
    n_log = L.shape[0]
    w = np.zeros(2 ** n_log)
    for b in itertools.product((0, 1), repeat=n):
        e = np.array(b)
        if not np.array_equal((H @ e) % 2, syndrome):
            continue
        pr = np.prod([priors[j] if e[j] else 1.0 - priors[j] for j in range(n)])
        bits = (L @ e) % 2
        cls = int(sum(int(bits[o]) << o for o in range(n_log)))
        w[cls] += pr
    return w


def _normalize(v):
    s = v.sum()
    return v / s if s > 0 else v


def test_coset_prob_matches_bruteforce_nlog2():
    # 6-mechanism, 2-logical toy. Compare FULL normalized probability vectors
    # (robust to degenerate argmax ties). On the pre-fix flatten the class
    # buckets 1 and 2 are swapped, so this fails pre-fix and passes post-fix.
    H = np.array([[1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1]], dtype=np.int64)
    L = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1]], dtype=np.int64)
    priors = np.full(6, 0.05)
    rng = np.random.default_rng(7)
    n_checks = H.shape[0]
    worst = 0.0
    for _ in range(50):
        synd = rng.integers(0, 2, size=n_checks)
        bf = _brute_coset_probs_multi(H, L, priors, synd)
        if bf.sum() == 0:  # syndrome unreachable; skip
            continue
        tn = _coset_prob(H, L, priors, synd)
        diff = np.max(np.abs(_normalize(tn) - _normalize(bf)))
        worst = max(worst, diff)
        assert diff < 1e-9, f"n_log=2 vector mismatch {diff} for synd {synd}"
    print(f"n_log=2 worst normalized-vector diff over 50 syndromes: {worst:.3e}")


def test_coset_prob_matches_bruteforce_bb_toy():
    # Genuine bivariate-bicycle toy: n=18, k=4 -> n_log=4 (16 classes).
    # 2^18 = 262144 brute enumerations per syndrome.
    bb = BBCode(
        l=3, m=3,
        A_terms=(("x", 0), ("x", 1), ("x", 2)),
        B_terms=(("x", 0), ("x", 1), ("y", 1)),
    )
    H = np.asarray(bb.HZ) % 2
    L = np.asarray(bb.logicals_X()) % 2
    assert bb.n == 18 and bb.k == 4 and L.shape[0] == 4, (bb.n, bb.k, L.shape)
    priors = np.full(18, 0.05)
    rng = np.random.default_rng(11)
    n_mech = H.shape[1]
    worst = 0.0
    n_done = 0
    while n_done < 15:
        # Build a reachable syndrome from a random error so brute force is nonzero.
        e = (rng.random(n_mech) < 0.15).astype(np.int64)
        synd = (H @ e) % 2
        bf = _brute_coset_probs_multi(H, L, priors, synd)
        if bf.sum() == 0:
            continue
        tn = _coset_prob(H, L, priors, synd)
        diff = np.max(np.abs(_normalize(tn) - _normalize(bf)))
        worst = max(worst, diff)
        assert diff < 1e-7, f"BB-toy vector mismatch {diff} for synd {synd}"
        n_done += 1
    print(f"BB-toy (n=18,k=4) worst normalized-vector diff over 15 syndromes: {worst:.3e}")


def test_matches_bruteforce_rep_code():
    # 3-bit repetition, p=0.1: TN coset-ML LER must EQUAL brute-force enumeration.
    H = np.array([[1,1,0],[0,1,1]], dtype=np.int64)
    L = np.array([[1,1,1]], dtype=np.int64)
    rng = np.random.default_rng(0)
    p = 0.1
    err = (rng.random((2000, 3)) < p).astype(np.int8)
    synd = (err @ H.T) % 2
    obs = (err @ L.T) % 2
    tn = coset_ml_ler(H, L, np.full(3, p), synd, obs)
    bf = np.mean([_brute_coset_ml_single(H, L, np.full(3, p), synd[i]) != obs[i,0] for i in range(len(err))])
    assert abs(tn - bf) < 1e-9, f"TN {tn} != brute {bf}"

def test_steane_like_small_css_runs():
    # a small 2-logical CSS-ish toy: just ensure multi-logical contraction runs + returns a valid LER
    H = np.array([[1,1,1,1,0,0],[0,0,1,1,1,1]], dtype=np.int64)
    L = np.array([[1,1,0,0,0,0],[0,0,0,0,1,1]], dtype=np.int64)
    rng = np.random.default_rng(1); p = 0.05
    err = (rng.random((500, 6)) < p).astype(np.int8)
    synd = (err @ H.T) % 2; obs = (err @ L.T) % 2
    ler = coset_ml_ler(H, L, np.full(6, p), synd, obs)
    assert 0.0 <= ler <= 1.0
