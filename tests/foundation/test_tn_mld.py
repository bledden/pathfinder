import numpy as np
from qldpc.foundation.tn_mld import coset_ml_ler, _brute_coset_ml_single

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
