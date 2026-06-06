import numpy as np

from qldpc.probe.degeneracy import verdict_from_gaps


# --------------------------------------------------------------------------- #
# LEVEL kill-switch (strict, unchanged -- fires regardless of CIs)
# --------------------------------------------------------------------------- #
def test_level_killswitch_fires():
    v = verdict_from_gaps([{"scale": "s", "ratio": 2.5}])
    assert v["verdict"] == "C" and "level" in v["reason"]


def test_level_killswitch_fires_with_cis():
    # A ratio above the level threshold fires even when its CI is present/wide.
    v = verdict_from_gaps([
        {"scale": "s", "ratio": 2.5, "ratio_lo": 2.1, "ratio_hi": 2.9, "bb": True},
    ])
    assert v["verdict"] == "C" and "level" in v["reason"]


# --------------------------------------------------------------------------- #
# TREND kill-switch -- NO-CI fallback (original strict point-estimate rule)
# --------------------------------------------------------------------------- #
def test_trend_killswitch_fires_no_ci():
    # No ratio_lo/ratio_hi -> strict point-estimate fallback fires on any
    # strictly-increasing BB sequence.
    v = verdict_from_gaps([{"scale": "a", "ratio": 1.1, "bb": True},
                           {"scale": "b", "ratio": 1.2, "bb": True},
                           {"scale": "c", "ratio": 1.35, "bb": True}])
    assert v["verdict"] == "C" and "trend" in v["reason"]


def test_b_mle_holds_no_ci():
    # Non-monotonic point estimates -> B-MLE (no-CI path).
    v = verdict_from_gaps([{"scale": "a", "ratio": 1.1, "bb": True},
                           {"scale": "b", "ratio": 1.05, "bb": True}])
    assert v["verdict"] == "B-MLE"


# --------------------------------------------------------------------------- #
# TREND kill-switch -- CI-AWARE rule
# --------------------------------------------------------------------------- #
def test_trend_ci_aware_noise_band_does_not_fire():
    # The demonstrated seed=123 noise case: strictly increasing point estimates
    # but OVERLAPPING CIs (largest-n ratio_lo NOT above smallest-n ratio_hi).
    # Must resolve to B-MLE -- the trend is within sampling noise.
    gaps = [
        {"scale": "BB n=12", "ratio": 0.985, "ratio_lo": 0.95, "ratio_hi": 1.02, "bb": True},
        {"scale": "BB n=18", "ratio": 0.995, "ratio_lo": 0.96, "ratio_hi": 1.03, "bb": True},
        {"scale": "BB n=24", "ratio": 1.001, "ratio_lo": 0.97, "ratio_hi": 1.04, "bb": True},
    ]
    v = verdict_from_gaps(gaps)
    assert v["verdict"] == "B-MLE", v


def test_trend_ci_aware_separated_fires():
    # Genuinely separated increasing trend: ascending point estimates AND
    # disjoint endpoint CIs (largest-n ratio_lo > smallest-n ratio_hi). Fires.
    gaps = [
        {"scale": "BB n=12", "ratio": 1.1, "ratio_lo": 1.05, "ratio_hi": 1.15, "bb": True},
        {"scale": "BB n=18", "ratio": 1.4, "ratio_lo": 1.35, "ratio_hi": 1.45, "bb": True},
        {"scale": "BB n=24", "ratio": 1.8, "ratio_lo": 1.75, "ratio_hi": 1.85, "bb": True},
    ]
    v = verdict_from_gaps(gaps)
    assert v["verdict"] == "C" and "trend" in v["reason"] and "CI-aware" in v["reason"], v


def test_level_fires_regardless_of_cis():
    # A ratio of 2.5 fires the LEVEL switch even with CIs present (CIs never
    # relax the level switch).
    gaps = [
        {"scale": "BB n=12", "ratio": 1.0, "ratio_lo": 0.95, "ratio_hi": 1.05, "bb": True},
        {"scale": "BB n=18", "ratio": 2.5, "ratio_lo": 2.3, "ratio_hi": 2.7, "bb": True},
    ]
    v = verdict_from_gaps(gaps)
    assert v["verdict"] == "C" and "level" in v["reason"], v


def test_trend_ci_aware_non_monotonic_does_not_fire():
    # Disjoint CIs but NOT strictly increasing in point estimate -> no trend.
    gaps = [
        {"scale": "BB n=12", "ratio": 1.5, "ratio_lo": 1.45, "ratio_hi": 1.55, "bb": True},
        {"scale": "BB n=18", "ratio": 1.2, "ratio_lo": 1.15, "ratio_hi": 1.25, "bb": True},
    ]
    v = verdict_from_gaps(gaps)
    assert v["verdict"] == "B-MLE", v


# --------------------------------------------------------------------------- #
# Memoized coset-ML == direct coset-ML (tiny case)
# --------------------------------------------------------------------------- #
def test_memoized_equals_direct():
    from qldpc.foundation.tn_mld import coset_ml_ler, coset_ml_ler_memoized

    # Tiny single-logical code: H = [[1,1,0],[0,1,1]], L = [[1,0,1]].
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
    L = np.array([[1, 0, 1]], dtype=np.int8)
    priors = np.full(3, 0.1)

    rng = np.random.default_rng(0)
    # Build many shots with deliberately REPEATED syndromes (exercise the cache).
    base_syn = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bool)
    idx = rng.integers(0, len(base_syn), size=200)
    syndromes = base_syn[idx]
    observables = rng.integers(0, 2, size=(200, 1)).astype(bool)

    direct = coset_ml_ler(H, L, priors, syndromes, observables)
    memo = coset_ml_ler_memoized(H, L, priors, syndromes, observables)
    assert direct == memo, (direct, memo)
