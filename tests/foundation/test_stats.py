from qldpc.foundation.stats import wilson_ci, tost_equivalent, holm_bonferroni, failures_to_shots

def test_wilson_known():
    lo, hi = wilson_ci(50, 1000)
    assert 0.037 < lo < 0.040 and 0.065 < hi < 0.068

def test_tost_equivalent_true_when_close():
    # margin_rel=0.15: at n=10000 the SE (~0.0031) vs margin (~0.0076) lets the
    # two one-sided tests both reject at alpha=0.05. (0.10 is too tight for this n —
    # neither close nor far passes there; see test_stats concern note.)
    assert tost_equivalent(500, 10000, 510, 10000, margin_rel=0.15) is True

def test_tost_equivalent_false_when_far():
    assert tost_equivalent(500, 10000, 800, 10000, margin_rel=0.15) is False

def test_holm_controls_family():
    rej = holm_bonferroni([0.001, 0.02, 0.04, 0.5], alpha=0.05)
    assert rej == [True, False, False, False]

def test_failures_to_shots():
    assert failures_to_shots(target_failures=300, ler=0.003) == 100000
