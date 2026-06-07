"""TDD for the statistics / verdict layer (qldpc/zoo/analysis.py).

This is the layer that turns the matched-grid manifests into defensible verdicts
per the pre-registration (qldpc/zoo/prereg.json):

  * beat / tie: Wilson-CI separation (with the 300-failure-target guard) and
    TOST equivalence at the declared margin.
  * gap_to_mle_bootstrap (R2, BINDING): the LOCKED gap statistic = a per-shot
    PAIRED bootstrap of decoder-vs-Tesseract-MLE failure indicators on the SAME
    shots (NOT an aggregate ratio with a single Wilson CI). Seeded/deterministic.
  * multiplicity_grid: Holm-Bonferroni (primary) + BH-FDR (secondary) over the
    FULL grid of (decoder, p, basis) comparisons, INCLUDING losses.
  * gate_a: the BP-OSD-vs-Tesseract 2x2 disagreement diagnostic + oracle bound.
  * per_shot_fails: the per-shot fail-mask helper the bootstrap / Gate-A consume.

Plus the harness hook: run_matched(..., keep_per_shot=True) exposes the per-shot
fail mask per decoder (aligned to the shared shots) so T6 can feed
decoder-vs-Tesseract masks straight into gap_to_mle_bootstrap / gate_a without
re-decoding, while the default manifest stays unchanged.
"""
import numpy as np
import pytest

from bb_code import BBCode
from qldpc.foundation.circuits import build_memory
from qldpc.foundation.stats import tost_equivalent, wilson_ci
from qldpc.zoo.adapters import APPROVED_TIE_BREAKS
from qldpc.zoo.analysis import (
    beat,
    gap_to_mle_bootstrap,
    gate_a,
    multiplicity_grid,
    per_shot_fails,
    tie,
)
from qldpc.zoo.harness import dem_hash, run_matched


# --- beat -------------------------------------------------------------------
def test_beat_clearly_separated_with_enough_failures():
    """A clearly lower with both cells >= 300 failures and non-overlapping CIs."""
    # A: 300/30000 = 0.010 ; B: 600/30000 = 0.020 — well separated, both >= 300.
    assert beat(300, 30000, 600, 30000) is True


def test_beat_overlapping_cis_is_false():
    """Overlapping Wilson CIs -> NOT a beat even with plenty of failures."""
    # A: 500/30000, B: 510/30000 — essentially identical, CIs overlap heavily.
    assert beat(500, 30000, 510, 30000) is False


def test_beat_separated_but_below_failure_target_is_false():
    """Non-overlapping CIs but < 300 failures in a cell -> CANNOT claim a beat."""
    # A: 5/100000, B: 100/10000 — A's rate far lower & CIs separated, but A has
    # only 5 failures (< 300) so the failure-target guard blocks the beat.
    la, ha = wilson_ci(5, 100000)
    lb, hb = wilson_ci(100, 10000)
    assert ha < lb  # CIs genuinely non-overlapping (A lower)
    assert beat(5, 100000, 100, 10000) is False


def test_beat_requires_a_lower_not_b():
    """If B is the lower one, A does NOT beat B (directional)."""
    assert beat(600, 30000, 300, 30000) is False


def test_beat_custom_failure_target():
    """The failure-target is configurable; lowering it permits a beat."""
    # A: 50/50000, B: 200/50000 — separated; both >= 40 but < 300.
    assert beat(50, 50000, 200, 50000) is False
    assert beat(50, 50000, 200, 50000, failure_target=40) is True


# --- tie --------------------------------------------------------------------
def test_tie_equivalent_rates_true():
    """Two essentially-equal rates at a generous margin -> TOST equivalent."""
    # 1000/100000 vs 1005/100000 at a 10% relative margin.
    assert tie(1000, 100000, 1005, 100000, margin_rel=0.10) is True
    # cross-check against the underlying TOST directly
    assert tost_equivalent(1000, 100000, 1005, 100000, margin_rel=0.10) is True


def test_tie_clearly_different_false():
    """Clearly different rates -> NOT a tie (and NOT confused with CI-overlap)."""
    assert tie(300, 30000, 600, 30000, margin_rel=0.10) is False


def test_tie_ci_overlap_is_not_a_tie():
    """Overlapping CIs (beat==False) is NOT sufficient for a tie at a tight margin."""
    # 100/2000 vs 130/2000: CIs overlap (no beat) but rates differ by ~30% — at a
    # 5% relative margin TOST should NOT declare equivalence.
    assert beat(100, 2000, 130, 2000) is False
    assert tie(100, 2000, 130, 2000, margin_rel=0.05) is False


# --- gap_to_mle_bootstrap (R2, BINDING) ------------------------------------
def _paired_masks(n=1000, n_dec_fail=120, n_anchor_fail=100, overlap=80, seed=7):
    """Build paired per-shot fail masks with a controlled overlap.

    decoder fails on `n_dec_fail` shots, anchor on `n_anchor_fail` shots, with
    `overlap` shots where BOTH fail (partial correlation). Distinct shots.
    """
    rng = np.random.default_rng(seed)
    assert overlap <= min(n_dec_fail, n_anchor_fail)
    assert (n_dec_fail - overlap) + (n_anchor_fail - overlap) + overlap <= n
    idx = rng.permutation(n)
    dec = np.zeros(n, dtype=bool)
    anch = np.zeros(n, dtype=bool)
    both = idx[:overlap]
    dec_only = idx[overlap:overlap + (n_dec_fail - overlap)]
    anch_only = idx[overlap + (n_dec_fail - overlap):
                    overlap + (n_dec_fail - overlap) + (n_anchor_fail - overlap)]
    dec[both] = True
    dec[dec_only] = True
    anch[both] = True
    anch[anch_only] = True
    return dec, anch


def test_gap_bootstrap_point_ratio_and_ci_bracket():
    """Known paired masks (dec 120/1000, anchor 100/1000) -> point ratio ~1.2 and
    a bootstrap CI that brackets it."""
    dec, anch = _paired_masks(n=1000, n_dec_fail=120, n_anchor_fail=100, overlap=80)
    assert dec.sum() == 120 and anch.sum() == 100
    out = gap_to_mle_bootstrap(dec, anch, n_boot=2000, seed=0)
    assert set(out) == {"ratio", "lo", "hi"}
    assert abs(out["ratio"] - 1.2) < 1e-9
    assert out["lo"] < out["ratio"] < out["hi"]
    assert out["lo"] > 0.5 and out["hi"] < 2.5  # sane spread for these counts


def test_gap_bootstrap_deterministic_under_seed():
    """Same seed -> byte-identical result; a different seed -> different CI."""
    dec, anch = _paired_masks()
    a = gap_to_mle_bootstrap(dec, anch, n_boot=1000, seed=42)
    b = gap_to_mle_bootstrap(dec, anch, n_boot=1000, seed=42)
    assert a == b
    c = gap_to_mle_bootstrap(dec, anch, n_boot=1000, seed=43)
    assert (a["lo"], a["hi"]) != (c["lo"], c["hi"])


def test_gap_bootstrap_is_paired_not_aggregate():
    """The bootstrap must be PAIRED (same resampled shot indices for both arms).

    Perfectly-correlated masks (decoder fails iff anchor fails, ratio==1) give a
    degenerate CI exactly at 1.0 under a PAIRED resample; an unpaired/aggregate
    resample would NOT collapse to a point. We assert the paired collapse.
    """
    mask = np.zeros(1000, dtype=bool)
    mask[:100] = True
    out = gap_to_mle_bootstrap(mask, mask, n_boot=500, seed=1)
    assert out["ratio"] == 1.0
    assert out["lo"] == 1.0 and out["hi"] == 1.0


def test_gap_bootstrap_anchor_zero_handled():
    """Anchor never fails -> ratio is +inf (decoder fails somewhere); reported as a
    bound rather than crashing on divide-by-zero."""
    dec = np.zeros(1000, dtype=bool)
    dec[:50] = True
    anch = np.zeros(1000, dtype=bool)
    out = gap_to_mle_bootstrap(dec, anch, n_boot=200, seed=0)
    assert out["ratio"] == float("inf")
    assert out["hi"] == float("inf")


def test_gap_bootstrap_both_zero_handled():
    """Neither fails -> 0/0; reported as a finite sentinel (ratio == 1.0), no crash."""
    z = np.zeros(500, dtype=bool)
    out = gap_to_mle_bootstrap(z, z, n_boot=100, seed=0)
    assert out["ratio"] == 1.0
    assert out["lo"] == 1.0 and out["hi"] == 1.0


def test_gap_bootstrap_length_mismatch_raises():
    with pytest.raises((ValueError, AssertionError)):
        gap_to_mle_bootstrap(np.zeros(10, dtype=bool), np.zeros(11, dtype=bool))


# --- multiplicity_grid ------------------------------------------------------
def _comp(decoder, p, basis, fa, na, fb, nb):
    return {"decoder": decoder, "p": p, "basis": basis,
            "fails_a": fa, "n_a": na, "fails_b": fb, "n_b": nb}


def test_multiplicity_full_grid_holm_and_bh():
    """One strong comparison + several marginals: the strong survives Holm; at
    least one marginal does NOT; the FULL grid (incl. losses) is returned."""
    comps = [
        _comp("D0", 0.003, "Z", 100, 20000, 400, 20000),   # strong (very low p)
        _comp("D1", 0.003, "Z", 300, 20000, 330, 20000),   # marginal
        _comp("D2", 0.003, "X", 300, 20000, 320, 20000),   # marginal
        _comp("D3", 0.005, "Z", 300, 20000, 305, 20000),   # null
        _comp("D4", 0.005, "X", 300, 20000, 300, 20000),   # null (identical)
    ]
    ann = multiplicity_grid(comps)
    assert len(ann) == len(comps)  # FULL grid, losses included
    by = {a["decoder"]: a for a in ann}
    # every comparison carries a raw p and both adjusted-significance flags
    for a in ann:
        assert 0.0 <= a["p"] <= 1.0
        assert isinstance(a["holm_sig"], (bool, np.bool_))
        assert isinstance(a["bh_sig"], (bool, np.bool_))
        # identity passthrough
        for k in ("decoder", "p", "basis"):
            assert k in a
    # the strong comparison survives Holm; the identical-rate null does not
    assert by["D0"]["holm_sig"] is True or by["D0"]["holm_sig"] == True  # noqa: E712
    assert bool(by["D0"]["holm_sig"]) is True
    assert bool(by["D4"]["holm_sig"]) is False
    # BH-FDR is at least as permissive as Holm for the strong one
    assert bool(by["D0"]["bh_sig"]) is True
    # the strong p is far smaller than the null p
    assert by["D0"]["p"] < by["D4"]["p"]


def test_multiplicity_empty():
    assert multiplicity_grid([]) == []


# --- gate_a -----------------------------------------------------------------
def test_gate_a_counts_and_bounds():
    """Synthetic BP-OSD vs Tesseract masks -> correct 2x2 + P(B|A fails) + oracle."""
    n = 1000
    a = np.zeros(n, dtype=bool)  # BP-OSD fail mask
    b = np.zeros(n, dtype=bool)  # Tesseract fail mask
    # construct disjoint groups:
    #   both wrong:      shots [0:30)      (30)
    #   only A right:    shots [30:80)     (50)  -> A correct, B wrong
    #   only B right:    shots [80:200)    (120) -> A wrong,   B correct
    #   both correct:    shots [200:1000)  (800)
    a[0:30] = True   # A wrong (both wrong)
    b[0:30] = True   # B wrong (both wrong)
    b[30:80] = True  # only A right: B wrong, A correct
    a[80:200] = True  # only B right: A wrong, B correct
    out = gate_a(a, b)
    assert out["n"] == n
    assert out["both_correct"] == 800
    assert out["both_wrong"] == 30
    assert out["only_a_right"] == 50   # A correct, B wrong
    assert out["only_b_right"] == 120  # A wrong, B correct
    # A (BP-OSD) fails on both_wrong + only_b_right = 30 + 120 = 150
    # of those, Tesseract succeeds on only_b_right = 120
    assert abs(out["p_tesseract_succeeds_given_bposd_fails"] - 120 / 150) < 1e-12
    # oracle bound = 1 - both_wrong/n
    assert abs(out["oracle_ler_bound"] - 30 / n) < 1e-12


def test_gate_a_no_bposd_failures():
    """BP-OSD never fails -> P(B | A fails) is undefined; report a sentinel, no crash."""
    n = 100
    a = np.zeros(n, dtype=bool)
    b = np.zeros(n, dtype=bool)
    b[:10] = True
    out = gate_a(a, b)
    assert out["both_wrong"] == 0
    assert out["only_a_right"] == 10
    assert out["oracle_ler_bound"] == 0.0
    # conditional undefined when A never fails
    assert out["p_tesseract_succeeds_given_bposd_fails"] is None


# --- per_shot_fails ---------------------------------------------------------
def test_per_shot_fails_mask():
    """per_shot_fails = np.any(pred != obs, axis=1) — one bool per shot."""
    pred = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=bool)
    obs = np.array([[0, 0], [0, 0], [1, 1], [0, 0]], dtype=bool)
    m = per_shot_fails(pred, obs)
    assert m.dtype == bool
    assert m.tolist() == [False, True, False, True]
    assert m.shape == (4,)


# --- run_matched keep_per_shot hook -----------------------------------------
class _StubDecoder:
    """Minimal adapter sharing the circuit's DEM with an APPROVED tie-break and a
    deterministic, controllable per-shot prediction. Lets the harness hook be
    tested without building the (slow) external decoder zoo."""

    def __init__(self, dem, name, tie_break, fail_every=0):
        self.dem = dem
        self.name = name
        self.config = {"decoder": name}
        self.tie_break = tie_break
        self._n_obs = dem.num_observables
        self._fail_every = fail_every  # 0 -> always "correct" (pred == obs path)

    def decode_batch(self, dets):
        dets = np.asarray(dets, dtype=bool)
        n = dets.shape[0]
        # Predict all-zeros; we then induce a known fail pattern by flipping obs[0]
        # on a deterministic subset of shots, so per_shot_fails is predictable.
        pred = np.zeros((n, self._n_obs), dtype=bool)
        if self._fail_every:
            pred[::self._fail_every, 0] = True
        return pred


def _small_circuit():
    return build_memory(BBCode(), rounds=2, p=0.005, basis="Z", noise="si1000")


def test_run_matched_keep_per_shot_default_unchanged():
    """Default run (no keep_per_shot) must NOT add per-shot arrays to the manifest."""
    circ = _small_circuit()
    dem = circ.detector_error_model(decompose_errors=False)
    tb = sorted(APPROVED_TIE_BREAKS)[0]
    decs = [_StubDecoder(dem, "Stub", tb)]
    m = run_matched(circ, decs, shots=200, rounds=2, seed=0)
    # default manifest schema is unchanged: no per-shot arrays anywhere
    assert "dets" not in m and "obs" not in m and "per_shot" not in m
    for r in m["decoders"]:
        assert "fail_mask" not in r
        assert "per_shot_fail" not in r


def test_run_matched_keep_per_shot_returns_masks():
    """keep_per_shot=True exposes a per-decoder per-shot fail mask aligned to the
    shared shots (length == shots), plus the shared dets/obs, so T6 can feed
    decoder-vs-Tesseract masks straight into the paired bootstrap / Gate-A."""
    circ = _small_circuit()
    dem = circ.detector_error_model(decompose_errors=False)
    tbs = sorted(APPROVED_TIE_BREAKS)
    SHOTS = 200
    decs = [
        _StubDecoder(dem, "AllRight", tbs[0], fail_every=0),   # never fails
        _StubDecoder(dem, "EveryOther", tbs[1 % len(tbs)], fail_every=2),
    ]
    m = run_matched(circ, decs, shots=SHOTS, rounds=2, seed=0, keep_per_shot=True)

    # shared shots exposed once (not per-decoder duplicated obs)
    assert "obs" in m
    obs = np.asarray(m["obs"], dtype=bool)
    assert obs.shape[0] == SHOTS

    masks = {r["name"]: np.asarray(r["fail_mask"], dtype=bool) for r in m["decoders"]}
    for name, mask in masks.items():
        assert mask.shape == (SHOTS,)
        assert mask.dtype == bool

    # the per-shot mask must reconcile with the reported scalar `fails`
    for r in m["decoders"]:
        assert int(np.asarray(r["fail_mask"]).sum()) == r["fails"]

    # masks are aligned to the SAME shots: feed two of them straight into the
    # paired bootstrap / Gate-A without re-decoding.
    a = masks["EveryOther"]
    b = masks["AllRight"]
    g = gate_a(a, b)
    assert g["n"] == SHOTS
    out = gap_to_mle_bootstrap(a, b, n_boot=100, seed=0)
    assert set(out) == {"ratio", "lo", "hi"}


def test_run_matched_keep_per_shot_preserves_gates():
    """G1/G2 still enforced under keep_per_shot=True (bad DEM still raises)."""
    circ_a = _small_circuit()
    dem_a = circ_a.detector_error_model(decompose_errors=False)
    tb = sorted(APPROVED_TIE_BREAKS)[0]
    decs = [_StubDecoder(dem_a, "Stub", tb)]
    circ_b = build_memory(BBCode(), rounds=2, p=0.007, basis="Z", noise="si1000")
    assert dem_hash(circ_b.detector_error_model(decompose_errors=False)) != dem_hash(dem_a)
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        run_matched(circ_b, decs, shots=50, rounds=2, seed=0, keep_per_shot=True)
