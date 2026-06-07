"""TDD for the portable min-sum BP baseline (qldpc/kernel/bp_baseline.py).

This is stage-(a): a from-scratch normalized-min-sum BP on the Tanner graph using
the SAME CSR/CSC layout the GPU "circulant-unroll" kernel (T7b) will consume. Its
four-phase per-iteration primitive (gather neighbor messages, exclude-self,
per-check min-sum transform, scatter) is exactly what the kernel will fuse, so we:

  1. pin the message MATH bit-identically against a HAND-DERIVED one-iteration
     normalized-min-sum reference on a tiny hand-checkable Tanner graph (the
     3-bit repetition code), and
  2. require FUNCTIONAL LER-equivalence to ldpc.BpDecoder (same config) on a small
     SI1000 BB memory DEM over the SAME shots.

Config matches the zoo BP adapter: bp_method='minimum_sum', ms_scaling_factor=0.625,
schedule='parallel', max_iter=30.
"""
import numpy as np
import pytest

from bb_code import BBCode
from canon_dem import extract
from qldpc.foundation.circuits import build_memory
from qldpc.foundation.stats import wilson_ci
from qldpc.kernel.bp_baseline import BpBaseline

MS = 0.625  # ms_scaling_factor (normalized min-sum), matches the zoo BP adapter


# --------------------------------------------------------------------------- #
# 1. HAND-DERIVED one-iteration normalized-min-sum, 3-bit repetition code.     #
# --------------------------------------------------------------------------- #
# Tanner graph: bits {0,1,2}, checks c0={0,1}, c1={1,2}.
#   H = [[1,1,0],
#        [0,1,1]]
# LLR convention: lambda_v = log(P(x_v=0)/P(x_v=1)); positive => bit likely 0.
# Channel: per-bit error prob p_v -> prior LLR lambda_v = log((1-p_v)/p_v).
# We pick distinct priors so the min-of-others is unambiguous, and a non-trivial
# syndrome to exercise the sign flip.
#
# One normalized-min-sum iteration (parallel/flooding), edges init to lambda_v:
#   check->bit:  m_{c->v} = ms * (-1)^{syndrome_c} * prod_{v'!=v} sign(m_{v'->c})
#                                                   * min_{v'!=v} |m_{v'->c}|
#   posterior:   L_v = lambda_v + sum_{c in N(v)} m_{c->v}
#   hard dec.:   e_v = 1 if L_v < 0 else 0
# (The syndrome enters as a (-1)^{s_c} factor on each check's outgoing message: a
# flipped check inverts the agreement of its bits -- the standard min-sum coset
# correction.)
def _hand_repetition_one_iter():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    p = np.array([0.10, 0.20, 0.05])              # distinct per-bit error probs
    lam = np.log((1.0 - p) / p)                   # prior LLRs (all positive)
    syndrome = np.array([1, 0], dtype=np.uint8)   # c0 fired, c1 did not

    # Initial bit->check messages = prior LLRs (one per edge, init flooding).
    # Edges: c0-b0, c0-b1, c1-b1, c1-b2.
    # Iteration 1 check->bit (each check has exactly 2 bits, so "min/sign of
    # others" is just the single other bit's message = its prior here):
    sgn = np.sign
    # check c0 (syndrome 1 -> factor -1): bits {0,1}
    s0 = (-1.0)
    m_c0_b0 = MS * s0 * sgn(lam[1]) * abs(lam[1])   # to b0: uses other bit b1
    m_c0_b1 = MS * s0 * sgn(lam[0]) * abs(lam[0])   # to b1: uses other bit b0
    # check c1 (syndrome 0 -> factor +1): bits {1,2}
    s1 = (+1.0)
    m_c1_b1 = MS * s1 * sgn(lam[2]) * abs(lam[2])   # to b1: uses other bit b2
    m_c1_b2 = MS * s1 * sgn(lam[1]) * abs(lam[1])   # to b2: uses other bit b1

    check_to_bit = {(0, 0): m_c0_b0, (0, 1): m_c0_b1,
                    (1, 1): m_c1_b1, (1, 2): m_c1_b2}
    # Posterior LLRs after iteration 1.
    L0 = lam[0] + m_c0_b0
    L1 = lam[1] + m_c0_b1 + m_c1_b1
    L2 = lam[2] + m_c1_b2
    posterior = np.array([L0, L1, L2])
    hard = (posterior < 0).astype(np.uint8)
    return dict(H=H, priors=p, lam=lam, syndrome=syndrome,
                check_to_bit=check_to_bit, posterior=posterior, hard=hard)


def test_one_iteration_check_to_bit_matches_hand_reference():
    ref = _hand_repetition_one_iter()
    bp = BpBaseline(ref["H"], priors=ref["priors"], max_iter=1,
                    ms_scaling_factor=MS, schedule="parallel")
    trace = bp.run_iterations(ref["syndrome"], n_iter=1, return_messages=True)

    # Edge-indexed check->bit message array, addressable by (check, bit).
    got = trace["check_to_bit"]      # dict {(c, v): value}
    for key, want in ref["check_to_bit"].items():
        assert np.isclose(got[key], want, rtol=0, atol=1e-12), (
            f"check_to_bit{key}: got {got[key]} want {want}")


def test_one_iteration_posterior_and_hard_decision_match_hand_reference():
    ref = _hand_repetition_one_iter()
    bp = BpBaseline(ref["H"], priors=ref["priors"], max_iter=1,
                    ms_scaling_factor=MS, schedule="parallel")
    trace = bp.run_iterations(ref["syndrome"], n_iter=1, return_messages=True)
    assert np.allclose(trace["posterior"], ref["posterior"], rtol=0, atol=1e-12)
    e_hat = bp.decode(ref["syndrome"])  # default max_iter=1 here
    assert np.array_equal(e_hat.astype(np.uint8), ref["hard"])


def test_initial_bit_to_check_messages_are_priors():
    """Flooding init: every bit->check message starts at the bit's prior LLR.
    This pins the gather phase's starting state (phase-0 of the kernel)."""
    ref = _hand_repetition_one_iter()
    bp = BpBaseline(ref["H"], priors=ref["priors"], max_iter=1,
                    ms_scaling_factor=MS, schedule="parallel")
    trace = bp.run_iterations(ref["syndrome"], n_iter=0, return_messages=True)
    # With zero iterations the bit->check messages equal the per-edge prior LLR.
    b2c = trace["bit_to_check"]    # dict {(c, v): value}
    for (c, v), val in b2c.items():
        assert np.isclose(val, ref["lam"][v], atol=1e-12)


# --------------------------------------------------------------------------- #
# 2. Functional LER-equivalence to ldpc.BpDecoder on a small BB memory DEM.    #
# --------------------------------------------------------------------------- #
SHOTS = 1500
SEED = 0
ROUNDS = 2
P = 0.005


@pytest.fixture(scope="module")
def bb_dem_shots():
    circ = build_memory(BBCode(), rounds=ROUNDS, p=P, basis="Z", noise="si1000")
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=SEED).sample(
        SHOTS, separate_observables=True)
    return dem, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


def _ldpc_bp(dem):
    from ldpc import BpDecoder
    ex = extract(dem)
    H = ex["H"]
    pri = list(np.clip(ex["priors"], 1e-6, 1 - 1e-6))
    return BpDecoder(H, error_channel=pri, max_iter=30,
                     bp_method="minimum_sum", ms_scaling_factor=MS,
                     schedule="parallel"), ex


def test_decode_batch_shape_and_dtype(bb_dem_shots):
    dem, dets, obs = bb_dem_shots
    bp = BpBaseline.from_dem(dem, max_iter=30, ms_scaling_factor=MS,
                             schedule="parallel")
    pred = bp.decode_batch(dets)
    assert pred.shape == (SHOTS, dem.num_observables)
    assert pred.dtype == bool


def test_ler_matches_ldpc_bp_within_ci(bb_dem_shots):
    """Functional equivalence: BpBaseline's block LER must agree with ldpc.BpDecoder
    (identical config) on the SAME shots within overlapping Wilson CIs (and within a
    handful of shots). Bit-identical hard decisions to ldpc's C++ internals are not
    required -- ties / float-order can diverge; LER-equivalence + the hand-derived
    one-iteration test is the bar."""
    dem, dets, obs = bb_dem_shots
    ours = BpBaseline.from_dem(dem, max_iter=30, ms_scaling_factor=MS,
                               schedule="parallel")
    pred_ours = ours.decode_batch(dets)
    fails_ours = int(np.any(pred_ours != obs, axis=1).sum())

    ldpc_dec, ex = _ldpc_bp(dem)
    Lo = ex["Lo"].toarray().astype(np.uint8)
    syn = dets.astype(np.uint8)
    fails_ldpc = 0
    for i in range(SHOTS):
        e = np.asarray(ldpc_dec.decode(syn[i]), dtype=np.uint8)
        p = (Lo @ e) & 1
        if not np.array_equal(p.astype(bool), obs[i]):
            fails_ldpc += 1

    lo_o, hi_o = wilson_ci(fails_ours, SHOTS)
    lo_l, hi_l = wilson_ci(fails_ldpc, SHOTS)
    # Overlapping Wilson CIs AND within a handful of shots of each other.
    assert lo_o <= hi_l and lo_l <= hi_o, (
        f"CIs disjoint: ours fails={fails_ours} CI=({lo_o:.4f},{hi_o:.4f}) "
        f"vs ldpc fails={fails_ldpc} CI=({lo_l:.4f},{hi_l:.4f})")
    assert abs(fails_ours - fails_ldpc) <= max(5, int(0.05 * max(fails_ldpc, 1))), (
        f"fail counts diverge: ours={fails_ours} ldpc={fails_ldpc}")


def test_better_than_chance(bb_dem_shots):
    dem, dets, obs = bb_dem_shots
    bp = BpBaseline.from_dem(dem, max_iter=30, ms_scaling_factor=MS,
                             schedule="parallel")
    pred = bp.decode_batch(dets)
    ler = float(np.any(pred != obs, axis=1).mean())
    assert ler < 0.5
