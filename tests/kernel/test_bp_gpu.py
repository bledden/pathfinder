"""TDD for the torch edge-list GPU BP (qldpc/kernel/bp_gpu.py).

Stage-(a) GPU baseline: a device-agnostic (CPU now / CUDA later) torch min-sum BP
on the SAME CSR/CSC edge layout the CPU baseline (``bp_baseline.py``) uses,
**batched over shots**, that reproduces ``BpBaseline`` BIT-IDENTICALLY for one
iteration. This is the honest GPU baseline + latency reference the later Triton
circulant kernel (T7b-b) must beat.

The tests run on CPU (``device="cpu"``); the GPU latency run happens on a pod.

Gates:
  1. ONE-ITERATION BIT-IDENTICAL: BpGpu's per-edge check->bit and bit->check
     messages (+ posterior) after one iteration equal BpBaseline's to atol 1e-6
     on (a) the 3-bit repetition code and (b) a small SI1000 BB memory DEM.
  2. FULL-DECODE EQUIVALENCE: BpGpu.decode_batch == BpBaseline.decode_batch (same
     predicted observables, all shots) on the small BB DEM (~500 shots).
  3. bench_latency returns a well-formed dict on CPU (no speed assertion).
"""
import numpy as np
import pytest
import torch

from bb_code import BBCode
from canon_dem import extract
from qldpc.foundation.circuits import build_memory
from qldpc.kernel.bp_baseline import BpBaseline
from qldpc.kernel.bp_gpu import BpGpu

MS = 0.625  # ms_scaling_factor (normalized min-sum), matches the CPU baseline

DEVICE = "cpu"  # local correctness gate runs on CPU; pod runs device="cuda"


# --------------------------------------------------------------------------- #
# Shared small fixtures.                                                       #
# --------------------------------------------------------------------------- #
def _rep_code():
    """3-bit repetition code H + distinct priors + a non-trivial syndrome."""
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    p = np.array([0.10, 0.20, 0.05])
    syndrome = np.array([1, 0], dtype=np.uint8)
    return H, p, syndrome


BB_SHOTS = 500
BB_SEED = 0
BB_ROUNDS = 2
BB_P = 0.005


@pytest.fixture(scope="module")
def bb_dem_shots():
    circ = build_memory(BBCode(), rounds=BB_ROUNDS, p=BB_P, basis="Z",
                        noise="si1000")
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=BB_SEED).sample(
        BB_SHOTS, separate_observables=True)
    return dem, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


# --------------------------------------------------------------------------- #
# 1. ONE-ITERATION BIT-IDENTICAL to BpBaseline.                                #
# --------------------------------------------------------------------------- #
def test_one_iter_bit_identical_rep_code():
    """One iteration of BpGpu's per-edge messages == BpBaseline's (rep code)."""
    H, p, syndrome = _rep_code()
    cpu = BpBaseline(H, priors=p, max_iter=1, ms_scaling_factor=MS,
                     schedule="parallel")
    ref = cpu.run_iterations(syndrome, n_iter=1, return_messages=True)

    gpu = BpGpu(H, priors=p, max_iter=1, ms_scaling_factor=MS)
    trace = gpu.run_iterations(syndrome, n_iter=1, device=DEVICE,
                               return_messages=True)

    # Per-edge check->bit and bit->check messages, addressable by (check, bit).
    for key, want in ref["check_to_bit"].items():
        assert np.isclose(trace["check_to_bit"][key], want, rtol=0, atol=1e-6), (
            f"check_to_bit{key}: got {trace['check_to_bit'][key]} want {want}")
    for key, want in ref["bit_to_check"].items():
        assert np.isclose(trace["bit_to_check"][key], want, rtol=0, atol=1e-6), (
            f"bit_to_check{key}: got {trace['bit_to_check'][key]} want {want}")
    assert np.allclose(trace["posterior"], ref["posterior"], rtol=0, atol=1e-6)
    assert np.array_equal(trace["hard"], ref["hard"])


def test_one_iter_bit_identical_bb_dem(bb_dem_shots):
    """One iteration bit-identical on a real BB DEM, across many shots/edges.

    Pin every per-edge message (check->bit, bit->check) and posterior LLR for a
    handful of distinct syndromes drawn from the sampled shots."""
    dem, dets, _ = bb_dem_shots
    cpu = BpBaseline.from_dem(dem, max_iter=1, ms_scaling_factor=MS,
                              schedule="parallel")
    gpu = BpGpu.from_dem(dem, max_iter=1, ms_scaling_factor=MS)

    syn = dets.astype(np.uint8)
    # A spread of syndromes (including some all-zero / heavy ones if present).
    idxs = list(range(0, min(20, syn.shape[0])))
    for i in idxs:
        ref = cpu.run_iterations(syn[i], n_iter=1, return_messages=True)
        trace = gpu.run_iterations(syn[i], n_iter=1, device=DEVICE,
                                   return_messages=True)
        # Compare the full per-edge message arrays (not just a few hand edges).
        c2b_ref = np.array([ref["check_to_bit"][k]
                            for k in sorted(ref["check_to_bit"])])
        c2b_got = np.array([trace["check_to_bit"][k]
                            for k in sorted(trace["check_to_bit"])])
        b2c_ref = np.array([ref["bit_to_check"][k]
                            for k in sorted(ref["bit_to_check"])])
        b2c_got = np.array([trace["bit_to_check"][k]
                            for k in sorted(trace["bit_to_check"])])
        assert np.allclose(c2b_got, c2b_ref, rtol=0, atol=1e-6), (
            f"shot {i}: check->bit mismatch max "
            f"{np.max(np.abs(c2b_got - c2b_ref)) if len(c2b_ref) else 0}")
        assert np.allclose(b2c_got, b2c_ref, rtol=0, atol=1e-6), (
            f"shot {i}: bit->check mismatch")
        assert np.allclose(trace["posterior"], ref["posterior"], rtol=0,
                           atol=1e-6), f"shot {i}: posterior mismatch"


def test_batched_one_iter_matches_per_shot(bb_dem_shots):
    """The batched (shots x edges) path must equal the per-shot path: run one
    iteration on a batch and confirm each row matches the single-syndrome run."""
    dem, dets, _ = bb_dem_shots
    gpu = BpGpu.from_dem(dem, max_iter=1, ms_scaling_factor=MS)
    syn = dets.astype(np.uint8)[:16]

    # Batched posteriors after one iteration.
    batch_post = gpu.run_iterations_batch(syn, n_iter=1, device=DEVICE)
    for i in range(syn.shape[0]):
        single = gpu.run_iterations(syn[i], n_iter=1, device=DEVICE)["posterior"]
        assert np.allclose(batch_post[i], single, rtol=0, atol=1e-6), (
            f"row {i}: batched posterior != single-shot posterior")


# --------------------------------------------------------------------------- #
# 2. FULL-DECODE EQUIVALENCE to BpBaseline (same predicted observables).        #
# --------------------------------------------------------------------------- #
def test_full_decode_equivalence(bb_dem_shots):
    """BpGpu.decode_batch == BpBaseline.decode_batch on the SAME shots.

    Both are the same deterministic flooding min-sum; we require exact per-shot
    observable match. If float-order ever diverges this asserts >=99% exact AND
    LER within one shot (documented fallback)."""
    dem, dets, obs = bb_dem_shots
    cpu = BpBaseline.from_dem(dem, max_iter=30, ms_scaling_factor=MS,
                              schedule="parallel")
    gpu = BpGpu.from_dem(dem, max_iter=30, ms_scaling_factor=MS)

    pred_cpu = cpu.decode_batch(dets)
    pred_gpu = gpu.decode_batch(dets, device=DEVICE)

    assert pred_gpu.shape == pred_cpu.shape
    assert pred_gpu.dtype == bool

    exact = np.all(pred_gpu == pred_cpu, axis=1)
    frac_exact = float(exact.mean())

    ler_cpu = int(np.any(pred_cpu != obs, axis=1).sum())
    ler_gpu = int(np.any(pred_gpu != obs, axis=1).sum())

    if frac_exact < 1.0:
        # Documented fallback: float-order divergence on near-ties.
        assert frac_exact >= 0.99, (
            f"per-shot exact match {frac_exact:.4f} < 0.99")
        assert abs(ler_gpu - ler_cpu) <= 1, (
            f"LER differs by >1 shot: cpu={ler_cpu} gpu={ler_gpu}")
    else:
        assert frac_exact == 1.0
        assert ler_gpu == ler_cpu


def test_decode_batch_shape_and_dtype(bb_dem_shots):
    dem, dets, _ = bb_dem_shots
    gpu = BpGpu.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    pred = gpu.decode_batch(dets, device=DEVICE)
    assert pred.shape == (BB_SHOTS, dem.num_observables)
    assert pred.dtype == bool


# --------------------------------------------------------------------------- #
# 3. bench_latency returns a well-formed dict on CPU.                          #
# --------------------------------------------------------------------------- #
def test_bench_latency_well_formed(bb_dem_shots):
    dem, _, _ = bb_dem_shots
    out = BpGpu.bench_latency(dem, shots=64, device=DEVICE, n_warmup=1,
                              n_iter=2)
    assert isinstance(out, dict)
    for k in ("mean_ms", "p99_9_ms", "throughput_shots_per_s", "device",
              "shots", "n_iter"):
        assert k in out, f"missing key {k}"
    assert out["device"] == DEVICE
    assert out["shots"] == 64
    assert out["mean_ms"] > 0.0
    assert out["throughput_shots_per_s"] > 0.0
