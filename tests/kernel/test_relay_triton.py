"""TDD for the Triton Relay-BP kernel (qldpc/kernel/relay_triton.py).

Relay-BP (IBM, arXiv:2506.01779 "Improved belief propagation is sufficient for
real-time decoding of quantum memory") is a min-sum BP decoder with **disordered
per-node memory** run as a **relay ensemble of legs**. ``RelayBpTriton`` builds on
the existing fused min-sum check/bit-update Triton kernels (``bp_triton.py``) and
adds (a) the per-node gamma-memory term in the bit update, (b) the host-side relay
leg loop, (c) the nconv ensemble selection (lowest-weight valid solution).

The bar is **LER-identity** with the ``relay_bp`` oracle (the compiled IBM Rust
``RelayDecoderF64`` via the zoo ``RelayBPAdapter``): matching the logical-error
COUNT within fp/ensemble noise on the canonical DEM, plus a reported per-shot
agreement %. NOT Rust-bit-identity (the gamma-draw RNG lives in compiled Rust and
fp32-vs-fp64 differs).

This module SKIPS cleanly without triton/CUDA (the local Mac) and RUNS on the pod.

Gates (pod only):
  1. PRE-LEG MIN-SUM IDENTITY: one min-sum iteration (no memory, alpha=1) matches
     relay_bp's MinSumBPDecoderF64 posterior to fp tolerance (the textbook
     exclude-self min-sum primitive both share).
  2. MEMORY-TERM IDENTITY: a few flooding iterations with a uniform gamma0 memory
     term match relay_bp's MinSumBPDecoderF64(gamma0=...) hard decision on shots
     that converge (the Eq-4 memory update is implemented correctly).
  3. FULL RELAY LER-IDENTITY: RelayBpTriton.decode_batch vs the relay_bp oracle on
     N>=2000 canonical shots -- logical-error COUNT within noise + per-shot
     observable agreement reported.
"""
import numpy as np
import pytest

triton = pytest.importorskip("triton")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for the Triton Relay-BP kernel",
                allow_module_level=True)
relay_bp = pytest.importorskip("relay_bp")

from bb_code import BBCode
from qldpc.foundation.circuits import build_memory
from qldpc.kernel.relay_triton import RelayBpTriton

DEVICE = "cuda"
BB_SHOTS = 2000
BB_SEED = 0
BB_ROUNDS = 6
BB_P = 0.003

# Relay-BP reference config (the zoo _RELAY_BP_DEFAULTS).
RELAY_CFG = dict(
    gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60,
    gamma_dist_interval=(-0.24, 0.66), stop_nconv=5, stopping_criterion="nconv",
)


@pytest.fixture(scope="module")
def bb_dem_shots():
    circ = build_memory(BBCode(), rounds=BB_ROUNDS, p=BB_P, basis="Z",
                        noise="si1000")
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=BB_SEED).sample(
        BB_SHOTS, separate_observables=True)
    return dem, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


def _relay_oracle(dem):
    """The relay_bp oracle decoder (zoo RelayBPAdapter wiring)."""
    import relay_bp
    from relay_bp.stim import CheckMatrices
    cm = CheckMatrices.from_dem(dem)
    dec = relay_bp.RelayDecoderF64(cm.check_matrix,
                                   error_priors=cm.error_priors, **RELAY_CFG)
    runner = relay_bp.ObservableDecoderRunner(
        dec, cm.observables_matrix, include_decode_result=False)
    return cm, dec, runner


def test_pre_leg_minsum_identity(bb_dem_shots):
    """One min-sum iteration (alpha=1, no memory) matches relay_bp's
    MinSumBPDecoderF64 posterior to fp tolerance -- the shared exclude-self
    min-sum primitive."""
    import relay_bp
    from relay_bp.stim import CheckMatrices
    dem, dets, _ = bb_dem_shots
    cm = CheckMatrices.from_dem(dem)
    syn = dets.astype(np.uint8)

    trt = RelayBpTriton.from_dem(dem, **RELAY_CFG)
    post_trt = trt.minsum_posterior_batch(syn, n_iter=1, gamma=0.0,
                                          alpha=1.0, device=DEVICE)  # (S, N)

    dec = relay_bp.MinSumBPDecoderF64(cm.check_matrix,
                                      error_priors=cm.error_priors,
                                      max_iter=1, alpha=1.0, gamma0=0.0)
    post_ref = np.stack([np.asarray(dec.decode_detailed(syn[i]).posterior_ratios)
                         for i in range(min(64, len(syn)))])
    maxdiff = float(np.max(np.abs(post_trt[:post_ref.shape[0]] - post_ref)))
    assert maxdiff < 1e-3, f"pre-leg min-sum posterior maxdiff {maxdiff} too large"


def test_memory_term_identity(bb_dem_shots):
    """Flooding iterations with a uniform gamma0 memory term reproduce relay_bp's
    MinSumBPDecoderF64(gamma0=...) hard decision >=99% per-bit (Eq-4 memory)."""
    import relay_bp
    from relay_bp.stim import CheckMatrices
    dem, dets, _ = bb_dem_shots
    cm = CheckMatrices.from_dem(dem)
    syn = dets[:256].astype(np.uint8)

    n_iter = 30
    trt = RelayBpTriton.from_dem(dem, **RELAY_CFG)
    post_trt = trt.minsum_posterior_batch(syn, n_iter=n_iter, gamma=0.1,
                                          alpha=1.0, device=DEVICE)
    hard_trt = (post_trt < 0.0).astype(np.uint8)

    dec = relay_bp.MinSumBPDecoderF64(cm.check_matrix,
                                      error_priors=cm.error_priors,
                                      max_iter=n_iter, alpha=1.0, gamma0=0.1)
    hard_ref = np.stack([np.asarray(dec.decode_detailed(syn[i]).decoding)
                         for i in range(len(syn))]).astype(np.uint8)
    per_bit = float((hard_trt == hard_ref).mean())
    assert per_bit >= 0.99, f"memory-term per-bit agreement {per_bit:.5f} < 0.99"


def test_full_relay_ler_identity(bb_dem_shots):
    """Full relay decode: RelayBpTriton.decode_batch vs the relay_bp oracle --
    logical-error COUNT within noise + per-shot observable agreement reported."""
    dem, dets, obs = bb_dem_shots
    _, _, runner = _relay_oracle(dem)
    pred_ref = np.asarray(
        runner.decode_observables_batch(dets.astype(np.uint8))) % 2
    pred_ref = pred_ref.astype(bool)
    if pred_ref.ndim == 1:
        pred_ref = pred_ref.reshape(-1, 1)

    trt = RelayBpTriton.from_dem(dem, **RELAY_CFG)
    pred_trt = trt.decode_batch(dets, device=DEVICE)

    assert pred_trt.shape == pred_ref.shape
    assert pred_trt.dtype == bool

    ler_ref = int(np.any(pred_ref != obs, axis=1).sum())
    ler_trt = int(np.any(pred_trt != obs, axis=1).sum())
    per_shot = float(np.all(pred_trt == pred_ref, axis=1).mean())
    print(f"\nRELAY LER-IDENTITY: oracle={ler_ref} triton={ler_trt} "
          f"(N={len(dets)}) per-shot-agreement={per_shot:.4f}")
    # LER within ensemble/fp noise: a small absolute tolerance on the count.
    assert abs(ler_trt - ler_ref) <= max(5, int(0.01 * len(dets))), (
        f"relay LER differs too much: oracle={ler_ref} triton={ler_trt}")


def test_decode_batch_shape_and_dtype(bb_dem_shots):
    dem, dets, _ = bb_dem_shots
    trt = RelayBpTriton.from_dem(dem, **RELAY_CFG)
    pred = trt.decode_batch(dets[:128], device=DEVICE)
    assert pred.shape == (128, dem.num_observables)
    assert pred.dtype == bool


def test_bench_latency_well_formed(bb_dem_shots):
    dem, _, _ = bb_dem_shots
    out = RelayBpTriton.bench_latency(dem, shots=256, device=DEVICE,
                                      n_warmup=2, n_iter=3)
    assert isinstance(out, dict)
    for k in ("mean_ms", "p99_9_ms", "throughput_shots_per_s", "device",
              "shots", "n_iter"):
        assert k in out
    assert out["mean_ms"] > 0.0
    assert out["throughput_shots_per_s"] > 0.0
