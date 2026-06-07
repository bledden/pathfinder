"""TDD for the Triton min-sum BP kernel (qldpc/kernel/bp_triton.py).

Stage-(b) GPU kernel: a Triton normalized-min-sum BP on the SAME DEM Tanner-graph
edge layout the CPU baseline (``bp_baseline.py``) and the torch GPU baseline
(``bp_gpu.py``) use, batched over shots, that reproduces ``BpBaseline`` /
``BpGpu`` FUNCTIONALLY for one flooding iteration (float32 GPU near-tie flips on
the two-smallest-magnitude min are tolerated -- the gate is >=99.5% hard-decision
agreement + LER within noise, NOT bit-identity).

This whole module SKIPS cleanly when triton or CUDA is unavailable (the local
Mac), and RUNS on the H200 pod.

Gates (pod only):
  1. ONE-ITERATION HARD AGREEMENT: BpTriton's per-edge hard sign + per-bit hard
     decision after one iteration agree with BpBaseline >=99.5% on the BB DEM.
  2. FULL-DECODE LER: BpTriton.decode_batch vs BpBaseline/BpGpu on the BB DEM --
     per-shot hard-decision agreement >=99.5% AND LER within noise.
"""
import numpy as np
import pytest

triton = pytest.importorskip("triton")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for the Triton BP kernel", allow_module_level=True)

from bb_code import BBCode
from qldpc.foundation.circuits import build_memory
from qldpc.kernel.bp_baseline import BpBaseline
from qldpc.kernel.bp_gpu import BpGpu
from qldpc.kernel.bp_triton import BpTriton

MS = 0.625
DEVICE = "cuda"

BB_SHOTS = 2000
BB_SEED = 0
BB_ROUNDS = 6
BB_P = 0.003


@pytest.fixture(scope="module")
def bb_dem_shots():
    circ = build_memory(BBCode(), rounds=BB_ROUNDS, p=BB_P, basis="Z",
                        noise="si1000")
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=BB_SEED).sample(
        BB_SHOTS, separate_observables=True)
    return dem, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


def test_one_iter_hard_agreement(bb_dem_shots):
    """One iteration: BpTriton per-bit hard decision agrees with BpBaseline
    >=99.5% (per-bit, over all shots x bits)."""
    dem, dets, _ = bb_dem_shots
    cpu = BpBaseline.from_dem(dem, max_iter=1, ms_scaling_factor=MS)
    trt = BpTriton.from_dem(dem, max_iter=1, ms_scaling_factor=MS)

    syn = dets.astype(np.uint8)
    post_trt = trt.run_iterations_batch(syn, n_iter=1, device=DEVICE)  # (S, N)
    hard_trt = (post_trt < 0.0).astype(np.uint8)

    hard_cpu = np.empty_like(hard_trt)
    for i in range(syn.shape[0]):
        hard_cpu[i] = cpu.run_iterations(syn[i], n_iter=1)["hard"]

    agree = float((hard_trt == hard_cpu).mean())
    assert agree >= 0.995, f"one-iter hard agreement {agree:.5f} < 0.995"


def test_full_decode_per_bit_agreement(bb_dem_shots):
    """Full decode (max_iter=30): BpTriton per-BIT hard decision agrees with
    BpBaseline >=99.5% (the spike's hard-decision gate). float32 GPU near-tie
    flips on the two-smallest-magnitude min are expected; per-shot all-bits-match
    is correspondingly lower but the per-bit rate is ~1.0."""
    dem, dets, _ = bb_dem_shots
    cpu = BpBaseline.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    trt = BpTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS)

    syn = dets.astype(np.uint8)
    post_trt = trt.run_iterations_batch(syn, n_iter=30, device=DEVICE)
    hard_trt = (post_trt < 0.0).astype(np.uint8)
    hard_cpu = np.empty_like(hard_trt)
    for i in range(syn.shape[0]):
        hard_cpu[i] = cpu.run_iterations(syn[i], n_iter=30)["hard"]

    per_bit = float((hard_trt == hard_cpu).mean())
    assert per_bit >= 0.995, f"30-iter per-bit hard agreement {per_bit:.5f} < 0.995"


def test_full_decode_ler_agreement(bb_dem_shots):
    """Full decode (max_iter=30): BpTriton vs BpBaseline -- LER within noise.

    Per-shot observable match is reported (float32 near-ties flip whole shots),
    but the binding gate is LER (logical error rate) within a handful of shots."""
    dem, dets, obs = bb_dem_shots
    cpu = BpBaseline.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    trt = BpTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS)

    pred_cpu = cpu.decode_batch(dets)
    pred_trt = trt.decode_batch(dets, device=DEVICE)

    assert pred_trt.shape == pred_cpu.shape
    assert pred_trt.dtype == bool

    ler_cpu = int(np.any(pred_cpu != obs, axis=1).sum())
    ler_trt = int(np.any(pred_trt != obs, axis=1).sum())
    # LER within noise: a handful of near-tie flips at most.
    assert abs(ler_trt - ler_cpu) <= max(3, int(0.005 * len(dets))), (
        f"LER differs too much: cpu={ler_cpu} triton={ler_trt}")


def test_decode_batch_shape_and_dtype(bb_dem_shots):
    dem, dets, _ = bb_dem_shots
    trt = BpTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    pred = trt.decode_batch(dets, device=DEVICE)
    assert pred.shape == (BB_SHOTS, dem.num_observables)
    assert pred.dtype == bool


def test_matches_bp_gpu_hard(bb_dem_shots):
    """BpTriton per-bit hard decisions agree with the torch BpGpu baseline
    >=99.5% (both the same flooding min-sum; float32 vs float64 near-tie only)."""
    dem, dets, _ = bb_dem_shots
    gpu = BpGpu.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    trt = BpTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    syn = dets.astype(np.uint8)
    post_gpu = gpu.run_iterations_batch(syn, n_iter=30, device=DEVICE)
    post_trt = trt.run_iterations_batch(syn, n_iter=30, device=DEVICE)
    hard_gpu = (post_gpu < 0.0).astype(np.uint8)
    hard_trt = (post_trt < 0.0).astype(np.uint8)
    per_bit = float((hard_gpu == hard_trt).mean())
    assert per_bit >= 0.995, f"BpTriton vs BpGpu per-bit agreement {per_bit:.5f}"


def test_bench_latency_well_formed(bb_dem_shots):
    dem, _, _ = bb_dem_shots
    out = BpTriton.bench_latency(dem, shots=256, device=DEVICE, n_warmup=2,
                                 n_iter=3)
    assert isinstance(out, dict)
    for k in ("mean_ms", "p99_9_ms", "throughput_shots_per_s", "device",
              "shots", "n_iter"):
        assert k in out
    assert out["mean_ms"] > 0.0
    assert out["throughput_shots_per_s"] > 0.0
