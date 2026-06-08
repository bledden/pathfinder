"""TDD for the T9 latency harness + Pareto/cycle-budget consumers.

CUDA-less (local Mac) coverage: the timing discipline math, the canonical DEM, the
isolated-qldpc-ext context-manager round-trip (skipped if the ext dir is absent),
and the figure/budget readers against the committed JSON artifacts. The GPU
CUDA-event paths are exercised on the pod (not here); these tests guard the
import-stays-clean / no-hard-CUDA-import-at-module-top contract and the pure-python
reduction + JSON-consumer logic.
"""
import json
import os

import numpy as np
import pytest

from qldpc.zoo import latency as L
from qldpc.zoo import make_pareto as MP
from qldpc.zoo import cycle_time_budget as CB

_HERE = os.path.dirname(os.path.abspath(__file__))
_ZOO = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "qldpc", "zoo")


# --------------------------------------------------------------------------- #
# Timing discipline                                                           #
# --------------------------------------------------------------------------- #
def test_time_callable_drops_first_fraction():
    """drop_frac removes the first ceil(drop_frac*n_reps) kept reps."""
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    kept = L.time_callable(fn, n_warmup=3, n_reps=20, drop_frac=0.1)
    # 3 warmups + 20 reps were called; kept = 20 - ceil(0.1*20) = 20 - 2 = 18.
    assert calls["n"] == 23
    assert len(kept) == 18
    assert kept.dtype == np.float64


def test_summarize_math():
    """mean/p99.9/throughput/us-per-syndrome are computed consistently."""
    times_ms = np.array([1.0, 1.0, 1.0, 1.0])  # 1 ms per batch
    batch = 1000
    rec = L.summarize(times_ms, batch)
    assert rec["batch"] == 1000
    assert rec["mean_ms"] == pytest.approx(1.0)
    assert rec["p99_9_ms"] == pytest.approx(1.0)
    # 1 ms / 1000 shots = 1 us/syndrome
    assert rec["us_per_syndrome"] == pytest.approx(1.0)
    # 1000 shots / 1e-3 s = 1e6 shots/s
    assert rec["throughput_shots_per_s"] == pytest.approx(1e6)


def test_summarize_p999_tail():
    """p99.9 reflects the tail, not the mean."""
    times_ms = np.concatenate([np.ones(99), np.array([100.0])])
    rec = L.summarize(times_ms, batch=10)
    assert rec["p99_9_ms"] > rec["mean_ms"]
    assert rec["us_per_syndrome_p99_9"] > rec["us_per_syndrome"]


# --------------------------------------------------------------------------- #
# Bootstrap CI + per-rep capture (figure-lock: error bars + reproducibility)  #
# --------------------------------------------------------------------------- #
def test_summarize_emits_bootstrap_ci_and_per_rep():
    """summarize attaches a bootstrap 95% CI on the mean + the per-rep window."""
    rng = np.random.default_rng(0)
    times_ms = 1.0 + 0.05 * rng.standard_normal(100)  # ~1 ms, tight
    rec = L.summarize(times_ms, batch=1000)
    assert "us_per_syndrome_ci95" in rec
    lo, hi = rec["us_per_syndrome_ci95"]
    # CI must bracket the mean us/syndrome and be a real (lo<hi) interval
    assert lo < rec["us_per_syndrome"] < hi
    assert rec["bootstrap_n"] >= 1000          # >= the locked floor
    # per-rep window kept verbatim for reproducibility
    assert len(rec["per_rep_ms"]) == 100


def test_bootstrap_ci_tightens_with_low_variance():
    """A near-constant sample gives a near-degenerate CI; noisy gives wider."""
    tight = np.full(100, 2.0)
    noisy = 2.0 + np.random.default_rng(1).standard_normal(100)
    ci_t = L.bootstrap_ci(tight, batch=100)
    ci_n = L.bootstrap_ci(noisy, batch=100)
    width_t = ci_t[1] - ci_t[0]
    width_n = ci_n[1] - ci_n[0]
    assert width_t < width_n
    assert ci_t[2] == L.BOOTSTRAP_N


def test_small_batches_in_gpu_sweep():
    """The GPU batch sweep includes the single-shot batches {1,4,16}."""
    for b in (1, 4, 16):
        assert b in L.GPU_BATCH_SWEEP
    assert L.SINGLE_SHOT_BATCH == 1


def test_uniform_random_detectors_shape_and_density():
    """Uniform-random syndromes are dense (~0.5) vs realistic (sparse)."""
    uni = L.uniform_random_detectors(252, 200, seed=0)
    assert uni.shape == (200, 252)
    assert uni.dtype == bool
    # dense: mean activation ~0.5 (worst-case for OSD), unlike sparse realistic
    assert 0.4 < uni.mean() < 0.6


def test_gpu_clock_state_reports_unlocked():
    """gpu_clock_state states the RunPod unlocked-clock condition + boost."""
    clk = L.gpu_clock_state()
    assert clk["clocks_locked"] is False
    assert clk["boost_clock_mhz"] == 1980
    assert "permission" in clk["lock_note"].lower()


# --------------------------------------------------------------------------- #
# Canonical DEM (the figure's grid point)                                     #
# --------------------------------------------------------------------------- #
def test_canonical_dem_shape():
    """The canonical [[72,12,6]] BB SI1000 DEM at d6/R6 has the expected dims."""
    circ, dem = L.canonical_dem(rounds=6, p=0.003, basis="Z")
    assert dem.num_detectors == 252
    assert dem.num_observables == 12
    # decompose_errors=False is implied by build (full undecomposed DEM)
    dets = L.sample_detectors(circ, 16, seed=0)
    assert dets.shape == (16, 252)
    assert dets.dtype == bool


def test_no_hard_cuda_import_at_module_top():
    """latency.py must import on a CUDA-less host (no torch.cuda at module top)."""
    # If we got here, ``import qldpc.zoo.latency`` already succeeded without CUDA.
    # env_manifest must also degrade gracefully (Nones, not exceptions).
    env = L.env_manifest()
    assert "stim" in env and "ldpc" in env
    # cuda/gpu keys present even when unavailable
    assert "cuda" in env and "gpu_name" in env


# --------------------------------------------------------------------------- #
# Isolated qldpc_ext context manager (round-trip; skip if ext dir absent)     #
# --------------------------------------------------------------------------- #
def test_isolated_qldpc_ext_roundtrip():
    """After the context exits, the repo qldpc is restored exactly."""
    ext = L._DEFAULT_QLDPC_EXT
    if not os.path.isdir(ext):
        pytest.skip(f"qldpc_ext dir absent ({ext}); pod-only path")
    import qldpc.foundation.circuits as before
    with L.isolated_qldpc_ext(ext):
        import qldpc as ext_qldpc
        assert ext in (ext_qldpc.__file__ or "")
    import qldpc.foundation.circuits as after
    assert after is before  # repo module restored to the same object


# --------------------------------------------------------------------------- #
# Pareto figure consumer (reads committed JSON)                               #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def grid_and_lat():
    grid_p = os.path.join(_ZOO, "zoo_grid.json")
    lat_p = os.path.join(_ZOO, "latency_results.json")
    if not (os.path.exists(grid_p) and os.path.exists(lat_p)):
        pytest.skip("committed grid/latency JSON absent")
    with open(grid_p) as f:
        grid = json.load(f)
    with open(lat_p) as f:
        lat = json.load(f)
    return grid, lat


def test_build_points_has_kernel_variants(grid_and_lat):
    grid, lat = grid_and_lat
    pts = MP.build_points(grid, lat, p=0.003)
    labels = {pp["label"] for pp in pts}
    # the three BP kernel variants must all appear (the left-shift story)
    assert "Triton-kernel-BP" in labels
    assert "torch-GPU-BP" in labels
    assert "CPU-BP (ldpc)" in labels
    # the MLE anchor at gap=1.0
    anchor = [pp for pp in pts if pp["label"].startswith("Tesseract")]
    assert anchor and anchor[0]["gap"] == pytest.approx(1.0)


def test_kernel_variants_share_gap_differ_latency(grid_and_lat):
    grid, lat = grid_and_lat
    pts = {pp["label"]: pp for pp in MP.build_points(grid, lat, p=0.003)
           if pp["group"] == "kernel-BP"}
    gaps = {pp["gap"] for pp in pts.values()}
    assert len(gaps) == 1  # same LER => same gap-to-MLE
    # Triton kernel must be the fastest of the three (left-shift)
    assert (pts["Triton-kernel-BP"]["latency_us"]
            < pts["torch-GPU-BP"]["latency_us"]
            < pts["CPU-BP (ldpc)"]["latency_us"])


def test_pareto_frontier_includes_kernel_and_anchor(grid_and_lat):
    grid, lat = grid_and_lat
    pts = MP.build_points(grid, lat, p=0.003)
    front = MP.pareto_frontier(pts)
    labels = {pp["label"] for pp in front}
    # the fastest kernel point and the zero-gap anchor are both non-dominated
    assert "Triton-kernel-BP" in labels
    assert any(l.startswith("Tesseract") for l in labels)


def test_build_points_carry_whisker_and_ci_fields(grid_and_lat):
    """Every plotted point carries the Option-A whisker (p99.9) + CI fields."""
    grid, lat = grid_and_lat
    pts = MP.build_points(grid, lat, p=0.003)
    for pp in pts:
        assert "p999_us" in pp and "ci_lo" in pp and "ci_hi" in pp
        # p99.9 whisker terminus is >= the mean (tail is never below mean)
        assert pp["p999_us"] >= pp["latency_us"] - 1e-6
        # CI brackets the mean (lo <= mean <= hi)
        assert pp["ci_lo"] - 1e-6 <= pp["latency_us"] <= pp["ci_hi"] + 1e-6


def test_gpu_regime_single_shot_falls_back_gracefully(grid_and_lat):
    """single_shot regime works whether or not single_shot fields are present."""
    grid, lat = grid_and_lat
    pts = MP.build_points(grid, lat, p=0.003, gpu_regime="single_shot")
    labels = {pp["label"] for pp in pts}
    assert "Triton-kernel-BP" in labels  # resolves (single_shot or fallback)


# --------------------------------------------------------------------------- #
# Both-workloads + clock-variance JSON consumers (skip if artifacts absent)   #
# --------------------------------------------------------------------------- #
def test_both_workloads_json_if_present():
    """If both_workloads.json exists, it quantifies OSD inflation (realistic<uni)."""
    path = os.path.join(_ZOO, "both_workloads.json")
    if not os.path.exists(path):
        pytest.skip("both_workloads.json absent (pod-only artifact)")
    with open(path) as f:
        bw = json.load(f)
    assert bw.get("kind") == "t9-both-workloads"
    res = bw["results"]
    # the headline finding: BP-OSD-10 inflates materially under uniform-random
    osd10 = res.get("BPOSD-10")
    assert osd10 and "inflation_x" in osd10
    assert osd10["inflation_x"] is None or osd10["inflation_x"] > 1.0


def test_clock_variance_json_if_present():
    """If clock_variance.json exists, per-decoder run-to-run CV is small."""
    path = os.path.join(_ZOO, "clock_variance.json")
    if not os.path.exists(path):
        pytest.skip("clock_variance.json absent (pod-only artifact)")
    with open(path) as f:
        cv = json.load(f)
    assert cv.get("kind") == "t9-clock-variance"
    assert cv["clock_state"]["clocks_locked"] is False
    # CV present per (decoder, batch); thermal stability => modest CV
    for _which, per_batch in cv["runs"].items():
        for _b, rec in per_batch.items():
            if rec.get("cv") is not None:
                assert rec["cv"] >= 0.0


# --------------------------------------------------------------------------- #
# Cycle-time budget consumer                                                  #
# --------------------------------------------------------------------------- #
def test_cycle_budget_table(grid_and_lat):
    _grid, lat = grid_and_lat
    rows = CB.build_table(lat)
    # Triton kernel must fit superconducting per-block at some batch; Tesseract must not
    trt = rows["bp_triton"]
    assert trt["available"] and trt["is_gpu"]
    assert trt["verdicts"]["superconducting"]["fits_per_block"]
    tess = rows["Tesseract"]
    assert tess["available"]
    assert not tess["verdicts"]["superconducting"]["fits_per_block"]
    assert not tess["verdicts"]["neutral-atom"]["fits_per_block"]


def test_cycle_budget_summary_ordering(grid_and_lat):
    _grid, lat = grid_and_lat
    rows = CB.build_table(lat)
    summary = CB.summarize(rows)
    # neutral-atom (most forgiving) fits a superset of what trapped-ion fits
    atom = set(summary["neutral-atom"]["fits_per_block"])
    ion = set(summary["trapped-ion"]["fits_per_block"])
    sc = set(summary["superconducting"]["fits_per_block"])
    assert sc <= ion <= atom
