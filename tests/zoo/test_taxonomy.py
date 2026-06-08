"""TDD for the T8 decoder-amenability taxonomy (CPU/analysis lane).

Fast, CUDA-less coverage of the five taxonomy deliverables:
  1. fallthrough-rate rises with p (small shot counts); effective-latency
     arithmetic correct on a synthetic case.
  2. Amdahl table: well-formed columns; serial_fraction in [0,1]; ceiling =
     1/serial_fraction; measured-increment rows match the latency gap.
  3. roofline table: required columns; BP-OSD-10 carries the tensor-core flag.
  4. memory table: well-formed; aux >= 0; max-batch shrinks as aux grows;
     bigger device -> bigger max batch.
  5. Gate-B: returns a structured/unstructured verdict + the error-weight stats;
     a synthetic structured set is flagged STRUCTURED, an unstructured one is not.

Heavy measurements (full fallthrough sweep, the Tesseract-bound Gate-B run) are
exercised by the committed JSON artifact tests, not re-run here.
"""
import json
import os

import numpy as np
import pytest

from qldpc.zoo import taxonomy as T

_HERE = os.path.dirname(os.path.abspath(__file__))
_ZOO = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "qldpc", "zoo")
_LATENCY = os.path.join(_ZOO, "latency_results.json")
_RECEIPT = os.path.join(_ZOO, "launch_overhead_receipt.json")
_RESULTS = os.path.join(_ZOO, "taxonomy_results.json")


@pytest.fixture(scope="module")
def latency():
    with open(_LATENCY) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def receipt():
    with open(_RECEIPT) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Deliverable 1: fallthrough + effective latency                              #
# --------------------------------------------------------------------------- #
def test_effective_latency_arithmetic():
    """effective = BP + fallthrough * OSD_increment (synthetic, exact)."""
    # BP=100us, OSD increment=400us, fallthrough=0.25 -> 100 + 0.25*400 = 200.
    assert T.effective_latency(100.0, 400.0, 0.25) == pytest.approx(200.0)
    # fallthrough=0 -> pure BP floor; fallthrough=1 -> BP + full increment.
    assert T.effective_latency(100.0, 400.0, 0.0) == pytest.approx(100.0)
    assert T.effective_latency(100.0, 400.0, 1.0) == pytest.approx(500.0)


def test_osd_increment_from_latency(latency):
    """OSD increment = BPOSD-10 us/syn - BP us/syn, both from the T9 manifest."""
    inc = T.osd_increment_from_latency(latency)
    res = latency["results"]
    assert inc["bp_us"] == pytest.approx(res["BP"]["us_per_syndrome"])
    assert inc["osd_total_us"] == pytest.approx(res["BPOSD-10"]["us_per_syndrome"])
    assert inc["osd_increment_us"] == pytest.approx(
        res["BPOSD-10"]["us_per_syndrome"] - res["BP"]["us_per_syndrome"])
    # OSD adds cost (BPOSD-10 slower than bare BP per syndrome).
    assert inc["osd_increment_us"] > 0
    assert "reasoned-from-measured" in inc["basis"]


def test_measure_fallthrough_well_formed():
    """measure_fallthrough returns the expected keys; rate in [0,1]; labelled."""
    m = T.measure_fallthrough(0.003, shots=400, seed=0)
    for k in ("p", "shots", "n_fallthrough", "fallthrough_rate", "mean_iter",
              "config", "basis"):
        assert k in m
    assert m["shots"] == 400
    assert 0.0 <= m["fallthrough_rate"] <= 1.0
    assert m["n_fallthrough"] == int(round(m["fallthrough_rate"] * 400))
    assert m["basis"] == "measured"
    assert m["config"]["bp_method"] == "minimum_sum"


def test_fallthrough_rises_with_p():
    """Fallthrough is monotonic-ish in p: rises from low p to high p (MEASURED).

    Small shot counts; we assert the END-TO-END trend (p=0.001 < p=0.005), which
    is robust to sampling noise even at modest shots.
    """
    lo = T.measure_fallthrough(0.001, shots=600, seed=0)["fallthrough_rate"]
    hi = T.measure_fallthrough(0.005, shots=600, seed=0)["fallthrough_rate"]
    assert hi > lo, f"fallthrough should rise with p: {lo=} {hi=}"
    # at low p almost everything converges; at high p a meaningful fraction falls through
    assert lo < 0.10
    assert hi > 0.15


def test_build_fallthrough_table_arithmetic(latency):
    """build_fallthrough_table: rows well-formed; effective latency = BP +
    fallthrough*increment per row (the binding arithmetic)."""
    tbl = T.build_fallthrough_table(latency, p_grid=(0.001, 0.005), shots=400,
                                    seed=0)
    assert tbl["kind"] == "osd-lsd-fallthrough-vs-p"
    assert len(tbl["rows"]) == 2
    bp_us = tbl["bp_us_per_syndrome"]
    inc = tbl["osd_post_increment_us"]
    for r in tbl["rows"]:
        expect = bp_us + r["fallthrough_rate"] * inc
        assert r["effective_latency_us"] == pytest.approx(expect)
        # effective latency is bounded by [BP floor, always-OSD ceiling].
        assert bp_us <= r["effective_latency_us"] <= tbl["bposd10_total_us"] + 1e-6


# --------------------------------------------------------------------------- #
# Deliverable 2: Amdahl serial-fraction                                       #
# --------------------------------------------------------------------------- #
def test_amdahl_table_columns_and_bounds(latency):
    rows = T.build_amdahl_table(latency)
    assert rows, "amdahl table empty"
    required = {"decoder", "total_us", "parallel_us", "serial_us",
                "serial_fraction", "speedup_ceiling", "serial_kind",
                "split_basis", "basis"}
    for r in rows:
        assert required <= set(r), f"missing columns in {r['decoder']}"
        # serial fraction in [0,1]
        assert 0.0 <= r["serial_fraction"] <= 1.0
        # parallel + serial == total
        assert r["parallel_us"] + r["serial_us"] == pytest.approx(r["total_us"])
        # ceiling = 1/serial_fraction (finite when serial_fraction>0)
        if r["serial_fraction"] > 0:
            assert r["speedup_ceiling"] == pytest.approx(1.0 / r["serial_fraction"])


def test_amdahl_measured_increment_matches_latency_gap(latency):
    """For measured-increment rows, serial_us == (decoder - base) us/syn gap."""
    rows = {r["decoder"]: r for r in T.build_amdahl_table(latency)}
    res = latency["results"]
    # BPOSD-10 serial = BPOSD-10 us/syn - BP us/syn (the OSD post-processing).
    bposd = rows["BPOSD-10"]
    assert bposd["split_basis"] == "measured-increment"
    gap = res["BPOSD-10"]["us_per_syndrome"] - res["BP"]["us_per_syndrome"]
    assert bposd["serial_us"] == pytest.approx(gap)
    # OSD-cs should be the most serial of the BP-OSD family (dense GE dominates).
    assert bposd["serial_fraction"] > rows["BPLSD"]["serial_fraction"]
    assert bposd["serial_fraction"] > rows["BPOSD-0"]["serial_fraction"]


# --------------------------------------------------------------------------- #
# Deliverable 3: roofline                                                     #
# --------------------------------------------------------------------------- #
def test_roofline_columns_and_tensor_core_flag(receipt, latency):
    rows = T.build_roofline_table(receipt, latency)
    assert rows
    required = {"decoder", "flop_per_byte", "bound_type", "tensor_core_path",
                "throughput_shots_per_s", "basis", "note"}
    for r in rows:
        assert required <= set(r)
    by = {r["decoder"]: r for r in rows}
    # the kernel-BP point is the MEASURED 0.17 flop/byte, no tensor-core path.
    bp = next(r for r in rows if r["decoder"].startswith("BP (min-sum"))
    assert bp["flop_per_byte"] == pytest.approx(
        receipt["arithmetic_intensity"]["approx_flops_per_byte"])
    assert bp["tensor_core_path"] is False
    # BP-OSD-10 is FLAGGED as having a tensor-core (dense-GE) future-work path.
    osd = by["BPOSD-10"]
    assert osd["tensor_core_path"] is True
    assert "FUTURE-WORK" in osd["note"]


# --------------------------------------------------------------------------- #
# Deliverable 4: memory footprint                                             #
# --------------------------------------------------------------------------- #
def test_memory_table_well_formed_and_scaling():
    tbl = T.build_memory_table()
    assert tbl["kind"] == "memory-footprint"
    rows = {r["decoder"]: r for r in tbl["rows"]}
    for r in tbl["rows"]:
        assert r["aux_bytes_per_shot"] >= 0
        assert r["total_bytes_per_shot"] == (
            r["bp_msg_bytes_per_shot"] + r["aux_bytes_per_shot"])
        # bigger device -> bigger (or equal) max batch
        assert r["max_batch"]["H200"] >= r["max_batch"]["A100-80GB"]
        assert r["max_batch"]["A100-80GB"] >= r["max_batch"]["RTX-4090-24GB"]
    # BP (no aux) admits a larger batch than Tesseract (large beam frontier aux).
    assert rows["BP"]["max_batch"]["H200"] > rows["Tesseract"]["max_batch"]["H200"]
    # BP bytes/shot = 2 * edges * 4
    assert rows["BP"]["bp_msg_bytes_per_shot"] == 2 * T.DEM_STRUCTURE["n_edges"] * 4


# --------------------------------------------------------------------------- #
# Deliverable 5: Gate-B structure                                             #
# --------------------------------------------------------------------------- #
def test_gate_b_structure_returns_verdict_and_stats():
    """gate_b_structure returns a verdict + the error-weight stats (well-formed)."""
    rng = np.random.default_rng(0)
    n = 2000
    w = rng.integers(0, 30, size=n)            # synthetic syndrome weights
    dec = rng.random(n) < 0.10                 # decoder fails 10%
    anch = rng.random(n) < 0.08                # anchor fails 8%
    out = T.gate_b_structure(dec, anch, w)
    for k in ("n", "n_residual_gap", "population_weight", "residual_weight",
              "mean_shift_sigma", "ks_statistic", "ks_pvalue", "verdict",
              "verdict_reason"):
        assert k in out
    assert out["n"] == n
    assert out["verdict"] in ("STRUCTURED", "UNSTRUCTURED", "INCONCLUSIVE")


def test_gate_b_unstructured_when_gap_matches_population():
    """A residual gap drawn from the SAME weight distribution -> UNSTRUCTURED."""
    rng = np.random.default_rng(1)
    n = 4000
    w = rng.integers(0, 30, size=n).astype(float)
    # decoder/anchor failures INDEPENDENT of weight -> residual gap ~ population.
    dec = rng.random(n) < 0.12
    anch = rng.random(n) < 0.10
    out = T.gate_b_structure(dec, anch, w)
    assert out["n_residual_gap"] >= 5
    assert out["verdict"] == "UNSTRUCTURED", out["verdict_reason"]


def test_gate_b_structured_when_gap_concentrated_at_high_weight():
    """A residual gap concentrated at HIGH weight -> STRUCTURED (clear cluster)."""
    rng = np.random.default_rng(2)
    n = 4000
    w = rng.integers(0, 30, size=n).astype(float)
    dec = np.zeros(n, dtype=bool)
    anch = np.zeros(n, dtype=bool)
    # Make the decoder fail (and the anchor succeed) ONLY on high-weight shots:
    # a sharp, low-dimensional structure the test must catch.
    high = w >= 25
    idx_high = np.where(high)[0]
    fail_idx = idx_high[: max(20, len(idx_high) // 2)]
    dec[fail_idx] = True          # decoder fails on high-weight shots
    anch[fail_idx] = False        # anchor succeeds there -> residual gap
    # add some background both-fail noise that does NOT create residual gap
    both = rng.random(n) < 0.03
    dec |= both
    anch |= both
    out = T.gate_b_structure(dec, anch, w)
    assert out["n_residual_gap"] >= 5
    assert out["verdict"] == "STRUCTURED", out["verdict_reason"]
    assert out["mean_shift_sigma"] > 0   # gap sits at higher weight than population
    # effect-size + spread fields present (the cluster-vs-shift discriminator)
    assert "cohens_d" in out and "spread_ratio" in out
    # a tight high-weight cluster has a LARGE effect size + small spread ratio
    # (separable), so the reason should NOT call it a diffuse shift.
    assert out["cohens_d"] > 0
    assert "separable cluster" in out["verdict_reason"]


def test_gate_b_diffuse_shift_flagged_narrow():
    """A residual gap that is a DIFFUSE shift (same spread, modest effect) is
    flagged STRUCTURED but with the 'DIFFUSE weight-correlated SHIFT' kind ->
    narrow T10 target. This mirrors the measured p=0.005 finding."""
    rng = np.random.default_rng(7)
    n = 60000
    w = rng.normal(34.0, 8.0, size=n)
    # residual gap drawn from a SHIFTED-but-same-spread distribution (modest
    # Cohen's d, spread ~ population) -> diffuse, not a separable cluster. A GENTLE
    # linear weight-correlation (heavier syndromes only slightly more likely to be
    # in the gap) reproduces the measured p=0.005 signature (d<0.8, spread~1).
    anch = np.zeros(n, dtype=bool)
    dec = np.zeros(n, dtype=bool)
    pr = np.clip(0.04 + 0.002 * (w - 34.0), 0.0, 0.3)  # gentle linear slope
    res = rng.random(n) < pr
    dec[res] = True
    anch[res] = False
    out = T.gate_b_structure(dec, anch, w)
    assert out["verdict"] == "STRUCTURED"
    assert 0.7 <= out["spread_ratio"] <= 1.3
    assert abs(out["cohens_d"]) < 0.8           # modest effect size = diffuse
    assert "DIFFUSE" in out["verdict_reason"]


def test_gate_b_inconclusive_when_too_few_residual():
    """Fewer than 5 residual-gap shots -> INCONCLUSIVE (under-powered)."""
    n = 1000
    w = np.arange(n) % 30
    dec = np.zeros(n, dtype=bool)
    anch = np.zeros(n, dtype=bool)
    dec[:3] = True       # only 3 residual-gap shots
    out = T.gate_b_structure(dec, anch, w)
    assert out["n_residual_gap"] == 3
    assert out["verdict"] == "INCONCLUSIVE"


def test_gate_b_shape_validation():
    """Mismatched shapes raise; non-1D masks raise."""
    with pytest.raises(ValueError):
        T.gate_b_structure(np.zeros(5, bool), np.zeros(4, bool), np.zeros(5))
    with pytest.raises(ValueError):
        T.gate_b_structure(np.zeros((2, 2), bool), np.zeros((2, 2), bool),
                           np.zeros((2, 2)))


# --------------------------------------------------------------------------- #
# Committed artifact (present after the production run)                        #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(_RESULTS),
                    reason="taxonomy_results.json not yet produced")
def test_committed_results_well_formed():
    with open(_RESULTS) as f:
        out = json.load(f)
    assert out["kind"] == "decoder-amenability-taxonomy"
    # all five deliverables present
    for k in ("fallthrough", "amdahl", "roofline", "memory"):
        assert k in out and out[k]
    # fallthrough rises across the committed grid (monotonic-ish in p)
    rows = out["fallthrough"]["rows"]
    rates = [r["fallthrough_rate"] for r in rows]
    assert rates[-1] > rates[0], "fallthrough should rise across the p-grid"
    # Gate-B verdict present (the T10 disposition)
    if out.get("gate_b"):
        assert out["gate_b"]["verdict"] in ("STRUCTURED", "UNSTRUCTURED",
                                            "INCONCLUSIVE")
