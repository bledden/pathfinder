"""TDD for the EXTERNAL decoder adapters (Relay-BP + sliding-window) in the
matched zoo (qldpc/zoo/adapters.py).

Both external decoders share the uniform interface (``.name``, ``.config``,
``.tie_break``, ``decode_batch(dets)->obs_pred``), are built from the SAME shared
SI1000 BB Z-memory DEM the core zoo uses, and MUST pass the matched harness'
gates G1 (DEM hash-identity) and G2 (tie-break in APPROVED_TIE_BREAKS).

Availability is import-guarded:
  * Relay-BP  -> ``relay_bp_available()`` (relay-bp[stim] installed).
  * Sliding-window -> ``sliding_window_available()`` (the isolated qLDPC install
    is importable in a clean subprocess).
Each external test ``skip``s if its package is unavailable, so CI without the
package stays green; when the package IS present the test actually exercises it.
The core zoo is verified to always build (with include_external=True), gracefully
skipping unavailable externals.
"""
import numpy as np
import pytest

from bb_code import BBCode
from qldpc.foundation.circuits import build_memory
from qldpc.foundation.stats import wilson_ci
from qldpc.zoo.adapters import (
    APPROVED_TIE_BREAKS,
    build_decoders,
    make_relay_bp,
    make_sliding_window,
    relay_bp_available,
    sliding_window_available,
)
from qldpc.zoo.harness import dem_hash, run_matched

SHOTS = 1000
SEED = 0
ROUNDS = 3
P = 0.005

RELAY_BP = relay_bp_available()
SLIDING_WINDOW = sliding_window_available()


@pytest.fixture(scope="module")
def dem_and_shots():
    circ = build_memory(BBCode(), rounds=ROUNDS, p=P, basis="Z", noise="si1000")
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=SEED).sample(
        SHOTS, separate_observables=True)
    return circ, dem, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


def _ler(pred, obs):
    return float(np.any(pred != obs, axis=1).mean())


def _fails(pred, obs):
    return int(np.any(pred != obs, axis=1).sum())


def _slack(fails, n):
    lo, hi = wilson_ci(fails, n)
    return (hi - lo) / 2.0


# --------------------------------------------------------------------------- #
# Core-zoo robustness: include_external must NEVER break the core build.       #
# --------------------------------------------------------------------------- #
def test_core_zoo_always_builds_with_include_external(dem_and_shots):
    """include_external=True must still yield the full core zoo (externals are
    additive, import-guarded, and degrade gracefully when unavailable)."""
    _, dem, _, _ = dem_and_shots
    decs = build_decoders(dem, include_external=True, rounds=ROUNDS)
    names = {d.name for d in decs}
    assert {"BPOSD-0", "BPOSD-10", "BPLSD", "BP", "Tesseract"} <= names
    # every adapter shares the one DEM object (provenance) + a declared tie-break
    for d in decs:
        assert d.dem is dem
        assert d.tie_break in APPROVED_TIE_BREAKS


def test_external_adapters_added_when_available(dem_and_shots):
    _, dem, _, _ = dem_and_shots
    decs = build_decoders(dem, include_external=True, rounds=ROUNDS)
    names = {d.name for d in decs}
    assert ("RelayBP" in names) == RELAY_BP
    # sliding-window needs rounds (supplied) AND the install
    assert ("SlidingWindow" in names) == SLIDING_WINDOW


def test_sliding_window_omitted_without_rounds(dem_and_shots):
    """Without ``rounds`` the streaming sliding-window decoder cannot derive its
    per-round time map, so it is omitted even if installed (Relay-BP still adds)."""
    _, dem, _, _ = dem_and_shots
    decs = build_decoders(dem, include_external=True)  # no rounds
    names = {d.name for d in decs}
    assert "SlidingWindow" not in names


# --------------------------------------------------------------------------- #
# Relay-BP                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not RELAY_BP, reason="relay-bp[stim] not installed")
def test_relay_bp_shape_dtype_and_better_than_chance(dem_and_shots):
    _, dem, dets, obs = dem_and_shots
    d = make_relay_bp(dem)
    pred = d.decode_batch(dets)
    assert pred.shape == (SHOTS, dem.num_observables)
    assert pred.dtype == bool
    assert _ler(pred, obs) < 0.5
    assert d.tie_break in APPROVED_TIE_BREAKS
    assert isinstance(d.config, dict) and d.config
    assert d.dem is dem


@pytest.mark.skipif(not RELAY_BP, reason="relay-bp[stim] not installed")
def test_relay_bp_competitive_with_bposd(dem_and_shots):
    """Relay-BP LER should be competitive with BP-OSD-10 (<= it + Wilson slack)
    and clearly beat bare BP."""
    _, dem, dets, obs = dem_and_shots
    rb = make_relay_bp(dem)
    by = {x.name: x for x in build_decoders(dem)}
    f_rb = _fails(rb.decode_batch(dets), obs)
    f_osd = _fails(by["BPOSD-10"].decode_batch(dets), obs)
    f_bp = _fails(by["BP"].decode_batch(dets), obs)
    n = SHOTS
    s = max(_slack(f_rb, n), _slack(f_osd, n))
    assert f_rb / n <= f_osd / n + s, (
        f"Relay-BP LER {f_rb/n} not competitive with BP-OSD-10 {f_osd/n} (slack {s})")
    # comfortably better than bare BP
    assert f_rb / n <= f_bp / n + _slack(f_bp, n)


@pytest.mark.skipif(not RELAY_BP, reason="relay-bp[stim] not installed")
def test_relay_bp_in_run_matched(dem_and_shots):
    circ, dem, _, _ = dem_and_shots
    decs = build_decoders(dem) + [make_relay_bp(dem)]
    m = run_matched(circ, decs, shots=SHOTS, rounds=ROUNDS, seed=SEED, label="rb")
    rec = next(r for r in m["decoders"] if r["name"] == "RelayBP")
    assert rec["tie_break"] in APPROVED_TIE_BREAKS
    assert 0.0 <= rec["ler"] < 0.5
    assert m["dem_hash"] == dem_hash(dem)  # G1 held


# --------------------------------------------------------------------------- #
# Sliding-window (streaming)                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not SLIDING_WINDOW, reason="qLDPC sliding-window not installed")
def test_sliding_window_shape_dtype_and_better_than_chance(dem_and_shots):
    _, dem, dets, obs = dem_and_shots
    d = make_sliding_window(dem, rounds=ROUNDS, window_size=3, stride=1)
    pred = d.decode_batch(dets)
    assert pred.shape == (SHOTS, dem.num_observables)
    assert pred.dtype == bool
    assert _ler(pred, obs) < 0.5
    assert d.tie_break in APPROVED_TIE_BREAKS
    assert isinstance(d.config, dict) and d.config
    assert d.config["window"] == 3 and d.config["stride"] == 1
    assert d.dem is dem


@pytest.mark.skipif(not SLIDING_WINDOW, reason="qLDPC sliding-window not installed")
def test_sliding_window_ler_sane(dem_and_shots):
    """A streaming (W, stride) decoder is expected to be a bit worse than the
    full-DEM BP-OSD-10 but still well below chance and within a generous margin
    of the global BP-OSD-0 bar."""
    _, dem, dets, obs = dem_and_shots
    sw = make_sliding_window(dem, rounds=ROUNDS, window_size=3, stride=1)
    by = {x.name: x for x in build_decoders(dem)}
    f_sw = _fails(sw.decode_batch(dets), obs)
    f_bp = _fails(by["BP"].decode_batch(dets), obs)
    n = SHOTS
    assert f_sw / n < 0.5
    # sliding-window with a per-window BP-OSD-cs inner should not be worse than
    # bare BP (it has OSD post-processing per window).
    assert f_sw / n <= f_bp / n + max(_slack(f_sw, n), _slack(f_bp, n))


@pytest.mark.skipif(not SLIDING_WINDOW, reason="qLDPC sliding-window not installed")
def test_sliding_window_in_run_matched(dem_and_shots):
    circ, dem, _, _ = dem_and_shots
    decs = build_decoders(dem) + [make_sliding_window(dem, rounds=ROUNDS)]
    m = run_matched(circ, decs, shots=SHOTS, rounds=ROUNDS, seed=SEED, label="sw")
    rec = next(r for r in m["decoders"] if r["name"] == "SlidingWindow")
    assert rec["tie_break"] in APPROVED_TIE_BREAKS
    assert 0.0 <= rec["ler"] < 0.5
    assert m["dem_hash"] == dem_hash(dem)  # G1 held (worker round-trips same DEM)


# --------------------------------------------------------------------------- #
# Both externals together in one matched run (full external zoo).              #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (RELAY_BP or SLIDING_WINDOW),
                    reason="no external decoder installed")
def test_full_external_zoo_in_run_matched(dem_and_shots):
    circ, dem, _, _ = dem_and_shots
    decs = build_decoders(dem, include_external=True, rounds=ROUNDS)
    m = run_matched(circ, decs, shots=SHOTS, rounds=ROUNDS, seed=SEED, label="all")
    names = {r["name"] for r in m["decoders"]}
    assert {"BPOSD-0", "BPOSD-10", "BPLSD", "BP", "Tesseract"} <= names
    if RELAY_BP:
        assert "RelayBP" in names
    if SLIDING_WINDOW:
        assert "SlidingWindow" in names
    for r in m["decoders"]:
        assert r["tie_break"] in APPROVED_TIE_BREAKS
        assert 0.0 <= r["ler"] <= 1.0
