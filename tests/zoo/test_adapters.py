"""TDD for the uniform decoder interface + classical adapters (qldpc/zoo/adapters.py).

A SMALL SI1000 BB Z-memory DEM is shared by ALL adapters (matched protocol: every
ldpc-family decoder derives its H/Lo/priors from the SAME DEM object via
canon_dem.extract; Tesseract ingests the same DEM directly). One fixed sampled
shot set is decoded by every adapter so the LER ordering is apples-to-apples.

Checks:
  * shape (shots, n_obs) + bool dtype from every decode_batch,
  * every adapter LER < chance (< ~0.5),
  * ordering (same shots): Tesseract-MLE <= BPOSD-10 + slack, BPOSD-10 <= BP + slack,
  * all adapters constructed from the SAME dem object (provenance guard).
"""
import numpy as np
import pytest

from bb_code import BBCode
from qldpc.foundation.circuits import build_memory
from qldpc.foundation.stats import wilson_ci
from qldpc.zoo.adapters import build_decoders

SHOTS = 1500
SEED = 0
ROUNDS = 2
P = 0.005


@pytest.fixture(scope="module")
def dem_and_shots():
    circ = build_memory(BBCode(), rounds=ROUNDS, p=P, basis="Z", noise="si1000")
    dem = circ.detector_error_model(decompose_errors=False)
    sampler = circ.compile_detector_sampler(seed=SEED)
    dets, obs = sampler.sample(SHOTS, separate_observables=True)
    return dem, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


@pytest.fixture(scope="module")
def decoders(dem_and_shots):
    dem, _, _ = dem_and_shots
    return build_decoders(dem)


def _ler(pred, obs):
    return float(np.any(pred != obs, axis=1).mean())


def _fails(pred, obs):
    return int(np.any(pred != obs, axis=1).sum())


def test_all_decoders_built(decoders):
    names = {d.name for d in decoders}
    assert names == {"BPOSD-0", "BPOSD-10", "BPLSD", "BP", "Tesseract"}


def test_each_adapter_shape_and_dtype(decoders, dem_and_shots):
    dem, dets, obs = dem_and_shots
    n_obs = dem.num_observables
    for d in decoders:
        pred = d.decode_batch(dets)
        assert pred.shape == (SHOTS, n_obs), f"{d.name} returned {pred.shape}"
        assert pred.dtype == bool, f"{d.name} dtype {pred.dtype}"


def test_each_adapter_ler_better_than_chance(decoders, dem_and_shots):
    _, dets, obs = dem_and_shots
    for d in decoders:
        pred = d.decode_batch(dets)
        ler = _ler(pred, obs)
        assert ler < 0.5, f"{d.name} LER {ler} not < chance"


def test_each_adapter_has_name_and_config(decoders):
    for d in decoders:
        assert isinstance(d.name, str) and d.name
        assert isinstance(d.config, dict) and d.config


def test_all_adapters_share_one_dem(decoders, dem_and_shots):
    """Provenance: every adapter must report the SAME dem object it was built from."""
    dem, _, _ = dem_and_shots
    for d in decoders:
        assert d.dem is dem, f"{d.name} not built from the shared dem object"


def test_ordering_tesseract_le_bposd10_le_bp(decoders, dem_and_shots):
    """Matched-protocol ordering on the SAME shots, with Wilson CI slack so the
    qualitative ordering (OSD helps vs bare BP; MLE <= OSD) is robust at 1500 shots."""
    _, dets, obs = dem_and_shots
    by_name = {d.name: d for d in decoders}
    preds = {name: by_name[name].decode_batch(dets) for name in
             ("BP", "BPOSD-10", "Tesseract")}
    fails = {name: _fails(preds[name], obs) for name in preds}
    n = SHOTS

    # Wilson slack = half-width of the wider of the two CIs being compared.
    def slack(name):
        lo, hi = wilson_ci(fails[name], n)
        return (hi - lo) / 2.0

    ler = {name: fails[name] / n for name in fails}

    # BPOSD-10 <= BP + slack  (OSD post-processing helps vs bare BP)
    s = max(slack("BPOSD-10"), slack("BP"))
    assert ler["BPOSD-10"] <= ler["BP"] + s, (
        f"BPOSD-10 LER {ler['BPOSD-10']} > BP LER {ler['BP']} + slack {s}")

    # Tesseract-MLE <= BPOSD-10 + slack  (near-optimal MLE anchor)
    s2 = max(slack("Tesseract"), slack("BPOSD-10"))
    assert ler["Tesseract"] <= ler["BPOSD-10"] + s2, (
        f"Tesseract LER {ler['Tesseract']} > BPOSD-10 LER {ler['BPOSD-10']} + slack {s2}")
