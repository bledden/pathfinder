"""TDD for the matched-protocol harness + three pre-committed binding gates
(qldpc/zoo/harness.py).

The harness wires the matched cross-decoder runner: every decoder must consume
the byte-IDENTICAL DEM (silent DEM drift across an external-install zoo is the #1
risk). Three gates are pre-committed before the real LER grid:

  * G1 (DEM identity): dem_hash(dem) = sha256 of the DEM's canonical bytes; the
    runner asserts every decoder was built from a DEM hashing to the shared DEM
    (ldpc adapters via .dem, Tesseract via `.dem is dem`). Mismatch -> raise.
  * G2 (tie-break policy): every adapter declares a deterministic .tie_break in
    APPROVED_TIE_BREAKS; the runner asserts it (missing/unknown -> raise).
  * G3 (smoke grid): a fast 1k-shot single-p single-d matched run across the full
    zoo, the per-push smoke check (DEM-drift / adapter regression / env breakage).
"""
import numpy as np
import pytest

from bb_code import BBCode
from qldpc.foundation.circuits import build_memory
from qldpc.foundation.stats import wilson_ci
from qldpc.zoo.adapters import APPROVED_TIE_BREAKS, build_decoders
from qldpc.zoo.harness import dem_hash, run_matched, smoke_grid

ROUNDS = 2
P = 0.005
SHOTS = 1000
SEED = 0


def _build_circuit(p=P, rounds=ROUNDS):
    return build_memory(BBCode(), rounds=rounds, p=p, basis="Z", noise="si1000")


@pytest.fixture(scope="module")
def circuit():
    return _build_circuit()


@pytest.fixture(scope="module")
def dem(circuit):
    return circuit.detector_error_model(decompose_errors=False)


@pytest.fixture(scope="module")
def decoders(dem):
    return build_decoders(dem)


@pytest.fixture(scope="module")
def manifest(circuit, decoders):
    return run_matched(circuit, decoders, shots=SHOTS, rounds=ROUNDS,
                       seed=SEED, label="test")


# --- G1: dem_hash -----------------------------------------------------------
def test_dem_hash_stable_and_sensitive():
    """Same (code, rounds, p) -> identical hash; different p -> different hash."""
    d_a = _build_circuit(p=P).detector_error_model(decompose_errors=False)
    d_a2 = _build_circuit(p=P).detector_error_model(decompose_errors=False)
    d_b = _build_circuit(p=P + 0.001).detector_error_model(decompose_errors=False)
    h_a, h_a2, h_b = dem_hash(d_a), dem_hash(d_a2), dem_hash(d_b)
    assert isinstance(h_a, str) and len(h_a) == 64
    assert all(ch in "0123456789abcdef" for ch in h_a)
    assert h_a == h_a2, "dem_hash not stable across rebuilds of the same DEM"
    assert h_a != h_b, "dem_hash not sensitive to a change in p"


# --- the manifest -----------------------------------------------------------
def test_run_matched_manifest(manifest):
    m = manifest
    # top-level schema
    assert isinstance(m["dem_hash"], str) and len(m["dem_hash"]) == 64
    assert all(ch in "0123456789abcdef" for ch in m["dem_hash"])
    assert m["shots"] == SHOTS
    assert m["seed"] == SEED
    assert m["rounds"] == ROUNDS
    assert m["label"] == "test"
    assert m["n_det"] == 108
    assert m["n_obs"] == 12

    names = {r["name"] for r in m["decoders"]}
    assert names == {"BPOSD-0", "BPOSD-10", "BPLSD", "BP", "Tesseract"}

    for r in m["decoders"]:
        assert isinstance(r["config"], dict) and r["config"]
        assert r["tie_break"] in APPROVED_TIE_BREAKS
        assert 0.0 <= r["ler"] <= 1.0
        lo, hi = r["ler_ci"]
        assert 0.0 <= lo <= r["ler"] <= hi <= 1.0
        assert 0 <= r["fails"] <= SHOTS
        assert r["decode_s"] >= 0.0
        # per-round lambda = 1 - (1-LER)^(1/rounds)
        assert 0.0 <= r["lambda_per_round"] <= 1.0
        exp_lam = 1.0 - (1.0 - r["ler"]) ** (1.0 / ROUNDS)
        assert abs(r["lambda_per_round"] - exp_lam) < 1e-12

    # ordering sane (matched shots), with Wilson CI slack
    by = {r["name"]: r for r in m["decoders"]}

    def slack(name):
        lo, hi = by[name]["ler_ci"]
        return (hi - lo) / 2.0

    s1 = max(slack("BPOSD-10"), slack("BP"))
    assert by["BPOSD-10"]["ler"] <= by["BP"]["ler"] + s1
    s2 = max(slack("Tesseract"), slack("BPOSD-10"))
    assert by["Tesseract"]["ler"] <= by["BPOSD-10"]["ler"] + s2


def test_manifest_json_serializable(manifest):
    import json
    s = json.dumps(manifest)
    assert isinstance(s, str) and len(s) > 0


def test_manifest_dem_hash_matches(manifest, dem):
    assert manifest["dem_hash"] == dem_hash(dem)


# --- G1 fail-fast -----------------------------------------------------------
def test_g1_dem_identity_failfast():
    """Decoders built from DEM_A, run on a circuit whose DEM differs (different p)
    -> the G1 DEM-identity guard must raise."""
    circ_a = _build_circuit(p=P)
    dem_a = circ_a.detector_error_model(decompose_errors=False)
    decs = build_decoders(dem_a)
    circ_b = _build_circuit(p=P + 0.002)  # different DEM
    assert dem_hash(circ_b.detector_error_model(decompose_errors=False)) != dem_hash(dem_a)
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        run_matched(circ_b, decs, shots=100, rounds=ROUNDS, seed=SEED)


# --- G2 fail-fast -----------------------------------------------------------
class _BogusTieBreakDecoder:
    """Stub adapter that shares the right DEM but declares an unapproved tie-break."""

    def __init__(self, dem, tie_break):
        self.dem = dem
        self.name = "Bogus"
        self.config = {"decoder": "Bogus"}
        self.tie_break = tie_break
        self._n_obs = dem.num_observables

    def decode_batch(self, dets):
        dets = np.asarray(dets, dtype=bool)
        return np.zeros((dets.shape[0], self._n_obs), dtype=bool)


def test_g2_tiebreak_failfast_unknown(circuit, dem):
    bad = _BogusTieBreakDecoder(dem, tie_break="bogus")
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        run_matched(circuit, [bad], shots=100, rounds=ROUNDS, seed=SEED)


def test_g2_tiebreak_failfast_missing(circuit, dem):
    bad = _BogusTieBreakDecoder(dem, tie_break=None)
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        run_matched(circuit, [bad], shots=100, rounds=ROUNDS, seed=SEED)


# --- lambda math ------------------------------------------------------------
def test_lambda_math():
    """lambda_per_round = 1 - (1-LER)^(1/R) on a known LER/R."""
    from qldpc.zoo.harness import _lambda_per_round
    # LER = 0.19, R = 2 -> 1 - sqrt(0.81) = 1 - 0.9 = 0.1
    assert abs(_lambda_per_round(0.19, 2) - 0.1) < 1e-12
    # LER = 0, any R -> 0
    assert _lambda_per_round(0.0, 5) == 0.0
    # general check at R=4
    ler, R = 0.3, 4
    assert abs(_lambda_per_round(ler, R) - (1 - (1 - ler) ** (1.0 / R))) < 1e-12


# --- G3: smoke grid ---------------------------------------------------------
def test_smoke_grid():
    """smoke_grid(...) returns a manifest with all decoders + a dem_hash (fast)."""
    m = smoke_grid(
        code_factory=lambda: BBCode(),
        decoders_factory=build_decoders,
        p=P,
        rounds=ROUNDS,
        shots=SHOTS,
        basis="Z",
        seed=SEED,
    )
    assert isinstance(m["dem_hash"], str) and len(m["dem_hash"]) == 64
    assert m["shots"] == SHOTS
    names = {r["name"] for r in m["decoders"]}
    assert names == {"BPOSD-0", "BPOSD-10", "BPLSD", "BP", "Tesseract"}
    for r in m["decoders"]:
        assert r["tie_break"] in APPROVED_TIE_BREAKS
