"""Matched-protocol cross-decoder runner + three pre-committed binding gates.

This is the apples-to-apples LER runner: ONE shared DEM, ONE sampled shot set,
every decoder decoding the BYTE-IDENTICAL DEM. Silent DEM drift across an
external-install zoo is the #1 risk, so three gates are checked fail-fast BEFORE
any LER is reported:

  * **G1 (DEM identity).** ``dem_hash(dem)`` is the sha256 of the DEM's canonical
    bytes (``str(dem.flattened()).encode()``). It is stable across rebuilds of the
    same (code, rounds, p, basis, noise) and sensitive to any change. ``run_matched``
    rebuilds the shared DEM from the circuit, hashes it, and asserts every ldpc
    adapter's ``.dem`` hashes to the SAME value and Tesseract's ``.dem is dem``.
    A decoder built from a different DEM raises immediately.
  * **G2 (tie-break policy).** Every adapter declares ``.tie_break`` (a short
    deterministic-tie-break string); ``run_matched`` asserts it is in
    ``adapters.APPROVED_TIE_BREAKS`` (missing / unknown -> raise).
  * **G3 (smoke grid).** ``smoke_grid`` is a fast 1k-shot single-p single-d matched
    run across the full zoo (the per-push smoke check that catches DEM-drift /
    adapter regression / env breakage before the CPU-week grid).

The manifest produced by ``run_matched`` is a JSON-serializable dict with the
DEM provenance (hash, sizes), the sampling parameters, an optional git HEAD, and
one record per decoder (name, config, tie_break, fails, ler, ler_ci,
lambda_per_round, decode_s).
"""
import hashlib
import os
import subprocess
import sys
import time

import numpy as np

# conftest.py puts repo root + qldpc/ on sys.path; ensure qldpc/ is importable
# when this module is imported directly (e.g. a per-push smoke runner).
_QLDPC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_QLDPC)
for _p in (_ROOT, _QLDPC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# foundation.stats: prefer the qldpc-qualified import (unambiguous — a bare
# ``from foundation import stats`` collides with tests/foundation when the full
# suite is collected), fall back to flat (qldpc/ on path, run directly).
try:
    from qldpc.foundation import stats
except ImportError:
    from foundation import stats

# Sibling adapters module: importable as a submodule of THIS package when loaded
# as ``qldpc.zoo.harness`` (relative), else flat via ``qldpc/zoo`` on sys.path.
try:
    from .adapters import APPROVED_TIE_BREAKS, build_decoders
except ImportError:  # loaded flat (qldpc/ on path, not as the qldpc.zoo package)
    from adapters import APPROVED_TIE_BREAKS, build_decoders


def dem_hash(dem):
    """Gate G1 primitive: sha256 hex of the DEM's canonical bytes.

    Uses ``str(dem.flattened()).encode()`` — the flattened DEM's text form is the
    canonical mechanism listing (detectors, observables, prior per mechanism in
    order). Stable across rebuilds of the same (code, rounds, p, basis, noise);
    sensitive to any change in the mechanism set or priors.
    """
    return hashlib.sha256(str(dem.flattened()).encode()).hexdigest()


def _lambda_per_round(ler, rounds):
    """Per-round logical error rate lambda = 1 - (1 - LER)^(1/rounds).

    Inverts LER = 1 - (1 - lambda)^rounds for a memory of ``rounds`` rounds.
    """
    if rounds <= 0:
        raise ValueError(f"rounds must be positive, got {rounds!r}")
    return 1.0 - (1.0 - ler) ** (1.0 / rounds)


def _git_head():
    """Best-effort current git HEAD (short SHA); None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _decoder_dem_hash(decoder):
    """The hash of the DEM a decoder was built from. Every adapter (ldpc + the
    Tesseract anchor) carries ``.dem`` (the shared DEM object it was built from);
    raise loudly if a decoder does not expose it (it cannot be provenance-checked
    and must NOT be silently trusted)."""
    dem = getattr(decoder, "dem", None)
    if dem is None:
        raise ValueError(
            f"G1: decoder {getattr(decoder, 'name', decoder)!r} exposes no .dem; "
            f"cannot verify DEM identity (build it via build_decoders(dem))."
        )
    return dem_hash(dem)


def run_matched(circuit, decoders, shots, rounds, seed=0, label=None):
    """Matched cross-decoder run on ONE shared DEM and ONE sampled shot set.

    Args:
        circuit: stim.Circuit (the memory experiment) — the single source of the
            shared DEM and the sampled detectors/observables.
        decoders: list of adapters (built via ``build_decoders(dem)`` on THIS
            circuit's DEM). Each must carry ``.dem`` (provenance) and a declared
            ``.tie_break``.
        shots: number of shots to sample and decode (all decoders see the SAME
            shots).
        rounds: number of syndrome rounds (for the per-round lambda).
        seed: detector-sampler seed (fixed-seed reproducibility).
        label: optional free-text label recorded in the manifest.

    Gates (fail-fast, BEFORE any decoding):
        * G1: rebuild the shared DEM from ``circuit``, hash it, assert every
          ldpc decoder's ``.dem`` hashes to the same value and Tesseract's
          ``.dem is dem``. Mismatch -> ValueError.
        * G2: assert every decoder's ``.tie_break`` is in APPROVED_TIE_BREAKS.

    Returns: a JSON-serializable manifest dict.
    """
    if not decoders:
        raise ValueError("run_matched: no decoders supplied")

    # --- canonical hash of THIS circuit's DEM (the drift reference) -------------
    # A FRESH DEM object every call (stim never returns the same object), so the
    # G1 drift check is by canonical HASH, not by object identity.
    circuit_h = dem_hash(circuit.detector_error_model(decompose_errors=False))

    # The matched protocol requires ONE shared DEM object behind ALL decoders
    # (build_decoders(dem) builds every adapter from a single dem). Take the
    # decoders' shared DEM as `dem` and require object-identity across them.
    dem = getattr(decoders[0], "dem", None)
    if dem is None:
        raise ValueError(
            f"G1: decoder {getattr(decoders[0], 'name', decoders[0])!r} exposes "
            f"no .dem; build decoders via build_decoders(dem)."
        )
    h = dem_hash(dem)

    # --- G1: every decoder consumes the byte-identical, SHARED DEM -------------
    if h != circuit_h:
        raise ValueError(
            f"G1 DEM-identity FAIL: the decoders were built from a DEM hashing to "
            f"{h}, but THIS circuit's DEM hashes to {circuit_h}. Silent DEM drift "
            f"(e.g. a different p/rounds/basis/noise) is forbidden — rebuild ALL "
            f"decoders via build_decoders(dem) on THIS circuit's DEM "
            f"(decompose_errors=False)."
        )
    for d in decoders:
        name = getattr(d, "name", repr(d))
        dh = _decoder_dem_hash(d)
        if dh != h:
            raise ValueError(
                f"G1 DEM-identity FAIL: decoder {name!r} was built from a DEM "
                f"hashing to {dh}, not the shared DEM ({h}). Every decoder must "
                f"consume the byte-identical DEM; rebuild via build_decoders(dem)."
            )
        # Tesseract ingests the DEM object directly — require it be the SAME shared
        # object, so a same-text but distinct DEM object cannot sneak past.
        if name == "Tesseract" and getattr(d, "dem", None) is not dem:
            raise ValueError(
                f"G1 DEM-identity FAIL: Tesseract's .dem is not the shared DEM "
                f"object the other decoders were built from (it ingests the DEM "
                f"directly and must be the same object). Rebuild via "
                f"build_decoders(dem) on ONE shared dem."
            )

    # --- G2: every decoder must declare an approved deterministic tie-break -----
    for d in decoders:
        name = getattr(d, "name", repr(d))
        tb = getattr(d, "tie_break", None)
        if tb not in APPROVED_TIE_BREAKS:
            raise ValueError(
                f"G2 tie-break FAIL: decoder {name!r} declares tie_break={tb!r}, "
                f"not in APPROVED_TIE_BREAKS={sorted(APPROVED_TIE_BREAKS)}. "
                f"Every adapter must pre-commit a verified deterministic tie-break."
            )

    # --- sample ONCE; all decoders see the SAME shots --------------------------
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    dets = np.asarray(dets, dtype=bool)
    obs = np.asarray(obs, dtype=bool)

    records = []
    for d in decoders:
        t0 = time.perf_counter()
        pred = d.decode_batch(dets)
        decode_s = time.perf_counter() - t0
        pred = np.asarray(pred, dtype=bool)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        fails = int(np.any(pred != obs, axis=1).sum())
        ler = fails / shots if shots else 0.0
        lo, hi = stats.wilson_ci(fails, shots)
        records.append({
            "name": d.name,
            "config": dict(d.config),
            "tie_break": d.tie_break,
            "fails": fails,
            "ler": ler,                       # block-LER (per-shot logical failure)
            "ler_ci": [lo, hi],
            "lambda_per_round": _lambda_per_round(ler, rounds),
            "decode_s": decode_s,
        })

    return {
        "dem_hash": h,
        "n_det": dem.num_detectors,
        "n_obs": dem.num_observables,
        "shots": shots,
        "seed": seed,
        "rounds": rounds,
        "label": label,
        "git_head": _git_head(),
        "decoders": records,
    }


def smoke_grid(code_factory, decoders_factory=build_decoders, p=0.005, rounds=2,
               shots=1000, basis="Z", seed=0):
    """Gate G3: a fast 1k-shot single-p single-d matched run across the full zoo.

    The per-push smoke check: build the circuit + shared DEM + the full decoder
    zoo, run the matched protocol, and return the manifest. Catches DEM-drift,
    adapter regression, and env breakage before committing the CPU-week grid.

    Args:
        code_factory: zero-arg callable returning a code instance (e.g.
            ``lambda: BBCode()``).
        decoders_factory: callable ``dem -> [adapters]`` (default: the zoo's
            ``build_decoders``).
        p, rounds, shots, basis, seed: matched-run parameters (kept small/fast).
    """
    try:
        from qldpc.foundation.circuits import build_memory
    except ImportError:
        from foundation.circuits import build_memory

    code = code_factory()
    circuit = build_memory(code, rounds=rounds, p=p, basis=basis, noise="si1000")
    dem = circuit.detector_error_model(decompose_errors=False)
    decoders = decoders_factory(dem)
    return run_matched(circuit, decoders, shots=shots, rounds=rounds, seed=seed,
                       label=f"smoke:{type(code).__name__} p={p} R={rounds} basis={basis}")
