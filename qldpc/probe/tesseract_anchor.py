"""Tesseract decoder as a near-exact most-likely-ERROR (MLE) anchor, plus a
beam-convergence protocol.

Tesseract (``quantumlib/tesseract-decoder``, Apache-2.0) is an A* / beam search
over a Stim ``DetectorErrorModel``. With a small ``beam`` (``det_beam``) it is a
fast heuristic; as the beam grows it approaches the **single most-likely error
configuration** consistent with each syndrome and reports the observables that
error flips.

HONEST LABELING
---------------
This is *most-likely-ERROR* (MLE): it finds the lowest-cost single error vector
``e`` (cost = sum of ``log((1-p)/p)`` over the chosen mechanisms) and returns
``L e``. It is **near-exact at large beam** (the search is exhaustive enough to
find the true minimum-weight error), but it is **NOT coset-ML** decoding: it
does not marginalize over the whole coset of errors producing the same logical
class. For the true coset maximum-likelihood (coset-ML) class probability, use
``qldpc.foundation.tn_mld.coset_ml_ler`` instead. Tesseract is the MLE anchor;
the TN contractor is the coset-ML anchor.

DISCOVERED API (tesseract-decoder 0.1.x, pybind11 .so)
------------------------------------------------------
* ``tesseract_decoder.tesseract.TesseractConfig(dem=<stim.DetectorErrorModel>,
  det_beam=<int>, ...)`` -- the beam knob is ``det_beam``.
* ``config.compile_decoder() -> TesseractDecoder``.
* ``decoder.decode_batch(syndromes: np.ndarray[bool] shape (shots, n_det))
  -> np.ndarray[bool] shape (shots, n_obs)`` -- the predicted observables.

Detection events / observables are sampled directly from the Stim circuit via
``circuit.compile_detector_sampler(seed=...).sample(shots,
separate_observables=True)``, which returns bool arrays matching the decoder's
expected ``(shots, n_det)`` / ``(shots, n_obs)`` shapes.
"""

import numpy as np
import stim

import tesseract_decoder as _td

from qldpc.foundation.stats import tost_equivalent


def _build_decoder(dem, beam):
    """Construct a Tesseract decoder for ``dem`` at the given beam (det_beam)."""
    cfg = _td.tesseract.TesseractConfig(dem=dem, det_beam=int(beam))
    return cfg.compile_decoder()


def _sample(circuit, shots, seed):
    """Sample ``shots`` (detection events, observables) from a Stim circuit.

    Returns ``(dets, obs)`` as bool arrays of shape ``(shots, n_det)`` and
    ``(shots, n_obs)``.
    """
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    return np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


def _ler_from_samples(dem, dets, obs, beam):
    """Decode pre-sampled ``dets`` with Tesseract at ``beam``; return (ler, fails)."""
    n_shots = len(dets)
    if n_shots == 0:
        return 0.0, 0
    decoder = _build_decoder(dem, beam)
    pred = np.asarray(decoder.decode_batch(dets), dtype=bool)
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    assert pred.shape == obs.shape, (
        f"Tesseract prediction shape {pred.shape} != observable shape {obs.shape} "
        f"(decoder returned a wrong n_obs column count)"
    )
    fails = int(np.any(pred != obs, axis=1).sum())
    return fails / n_shots, fails


def tesseract_ler(dem, circuit, shots, beam, seed=0):
    """Logical error rate of the Tesseract MLE anchor at a fixed beam.

    Samples ``shots`` (detection events, observables) from ``circuit``, decodes
    them with Tesseract at ``det_beam = beam`` (built from ``dem``), and returns
    the fraction of shots whose predicted observables differ from the actual
    ones.

    This is the most-likely-ERROR rate -- near-exact at large ``beam`` -- not
    coset-ML. See the module docstring.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model the decoder is built against (typically
        ``circuit.detector_error_model(decompose_errors=False)``).
    circuit : stim.Circuit
        The circuit to sample detection events / observables from. ``dem`` and
        ``circuit`` must agree on detector / observable counts.
    shots : int
        Number of Monte-Carlo shots.
    beam : int
        Tesseract ``det_beam`` cutoff. Larger -> closer to exact MLE.
    seed : int, default=0
        Seed for the detector sampler.

    Returns
    -------
    float
        Logical error rate in ``[0, 1]``.
    """
    dets, obs = _sample(circuit, shots, seed)
    ler, _ = _ler_from_samples(dem, dets, obs, beam)
    return ler


def beam_convergence(circuit, shots=20000, ladder=(8, 16, 32, 64),
                     margin_rel=0.10, seed=0, dem=None):
    """Beam-convergence protocol for the Tesseract MLE anchor.

    Decodes the SAME sampled shots at every beam in ``ladder`` (ascending),
    records the LER and failure count at each beam, and reports the smallest
    beam whose LER has *plateaued* -- i.e. is TOST-equivalent (at ``margin_rel``)
    to the next-larger beam's LER.

    ``beam_star`` falls back to the largest beam in the ladder when the plateau
    is **not** statistically resolved at the given shots/margin -- this is the
    conservative choice (use the strongest beam; never under-claim the beam
    needed for the MLE). The returned ``plateau_resolved`` flag is ``True`` only
    when some beam strictly smaller than the max was confirmed TOST-equivalent to
    its successor, and ``False`` when the result fell back to the max because no
    smaller beam was statistically confirmed to plateau. Callers wanting a
    resolved plateau must supply adequate shots for the LER/margin (a low LER at
    a tight margin needs many shots for TOST power).

    Because the same fixed shot set is reused across beams, only the decoder
    (not the noise realization) changes, so the comparison isolates the effect
    of the beam cutoff.

    Honest labeling: this measures convergence of the *most-likely-ERROR* anchor
    (near-exact at large beam), not coset-ML. See the module docstring.

    Parameters
    ----------
    circuit : stim.Circuit
        Circuit to sample and decode.
    shots : int, default=20000
        Number of Monte-Carlo shots (shared across all beams).
    ladder : sequence of int, default=(8, 16, 32, 64)
        Beam (``det_beam``) values to evaluate.
    margin_rel : float, default=0.10
        Relative TOST equivalence margin passed to
        ``qldpc.foundation.stats.tost_equivalent``.
    seed : int, default=0
        Seed for the detector sampler.
    dem : stim.DetectorErrorModel, optional
        DEM to build the decoders from. Defaults to
        ``circuit.detector_error_model(decompose_errors=False)``.

    Returns
    -------
    dict
        ``{"ler_by_beam": {beam: ler}, "fails_by_beam": {beam: fails},
        "beam_star": int, "plateau_resolved": bool, "shots": int,
        "margin_rel": float}``. ``plateau_resolved`` is ``True`` iff a beam
        strictly smaller than the max was confirmed TOST-equivalent to its
        successor; ``False`` when ``beam_star`` fell back to the largest beam.
    """
    if dem is None:
        dem = circuit.detector_error_model(decompose_errors=False)

    ladder = sorted(int(b) for b in ladder)
    dets, obs = _sample(circuit, shots, seed)
    n = len(dets)

    ler_by_beam = {}
    fails_by_beam = {}
    for beam in ladder:
        ler, fails = _ler_from_samples(dem, dets, obs, beam)
        ler_by_beam[beam] = ler
        fails_by_beam[beam] = fails

    # beam_star = smallest beam whose LER is TOST-equivalent to the next-larger
    # beam's LER (a plateau). Default to the largest beam if nothing plateaus.
    beam_star = ladder[-1]
    plateau_resolved = False
    for i in range(len(ladder) - 1):
        b_lo, b_hi = ladder[i], ladder[i + 1]
        if tost_equivalent(fails_by_beam[b_lo], n,
                           fails_by_beam[b_hi], n,
                           margin_rel=margin_rel):
            beam_star = b_lo
            plateau_resolved = True
            break

    return {
        "ler_by_beam": ler_by_beam,
        "fails_by_beam": fails_by_beam,
        "beam_star": beam_star,
        "plateau_resolved": plateau_resolved,
        "shots": n,
        "margin_rel": margin_rel,
    }
