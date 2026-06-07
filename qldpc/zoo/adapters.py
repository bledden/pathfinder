"""Uniform decoder interface + CPU-local classical adapters on a SHARED DEM.

Every decoder in the matched harness is constructed FROM one
``stim.DetectorErrorModel`` (``decompose_errors=False``) and exposes a single
``decode_batch(dets) -> obs_pred`` surface, so a matched runner can decode the
SAME shots with every decoder (apples-to-apples LER). All ldpc-family adapters
derive their (check matrix H, observable matrix Lo, priors) from the SAME DEM via
``canon_dem.extract`` — the matched-protocol requirement: identical DEM/priors
across decoders. Tesseract ingests the DEM directly (it does not use H/Lo).

Interface
---------
Each adapter exposes:
  * ``.name``   -- str identifier (e.g. ``"BPOSD-10"``),
  * ``.config`` -- dict of pinned hyperparameters (provenance / pre-registration),
  * ``.dem``    -- the shared ``stim.DetectorErrorModel`` it was built from
                   (provenance guard: a matched harness asserts all adapters
                   share one DEM object),
  * ``.decode_batch(dets: np.ndarray[bool] (shots, n_det))``
        ``-> np.ndarray[bool] (shots, n_obs)`` -- predicted observables.

For an ldpc decoder, each shot's detector syndrome is decoded to an error
estimate ``e_hat`` (length n_err); predicted observables = ``(Lo @ e_hat) % 2``.
ldpc 2.4.x exposes only single-shot ``decoder.decode(syndrome)`` (no batched
entry point), so ldpc adapters loop over shots.

DISCOVERED ldpc API (ldpc 2.4.1)
--------------------------------
* ``ldpc.BpDecoder(pcm, error_channel=..., max_iter=, bp_method='minimum_sum',
  ms_scaling_factor=, schedule='parallel'|'serial', ...)`` — pure min-sum BP.
* ``ldpc.BpOsdDecoder(pcm, error_channel=..., max_iter=, bp_method=,
  ms_scaling_factor=, schedule=, osd_method='osd_0'|'osd_cs', osd_order=)`` —
  BP with Ordered-Statistics post-processing. ``osd_method='osd_0'`` REQUIRES
  ``osd_order=0`` (a non-zero order with ``osd_0`` raises ValueError); the
  combination-sweep is ``osd_method='osd_cs'`` (order>0).
* ``ldpc.BpLsdDecoder(pcm, error_channel=..., max_iter=, bp_method=,
  ms_scaling_factor=, lsd_method='lsd_0'|'lsd_cs', lsd_order=)`` — BP +
  Localised-Statistics Decoder.
All return a uint8 error vector of length n_err from ``.decode(syndrome_uint8)``.
``error_channel`` is the per-mechanism prior list (the matched DEM priors).

DISCOVERED Tesseract API (tesseract_decoder 0.1.x, via probe/tesseract_anchor)
-----------------------------------------------------------------------------
* ``tesseract_decoder.tesseract.TesseractConfig(dem=<DEM>, det_beam=<int>)``,
  ``cfg.compile_decoder()`` -> decoder with native
  ``decode_batch(dets: bool[shots,n_det]) -> bool[shots,n_obs]``. MLE anchor.
"""
import numpy as np

from canon_dem import extract


# Pre-committed deterministic tie-break policy per adapter (gate G2). Each adapter
# declares its concrete tie-break (verified deterministic in the T1 review); the
# matched harness asserts every decoder's ``.tie_break`` is one of these BEFORE
# the LER grid runs, so no adapter can silently fall back to a default ordering.
#   BP        -> "min_sum_parallel_hard_decision" (hard decision off the parallel
#                min-sum messages; no post-processing, fully deterministic)
#   BPOSD-0   -> "osd0_reliability_order"  (OSD-0 pivots by BP reliability order)
#   BPOSD-10  -> "osd_cs_order10"          (combination-sweep order 10)
#   BPLSD     -> "lsd_cs_order10"          (localised-statistics combination-sweep)
#   Tesseract -> "astar_beam64_lowest_cost" (A* beam, lowest-cost coset wins ties)
APPROVED_TIE_BREAKS = {
    "min_sum_parallel_hard_decision",
    "osd0_reliability_order",
    "osd_cs_order10",
    "lsd_cs_order10",
    "astar_beam64_lowest_cost",
}


# Pinned min-sum BP hyperparameters shared across the BP-family adapters. These
# are the provenance constants the prereg/grid commit to. ``bp_method='ms'`` and
# ``osd_method='osd_cs'`` mirror what canon_dem.decode_bposd already uses; the
# remaining values (ms_scaling_factor, max_iter, schedule) are pinned here
# because canon_dem.decode_bposd does not set them (it takes ldpc's defaults).
_BP_MAX_ITER = 30
_BP_MS_SCALING = 0.625          # standard normalized-min-sum scaling factor
_BP_METHOD = "minimum_sum"      # min-sum BP (the kernel target)
_BP_SCHEDULE = "parallel"


class _LdpcAdapter:
    """Base for ldpc-family adapters: build H/Lo/priors from the shared DEM,
    decode each shot's syndrome to an error estimate, map to observables."""

    def __init__(self, dem, name, config, decoder, tie_break):
        self.dem = dem
        self.name = name
        self.config = dict(config)
        # Declared deterministic tie-break (gate G2). No silent default: the
        # matched harness asserts this is in APPROVED_TIE_BREAKS.
        self.tie_break = tie_break
        self._decoder = decoder
        ex = extract(dem)
        # Lo: (n_obs x n_err) GF2 map from error mechanisms to observables.
        self._Lo = ex["Lo"].toarray().astype(np.uint8)
        self._n_obs = ex["n_obs"]
        self._n_err = ex["n_err"]
        self._n_det = ex["n_det"]

    def decode_batch(self, dets):
        dets = np.asarray(dets, dtype=bool)
        shots = dets.shape[0]
        out = np.zeros((shots, self._n_obs), dtype=bool)
        syn_u8 = dets.astype(np.uint8)
        Lo = self._Lo
        for i in range(shots):
            e_hat = self._decoder.decode(syn_u8[i])
            # predicted observables = (Lo @ e_hat) % 2
            pred = (Lo @ np.asarray(e_hat, dtype=np.uint8)) & 1
            out[i] = pred.astype(bool)
        return out


def _priors(dem):
    """Per-mechanism priors from the shared DEM, clipped for ldpc stability."""
    pri = extract(dem)["priors"]
    return list(np.clip(pri, 1e-6, 1 - 1e-6))


def make_bp(dem):
    """Pure min-sum BP (no post-processing): baseline / kernel target."""
    from ldpc import BpDecoder

    H = extract(dem)["H"]
    cfg = dict(decoder="BpDecoder", bp_method=_BP_METHOD,
               ms_scaling_factor=_BP_MS_SCALING, max_iter=_BP_MAX_ITER,
               schedule=_BP_SCHEDULE)
    dec = BpDecoder(H, error_channel=_priors(dem), max_iter=_BP_MAX_ITER,
                    bp_method=_BP_METHOD, ms_scaling_factor=_BP_MS_SCALING,
                    schedule=_BP_SCHEDULE)
    return _LdpcAdapter(dem, "BP", cfg, dec, "min_sum_parallel_hard_decision")


def make_bposd0(dem):
    """BP-OSD order-0 (osd_0): cheapest OSD post-processing."""
    from ldpc import BpOsdDecoder

    H = extract(dem)["H"]
    cfg = dict(decoder="BpOsdDecoder", bp_method=_BP_METHOD,
               ms_scaling_factor=_BP_MS_SCALING, max_iter=_BP_MAX_ITER,
               schedule=_BP_SCHEDULE, osd_method="osd_0", osd_order=0)
    dec = BpOsdDecoder(H, error_channel=_priors(dem), max_iter=_BP_MAX_ITER,
                       bp_method=_BP_METHOD, ms_scaling_factor=_BP_MS_SCALING,
                       schedule=_BP_SCHEDULE, osd_method="osd_0", osd_order=0)
    return _LdpcAdapter(dem, "BPOSD-0", cfg, dec, "osd0_reliability_order")


def make_bposd10(dem):
    """BP-OSD order-10 combination-sweep (osd_cs): the strong classical bar."""
    from ldpc import BpOsdDecoder

    H = extract(dem)["H"]
    cfg = dict(decoder="BpOsdDecoder", bp_method=_BP_METHOD,
               ms_scaling_factor=_BP_MS_SCALING, max_iter=_BP_MAX_ITER,
               schedule=_BP_SCHEDULE, osd_method="osd_cs", osd_order=10)
    dec = BpOsdDecoder(H, error_channel=_priors(dem), max_iter=_BP_MAX_ITER,
                       bp_method=_BP_METHOD, ms_scaling_factor=_BP_MS_SCALING,
                       schedule=_BP_SCHEDULE, osd_method="osd_cs", osd_order=10)
    return _LdpcAdapter(dem, "BPOSD-10", cfg, dec, "osd_cs_order10")


def make_bplsd(dem):
    """BP + Localised-Statistics Decoder (lsd_cs, order 10): modern classical bar."""
    from ldpc import BpLsdDecoder

    H = extract(dem)["H"]
    lsd_order = 10
    cfg = dict(decoder="BpLsdDecoder", bp_method=_BP_METHOD,
               ms_scaling_factor=_BP_MS_SCALING, max_iter=_BP_MAX_ITER,
               schedule=_BP_SCHEDULE, lsd_method="lsd_cs", lsd_order=lsd_order)
    dec = BpLsdDecoder(H, error_channel=_priors(dem), max_iter=_BP_MAX_ITER,
                       bp_method=_BP_METHOD, ms_scaling_factor=_BP_MS_SCALING,
                       schedule=_BP_SCHEDULE, lsd_method="lsd_cs",
                       lsd_order=lsd_order)
    return _LdpcAdapter(dem, "BPLSD", cfg, dec, "lsd_cs_order10")


class TesseractAdapter:
    """Tesseract MLE anchor: ingests the DEM directly (NOT H/Lo) and exposes a
    native batched ``decode_batch``. Reuses probe/tesseract_anchor's build path."""

    def __init__(self, dem, det_beam=64):
        self.dem = dem
        self.name = "Tesseract"
        self.det_beam = int(det_beam)
        self.config = dict(decoder="Tesseract", det_beam=self.det_beam)
        # Declared deterministic tie-break (gate G2): A* beam search, lowest-cost
        # coset wins ties (beam width = det_beam = 64).
        self.tie_break = "astar_beam64_lowest_cost"
        from qldpc.probe.tesseract_anchor import _build_decoder

        self._decoder = _build_decoder(dem, self.det_beam)
        self._n_obs = dem.num_observables

    def decode_batch(self, dets):
        dets = np.asarray(dets, dtype=bool)
        pred = np.asarray(self._decoder.decode_batch(dets), dtype=bool)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred


def make_tesseract(dem, det_beam=64):
    return TesseractAdapter(dem, det_beam=det_beam)


# Registry: name -> factory(dem). Default order is the matched-grid order.
_FACTORIES = {
    "BPOSD-0": make_bposd0,
    "BPOSD-10": make_bposd10,
    "BPLSD": make_bplsd,
    "BP": make_bp,
    "Tesseract": make_tesseract,
}

DEFAULT_DECODERS = ("BPOSD-0", "BPOSD-10", "BPLSD", "BP", "Tesseract")


def build_decoders(dem, which=DEFAULT_DECODERS):
    """Construct all requested adapters from ONE shared DEM object.

    Every returned adapter has ``.dem is dem`` (provenance for the matched
    harness). ``which`` selects/orders the adapters by registry name.
    """
    decoders = []
    for name in which:
        if name not in _FACTORIES:
            raise KeyError(f"unknown decoder {name!r}; known: {sorted(_FACTORIES)}")
        decoders.append(_FACTORIES[name](dem))
    return decoders
