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
    # External adapters (see RelayBPAdapter / SlidingWindowAdapter below):
    #   Relay-BP        -> relay-bp's disjoint-ensemble relay schedule; the final
    #                      hard decision is the lowest-weight set that satisfies the
    #                      `nconv`/stop_nconv stopping rule (deterministic gammas).
    #   Sliding-window  -> per-window BP-OSD-cs (order 10) over the commit region;
    #                      the committed coset is fixed by OSD's combination-sweep
    #                      order, then carried forward — deterministic.
    "relay_bp_nconv_disjoint_ensemble",
    "sliding_window_bposd_cs_commit",
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


# --------------------------------------------------------------------------- #
# External adapter 1: Relay-BP (relay-bp[stim] 0.2.2)                          #
# --------------------------------------------------------------------------- #
# DISCOVERED Relay-BP API (relay-bp 0.2.2)
# ----------------------------------------
# Construct-from-DEM:
#   from relay_bp.stim import CheckMatrices
#   cm = CheckMatrices.from_dem(dem)        # -> .check_matrix (ndet x E csc),
#                                           #    .observables_matrix (nobs x E csc),
#                                           #    .error_priors (E,)
#   dec = relay_bp.RelayDecoderF64(cm.check_matrix, error_priors=cm.error_priors,
#             gamma0=, pre_iter=, num_sets=, set_max_iter=, gamma_dist_interval=,
#             stop_nconv=, stopping_criterion='nconv')   # disjoint-relay ensemble
#   runner = relay_bp.ObservableDecoderRunner(dec, cm.observables_matrix,
#                                             include_decode_result=False)
# Decode:
#   runner.decode_observables_batch(syndromes uint8 [shots, n_det])
#       -> predicted observables uint8 [shots, n_obs]
# This is exactly the path relay_bp.stim.SinterDecoder_RelayBP uses internally
# (verified by reading its .build_observable_decoder / .compile_decoder_for_dem),
# minus sinter's bit-packing — we drive the runner directly for a clean
# decode_batch surface. Sinter-compatible, so it satisfies the matched protocol.
_RELAY_BP_DEFAULTS = dict(
    gamma0=0.1,
    pre_iter=80,
    num_sets=60,
    set_max_iter=60,
    gamma_dist_interval=(-0.24, 0.66),
    stop_nconv=5,
    stopping_criterion="nconv",
)


class RelayBPAdapter:
    """Relay-BP adapter (in-process). Builds the relay-BP decoder from the SAME
    shared DEM via ``relay_bp.stim.CheckMatrices.from_dem`` and decodes a batch of
    syndromes straight to observables. G1 holds trivially: ``.dem is dem``."""

    def __init__(self, dem, **params):
        import importlib.metadata as _md

        import relay_bp
        from relay_bp.stim import CheckMatrices

        self.dem = dem
        self.name = "RelayBP"
        try:
            ver = _md.version("relay-bp")
        except Exception:  # pragma: no cover - metadata always present once installed
            ver = "unknown"
        cfg = dict(_RELAY_BP_DEFAULTS)
        cfg.update(params)
        self.config = dict(decoder="RelayBP", relay_bp_version=ver, **cfg)
        # Deterministic relay schedule (fixed gamma distribution + nconv stop).
        self.tie_break = "relay_bp_nconv_disjoint_ensemble"

        cm = CheckMatrices.from_dem(dem)
        self._n_obs = cm.observables_matrix.shape[0]
        decoder = relay_bp.RelayDecoderF64(
            cm.check_matrix,
            error_priors=cm.error_priors,
            **cfg,
        )
        self._runner = relay_bp.ObservableDecoderRunner(
            decoder, cm.observables_matrix, include_decode_result=False)

    def decode_batch(self, dets):
        dets = np.asarray(dets, dtype=bool)
        pred = np.asarray(
            self._runner.decode_observables_batch(dets.astype(np.uint8)))
        pred = (pred % 2).astype(bool)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred


def relay_bp_available():
    """True iff relay-bp[stim] is importable (import-guarded zoo membership)."""
    try:
        import relay_bp  # noqa: F401
        from relay_bp.stim import CheckMatrices  # noqa: F401
    except Exception:
        return False
    return True


def make_relay_bp(dem, **params):
    return RelayBPAdapter(dem, **params)


# --------------------------------------------------------------------------- #
# External adapter 2: sliding-window streaming decoder (qLDPC 0.2.9)           #
# --------------------------------------------------------------------------- #
# DISCOVERED sliding-window API (qLDPC 0.2.9, qLDPCOrg/qLDPC)
# ----------------------------------------------------------
#   from qldpc.decoders import SlidingWindowDecoder
#   swd = SlidingWindowDecoder(window_size=W, stride=S,
#                              detector_to_time=<int callable>,
#                              with_BP_OSD=True, max_iter=, bp_method='ms',
#                              osd_method='osd_cs', osd_order=10)  # per-window dec
#   compiled = swd.compile_decoder_for_dem(dem)   # builds (detection, commit)
#                                                 # regions from the time map
#   pred = compiled.decode_shots(dets uint8 [shots,n_det])  # -> obs uint8
# This is the TRUE streaming decoder: each window decodes its detection region,
# COMMITS the commit-region (first `stride` time layers) errors, then carries the
# committed correction forward by re-biasing later windows' syndromes
# (``net_error @ detector_flip_matrix.T``) — i.e. inter-window carry-over, not
# "BP-OSD on a windowed DEM".
#
# WHY A SUBPROCESS: the PyPI/GitHub package's import name is ALSO ``qldpc`` and
# collides byte-for-byte with this repo's local ``qldpc`` namespace package
# (installing it shadows the repo and breaks every ``qldpc.*`` import). It is
# therefore installed into an ISOLATED dir and driven from ``_swd_worker.py`` in
# a subprocess whose sys.path excludes the repo. See that worker's docstring.
#
# G1: the worker rebuilds the byte-identical DEM from ``str(dem)`` (same stim
# version, faithful round-trip) and decodes against it; THIS adapter still holds
# the original shared DEM object as ``.dem``, so the harness' G1 check (which
# inspects ``adapter.dem``) sees the genuine shared object. Same-DEM provenance,
# gate not weakened.
import json as _json
import os as _os
import subprocess as _subprocess
import sys as _sys
import tempfile as _tempfile

_SWD_WORKER = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            "_swd_worker.py")
_DEFAULT_QLDPC_EXT = _os.path.expanduser("~/.venvs/braket/qldpc_ext_clean")

# Per-window inner decoder (BP-OSD combination-sweep, order 10) pinned for the
# streaming lane; the commit coset is fixed by OSD-cs order -> deterministic.
_SWD_DECODER_KWARGS = dict(
    with_BP_OSD=True,
    max_iter=_BP_MAX_ITER,
    bp_method="ms",
    osd_method="osd_cs",
    osd_order=10,
)


def sliding_window_available(ext_dir=None):
    """True iff the isolated qLDPC sliding-window install can be imported in a
    clean subprocess (repo OFF path). Import-guarded: a missing install simply
    drops the adapter from the external zoo, never breaks the core zoo."""
    ext = ext_dir or _os.environ.get("QLDPC_EXT_DIR", _DEFAULT_QLDPC_EXT)
    if not _os.path.isdir(ext):
        return False
    probe = (
        "import sys;"
        "sys.path=[p for p in sys.path if 'Documents/pathfinder' not in p];"
        f"sys.path.append({ext!r});"
        "import qldpc;"
        "from qldpc.decoders import SlidingWindowDecoder"
    )
    try:
        r = _subprocess.run([_sys.executable, "-c", probe],
                            capture_output=True, text=True, timeout=120)
    except Exception:
        return False
    return r.returncode == 0


class SlidingWindowAdapter:
    """Streaming sliding-window adapter. Decodes via the isolated qLDPC install in
    a subprocess (see ``_swd_worker.py``). G1: ``.dem is dem`` (the shared object);
    the worker decodes a byte-identical DEM round-tripped from ``str(dem)``."""

    def __init__(self, dem, rounds, window_size=3, stride=1, ext_dir=None,
                 decoder_kwargs=None):
        self.dem = dem
        self.name = "SlidingWindow"
        self._rounds = int(rounds)
        self._window_size = int(window_size)
        self._stride = int(stride)
        self._ext_dir = ext_dir or _os.environ.get("QLDPC_EXT_DIR",
                                                    _DEFAULT_QLDPC_EXT)
        self._decoder_kwargs = dict(_SWD_DECODER_KWARGS)
        if decoder_kwargs:
            self._decoder_kwargs.update(decoder_kwargs)
        self.config = dict(
            decoder="SlidingWindow",
            qldpc_version="0.2.9",
            window=self._window_size,
            stride=self._stride,
            commit=self._stride,           # commit region size = stride
            rounds=self._rounds,
            inner=dict(self._decoder_kwargs),
        )
        # Per-window BP-OSD-cs over the commit region (deterministic OSD order).
        self.tie_break = "sliding_window_bposd_cs_commit"
        self._n_obs = dem.num_observables

    def decode_batch(self, dets):
        dets = np.asarray(dets, dtype=bool)
        shots = dets.shape[0]
        if shots == 0:
            return np.zeros((0, self._n_obs), dtype=bool)
        tmp = _tempfile.mkdtemp(prefix="swd_")
        dets_path = _os.path.join(tmp, "dets.npy")
        pred_path = _os.path.join(tmp, "pred.npy")
        np.save(dets_path, dets.astype(np.uint8))
        req = {
            "dem_text": str(self.dem),
            "window_size": self._window_size,
            "stride": self._stride,
            "rounds": self._rounds,
            "dets_npy": dets_path,
            "pred_npy": pred_path,
            "decoder_kwargs": self._decoder_kwargs,
        }
        env = dict(_os.environ, QLDPC_EXT_DIR=self._ext_dir)
        r = _subprocess.run(
            [_sys.executable, _SWD_WORKER],
            input=_json.dumps(req), capture_output=True, text=True,
            env=env, timeout=1800,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"sliding-window worker failed (rc={r.returncode}).\n"
                f"STDERR:\n{r.stderr}\nSTDOUT:\n{r.stdout}")
        meta = _json.loads(r.stdout)
        pred = np.load(meta["pred_npy"]).astype(bool)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred


def make_sliding_window(dem, rounds, window_size=3, stride=1, **kw):
    return SlidingWindowAdapter(dem, rounds, window_size=window_size,
                                stride=stride, **kw)


# Registry: name -> factory(dem). Default order is the matched-grid order.
_FACTORIES = {
    "BPOSD-0": make_bposd0,
    "BPOSD-10": make_bposd10,
    "BPLSD": make_bplsd,
    "BP": make_bp,
    "Tesseract": make_tesseract,
}

DEFAULT_DECODERS = ("BPOSD-0", "BPOSD-10", "BPLSD", "BP", "Tesseract")


def build_decoders(dem, which=DEFAULT_DECODERS, include_external=False,
                   rounds=None, sliding_window=None):
    """Construct all requested adapters from ONE shared DEM object.

    Every returned adapter has ``.dem is dem`` (provenance for the matched
    harness). ``which`` selects/orders the *core* adapters by registry name.

    External adapters (Relay-BP, sliding-window) are OPT-IN via
    ``include_external=True`` and are added ONLY when their package is available
    (import-guarded), so the core zoo always builds even without them:

      * Relay-BP is added iff ``relay_bp_available()``.
      * Sliding-window is added iff ``sliding_window_available()`` AND ``rounds``
        is supplied (the streaming decoder needs the per-round detector->time map;
        the DEM alone carries no detector coordinates in this repo). It is built
        from the SAME shared DEM (its worker round-trips the byte-identical DEM).

    Args:
        dem: the shared stim.DetectorErrorModel.
        which: core decoders to build (names in ``_FACTORIES``).
        include_external: also append the available external adapters.
        rounds: number of syndrome rounds (REQUIRED to add the sliding-window
            adapter; ignored otherwise).
        sliding_window: optional dict of sliding-window params (e.g.
            ``{"window_size": 3, "stride": 1}``).
    """
    decoders = []
    for name in which:
        if name not in _FACTORIES:
            raise KeyError(f"unknown decoder {name!r}; known: {sorted(_FACTORIES)}")
        decoders.append(_FACTORIES[name](dem))

    if include_external:
        if relay_bp_available():
            decoders.append(make_relay_bp(dem))
        if rounds is not None and sliding_window_available():
            sw_kw = dict(sliding_window or {})
            decoders.append(make_sliding_window(dem, rounds=rounds, **sw_kw))

    return decoders
