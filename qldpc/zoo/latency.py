"""T9 latency harness — the LATENCY axis of the Pareto hero figure.

Measures **decode latency per syndrome** for every zoo decoder on the canonical
[[72,12,6]] BB SI1000 circuit-level DEM at d=6, R=6, p=0.003, under Coda's locked
timing discipline:

    warmup >= 10 reps, measure >= 100 reps, drop the first 10% of measured reps,
    report mean, p99.9, throughput (shots/s), and latency/syndrome (us).

Two timing paths, matched to the decoder's execution model:

  * GPU decoders -- the fused Triton min-sum kernel ``BpTriton`` and the torch
    edge-list baseline ``BpGpu`` -- are timed with ``torch.cuda.Event`` (their own
    ``bench_latency`` classmethods, reused verbatim: warmup, CUDA-event start/stop,
    ``cuda.synchronize`` per rep). Each rep decodes the WHOLE batch; per-syndrome
    latency is the amortized throughput (mean batch ms / batch size).

  * CPU decoders -- ldpc BP / BP-OSD-0 / BP-OSD-10 / BP+LSD, relay_bp Relay-BP,
    tesseract Tesseract (the MLE anchor), and the in-process sliding-window
    decoder -- are timed with ``time.perf_counter`` over a fixed batch, amortized
    to per-syndrome. Same warmup/measure/drop discipline.

SLIDING-WINDOW LATENCY (the fair number) -- PATH (a), in-process isolated load.
------------------------------------------------------------------------------
The LER lane drives the sliding-window decoder in a SUBPROCESS (its PyPI import
name ``qldpc`` collides with the repo's namespace package). For latency that is
unfair: ~1-1.3 s of Python interpreter + import spawn per call dominates the
actual decode and is an integration artifact, not the decoder's cost. T9 instead
loads the isolated ``/workspace/qldpc_ext`` ``SlidingWindowDecoder`` IN-PROCESS
via ``isolated_qldpc_ext`` -- a context manager that evicts the repo ``qldpc.*``
modules, prepends the ext dir to ``sys.path``, imports, then restores the repo
modules so the rest of the harness is unaffected. Verified: the repo ``qldpc``
keeps working after the swap, and the SW decoder decodes in-process correctly.
The reported SW latency is the in-process ``compiled.decode_shots`` wall time
amortized per syndrome -- the genuine decode cost, spawn excluded.

ENV PIN (Triton kernels are version-fragile): the manifest records triton, torch,
CUDA, stim, ldpc, relay-bp, tesseract, scipy, numpy versions + GPU name/driver.

Run on the pod:
    cd /workspace/pf-mle && \
      PYTHONPATH=/workspace/pf-mle:/workspace/pf-mle/qldpc \
      python3 -m qldpc.zoo.latency --out qldpc/zoo/latency_results.json

CUDA is imported lazily (guarded) so this module imports on a CUDA-less Mac for
``pytest tests/`` collection; the GPU timings simply require ``--device cuda``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np

# Ensure ``qldpc/`` is importable when run as a module from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_QLDPC = os.path.dirname(_HERE)              # .../qldpc
_ROOT = os.path.dirname(_QLDPC)              # repo root
for _p in (_QLDPC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Canonical code/grid point for the hero figure.
CANON_ROUNDS = 6
CANON_P = 0.003
CANON_BASIS = "Z"
CANON_NOISE = "si1000"

# Coda's locked discipline.
DEFAULT_WARMUP = 10
DEFAULT_REPS = 100
DROP_FRAC = 0.10

# Default measurement batch (per-syndrome latency is amortized over this batch;
# also drives the throughput number). Larger batches favour the GPU decoders;
# we sweep a range of batches for the GPU/Triton path so the figure can show the
# batch dependence and separate the two deployment regimes:
#   batch 1   = single-shot REAL-TIME latency (no amortization; the binding
#               number for an SC decoding window) -- the headline latency story.
#   batch 16k = steady-state THROUGHPUT (fully amortized; the backlog metric).
# CPU decoders are ~batch-independent (per-syndrome latency is amortized and
# invariant), so they are NOT swept -- the existing single batch is reused.
DEFAULT_BATCH = 1024
# Small batches (1/4/16) added for the single-shot latency story; the large
# batches (256..16384) remain for throughput. Ordered small->large.
GPU_BATCH_SWEEP = (1, 4, 16, 256, 1024, 4096, 16384)
# The batch designated as the single-shot real-time point (vs the throughput
# representative, which stays the largest batch for cycle_time_budget back-compat).
SINGLE_SHOT_BATCH = 1

# Bootstrap-CI protocol (Coda figure-lock: error bars = bootstrap CI on each
# latency point, >=1000 resamples of the per-rep measure window, 95% CI).
BOOTSTRAP_N = 2000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 12345

# Per-decoder CPU batch caps: the ldpc-OSD combination-sweep family is
# milliseconds-per-shot (random/dense syndromes are worst-case for OSD), so a
# smaller batch keeps each rep tractable while still amortizing host overhead.
# Per-syndrome latency is invariant to the batch (it is amortized), so a smaller
# batch is fair -- it only changes the reported throughput's averaging window.
CPU_BATCH_CAPS = {
    "BP": 1024,
    "BPOSD-0": 256,
    "BPOSD-10": 128,
    "BPLSD": 256,
    "RelayBP": 512,
    "Tesseract": 256,
    "SlidingWindow": 128,
}

_DEFAULT_QLDPC_EXT = os.environ.get(
    "QLDPC_EXT_DIR",
    "/workspace/qldpc_ext"
    if os.path.isdir("/workspace/qldpc_ext")
    else os.path.expanduser("~/.venvs/braket/qldpc_ext_clean"),
)


# --------------------------------------------------------------------------- #
# Canonical DEM (the figure's code/grid point)                                #
# --------------------------------------------------------------------------- #
def canonical_dem(rounds=CANON_ROUNDS, p=CANON_P, basis=CANON_BASIS,
                  noise=CANON_NOISE):
    """Build the canonical [[72,12,6]] BB memory DEM (decompose_errors=False).

    Returns ``(circuit, dem)`` -- the SAME construction the LER grid uses, so the
    latency axis and the gap-to-MLE axis share one DEM family."""
    try:
        from qldpc.foundation.circuits import build_memory
        from qldpc.bb_code import BBCode
    except Exception:  # flat layout (qldpc/ on sys.path)
        from foundation.circuits import build_memory
        from bb_code import BBCode
    circ = build_memory(BBCode(), rounds=rounds, p=p, basis=basis, noise=noise)
    dem = circ.detector_error_model(decompose_errors=False)
    return circ, dem


def sample_detectors(circ, shots, seed=0):
    """Sample realistic detector syndromes from the circuit at the grid p.

    The fair latency workload is the actual error distribution the decoder sees
    in operation (sparse syndromes at p=0.003), NOT uniform-random/dense
    syndromes -- dense syndromes are worst-case for the OSD combination sweep and
    inflate latency unrealistically. Returns ``dets`` (bool[shots, n_det])."""
    sampler = circ.compile_detector_sampler(seed=seed)
    dets, _obs = sampler.sample(int(shots), separate_observables=True)
    return np.asarray(dets, dtype=bool)


def uniform_random_detectors(n_det, shots, seed=0):
    """Uniform-random (dense, p=0.5 per bit) detector syndromes.

    The OSD worst-case workload: dense syndromes drive the OSD combination sweep
    to a large residual support, inflating BP-OSD-10 latency vs the realistic
    (sparse, operational p=0.003) distribution. Used by the both-workloads table
    to quantify that inflation. NOT the headline latency (realistic is) -- this is
    the adversarial bound."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(int(shots), int(n_det))).astype(bool)


# --------------------------------------------------------------------------- #
# Discipline: warmup / measure / drop-first-10% timing of a callable          #
# --------------------------------------------------------------------------- #
def time_callable(fn, *, n_warmup=DEFAULT_WARMUP, n_reps=DEFAULT_REPS,
                  drop_frac=DROP_FRAC):
    """Time a no-arg ``fn()`` (one batch decode) with the locked discipline.

    Returns the kept per-rep wall times in ms (np.ndarray). Discipline:
    ``n_warmup`` untimed warmups, ``n_reps`` timed reps, drop the first
    ``ceil(drop_frac * n_reps)`` to remove residual warmup/cache effects."""
    for _ in range(int(n_warmup)):
        fn()
    times_ms = np.empty(int(n_reps), dtype=np.float64)
    for k in range(int(n_reps)):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_ms[k] = (t1 - t0) * 1e3
    drop = int(np.ceil(drop_frac * n_reps))
    return times_ms[drop:]


def bootstrap_ci(times_ms, *, batch, n_boot=BOOTSTRAP_N, alpha=BOOTSTRAP_ALPHA,
                 seed=BOOTSTRAP_SEED):
    """Percentile bootstrap 95% CI on the MEAN per-syndrome latency (us).

    Protocol (stated in the figure caption, reviewer-2 armor): the kept per-rep
    measure window (``times_ms``, ms, drop-first-10% already applied) is resampled
    WITH REPLACEMENT ``n_boot`` times (default 2000 >= the locked 1000 floor); for
    each resample the mean batch time is amortized to us/syndrome
    (``mean_ms*1e3/batch``); the 2.5th / 97.5th percentiles of those bootstrap
    means are the CI bounds. Returns (lo_us, hi_us, n_boot).

    This bounds the sampling uncertainty of the reported MEAN -- it does NOT widen
    to the p99.9 tail (that asymmetry is shown by the whisker, not the CI)."""
    t = np.asarray(times_ms, dtype=np.float64)
    if t.size == 0 or batch <= 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, t.size, size=(int(n_boot), t.size))
    boot_mean_ms = t[idx].mean(axis=1)
    boot_us = boot_mean_ms * 1e3 / batch
    lo = float(np.percentile(boot_us, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_us, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi, int(n_boot)


def summarize(times_ms, batch, *, keep_per_rep=True):
    """Reduce kept per-batch times (ms) to the reported latency record.

    Adds (figure-lock): the bootstrap 95% CI on the mean per-syndrome latency and
    the full per-rep sample window (ms) so the CI / whisker are reproducible from
    the committed JSON without re-running on the pod."""
    times_ms = np.asarray(times_ms, dtype=np.float64)
    mean_ms = float(np.mean(times_ms))
    p999_ms = float(np.percentile(times_ms, 99.9))
    median_ms = float(np.median(times_ms))
    thr = float(batch / (mean_ms / 1e3)) if mean_ms > 0 else 0.0
    us_per_syn = float(mean_ms * 1e3 / batch) if batch > 0 else 0.0
    us_per_syn_p999 = float(p999_ms * 1e3 / batch) if batch > 0 else 0.0
    ci = bootstrap_ci(times_ms, batch=batch)
    rec = dict(
        batch=int(batch),
        n_kept=int(len(times_ms)),
        mean_ms=mean_ms,
        median_ms=median_ms,
        p99_9_ms=p999_ms,
        throughput_shots_per_s=thr,
        us_per_syndrome=us_per_syn,
        us_per_syndrome_p99_9=us_per_syn_p999,
    )
    if ci is not None:
        rec["us_per_syndrome_ci95"] = [ci[0], ci[1]]
        rec["bootstrap_n"] = ci[2]
    if keep_per_rep:
        rec["per_rep_ms"] = [float(x) for x in times_ms]
    return rec


# --------------------------------------------------------------------------- #
# In-process isolated sliding-window load (the FAIR SW latency, path a)        #
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def isolated_qldpc_ext(ext_dir=None):
    """Context manager: import the isolated ``qldpc_ext`` package in-process.

    The PyPI ``qldpc`` import name collides with the repo's namespace package, so
    we (1) snapshot+evict every ``qldpc`` / ``qldpc.*`` module, (2) APPEND the ext
    dir to ``sys.path`` (NOT prepend), yield, then (3) evict the ext modules and
    restore the repo ones. After the ``with`` block the repo ``qldpc`` works
    exactly as before (verified).

    Why APPEND, not prepend: the ext dir bundles a compiled ``_cffi_backend.so``
    (v2.0.0, pulled in by its numba dep) that, if it precedes the venv on the
    path, shadows the venv's cffi (v1.17.1) and trips numba's cffi version check
    (``Version mismatch ... 1.17.1 ... 2.0.0``). Appending keeps the venv's
    site-packages (incl. the correct ``_cffi_backend``) at higher priority, while
    ``qldpc`` -- which is NOT in the venv -- still resolves to the ext dir. This
    is what makes an in-process (hence FAIR, no subprocess spawn) sliding-window
    latency measurable."""
    ext = ext_dir or _DEFAULT_QLDPC_EXT
    if not os.path.isdir(ext):
        raise FileNotFoundError(f"qldpc_ext dir not found: {ext}")
    saved = {k: v for k, v in sys.modules.items()
             if k == "qldpc" or k.startswith("qldpc.")}
    for k in list(saved):
        del sys.modules[k]
    sys.path.append(ext)
    try:
        yield ext
    finally:
        # remove ext modules, restore the repo ones, undo the path change.
        ext_mods = {k for k in sys.modules
                    if k == "qldpc" or k.startswith("qldpc.")}
        for k in ext_mods:
            del sys.modules[k]
        with contextlib.suppress(ValueError):
            sys.path.remove(ext)
        sys.modules.update(saved)


def time_sliding_window(dem, rounds, dets, ext_dir=None,
                        window_size=3, stride=1,
                        n_warmup=DEFAULT_WARMUP, n_reps=DEFAULT_REPS):
    """Fair in-process sliding-window decode latency (spawn excluded).

    Builds the isolated ``SlidingWindowDecoder`` in-process, compiles it for the
    shared DEM ONCE (compile is build-time, not per-syndrome), then times only
    ``compiled.decode_shots`` over the provided detector batch with the locked
    discipline. Per-syndrome latency = mean batch ms / batch (amortized)."""
    n_det = dem.num_detectors
    n_layers = rounds + 1
    if n_det % n_layers != 0:
        raise ValueError(
            f"n_det={n_det} not divisible by (rounds+1)={n_layers}")
    per_layer = n_det // n_layers
    dem_text = str(dem)
    batch = int(dets.shape[0])

    def detector_to_time(d):
        return d // per_layer

    dets = np.asarray(dets, dtype=np.uint8)

    with isolated_qldpc_ext(ext_dir):
        import stim as _stim  # the venv stim (ext dir carries none)
        from qldpc.decoders import SlidingWindowDecoder
        ext_dem = _stim.DetectorErrorModel(dem_text)
        swd = SlidingWindowDecoder(
            window_size=int(window_size), stride=int(stride),
            detector_to_time=detector_to_time,
            with_BP_OSD=True, max_iter=30, bp_method="ms",
            osd_method="osd_cs", osd_order=10,
        )
        compiled = swd.compile_decoder_for_dem(ext_dem)
        n_windows = len(compiled.window_decoders)

        def _decode():
            compiled.decode_shots(dets)

        times = time_callable(_decode, n_warmup=n_warmup, n_reps=n_reps)

    rec = summarize(times, batch)
    rec.update(decoder="SlidingWindow", timing="perf_counter_inprocess",
               sw_path="(a) in-process isolated importlib load (spawn excluded)",
               window_size=int(window_size), stride=int(stride),
               n_windows=int(n_windows),
               tie_break="sliding_window_bposd_cs_commit")
    return rec


# --------------------------------------------------------------------------- #
# CPU decoder timing (ldpc family, relay-bp, tesseract)                        #
# --------------------------------------------------------------------------- #
def time_cpu_decoder(adapter, dets, n_warmup=DEFAULT_WARMUP,
                     n_reps=DEFAULT_REPS):
    """Time a zoo adapter's ``decode_batch`` over a realistic detector batch.

    The adapter consumes the SHARED dem; the detector batch is sampled from the
    circuit at the grid p (the operational error distribution -- the fair latency
    workload; dense random syndromes are OSD worst-case). Per-syndrome latency is
    amortized over ``batch``."""
    dets = np.asarray(dets, dtype=bool)
    batch = int(dets.shape[0])

    def _decode():
        adapter.decode_batch(dets)

    times = time_callable(_decode, n_warmup=n_warmup, n_reps=n_reps)
    rec = summarize(times, batch)
    rec.update(decoder=adapter.name, timing="perf_counter_batch",
               tie_break=getattr(adapter, "tie_break", None),
               config=getattr(adapter, "config", None))
    return rec


# --------------------------------------------------------------------------- #
# GPU decoder timing (Triton kernel + torch baseline, CUDA events)            #
# --------------------------------------------------------------------------- #
def time_gpu_decoder(which, dem, batch, device="cuda", n_warmup=DEFAULT_WARMUP,
                     n_reps=DEFAULT_REPS, seed=0, block_s=256):
    """Time a GPU BP decoder via its CUDA-event ``bench_latency`` classmethod.

    ``which`` in {"bp_triton", "bp_gpu"}. Reuses the kernel modules' own
    CUDA-event harness (warmup + per-rep event start/stop + synchronize), which
    already returns mean/p99.9/throughput. We add the per-syndrome amortization
    and the drop-first-10% by re-binning here is not possible (the classmethod
    keeps all reps); instead we request n_reps and read its mean/p99.9 (the
    classmethod's reps already follow warmup>=10, measure>=n_reps). Per-syndrome
    latency = mean_ms*1e3/batch."""
    if which == "bp_triton":
        from qldpc.kernel.bp_triton import BpTriton as Kern
        extra = dict(block_s=block_s)
    elif which == "bp_gpu":
        from qldpc.kernel.bp_gpu import BpGpu as Kern
        extra = {}
    else:
        raise ValueError(f"unknown gpu decoder {which!r}")

    res = Kern.bench_latency(dem, shots=int(batch), device=device,
                             n_warmup=n_warmup, n_iter=n_reps, seed=seed, **extra)
    # Apply the SAME drop-first-10% discipline as the CPU path (the kernel
    # classmethod keeps all reps), then derive mean/p99.9/CI/per-rep from the
    # kept window so the GPU and CPU points are computed identically.
    per_rep = res.get("per_rep_ms")
    if per_rep is not None and len(per_rep) > 0:
        all_ms = np.asarray(per_rep, dtype=np.float64)
        drop = int(np.ceil(DROP_FRAC * len(all_ms)))
        kept = all_ms[drop:]
        rec = summarize(kept, batch)
    else:  # no per-rep window (shouldn't happen): fall back to reported scalars
        mean_ms = res["mean_ms"]
        p999_ms = res["p99_9_ms"]
        rec = dict(
            batch=int(batch), n_kept=int(res["n_iter"]),
            mean_ms=mean_ms, median_ms=mean_ms, p99_9_ms=p999_ms,
            throughput_shots_per_s=res["throughput_shots_per_s"],
            us_per_syndrome=float(mean_ms * 1e3 / batch) if batch else 0.0,
            us_per_syndrome_p99_9=float(p999_ms * 1e3 / batch) if batch else 0.0,
        )
    rec.update(
        decoder=which,
        timing=("cuda_event" if res.get("used_cuda_events") else "perf_counter"),
        device=res["device"],
        max_iter=res["max_iter"],
        used_cuda_events=bool(res.get("used_cuda_events")),
        workload="uniform_random",  # GPU bench uses random syndromes; BP is
                                    # workload-insensitive (flooding min-sum is
                                    # data-independent in latency) -- noted, not
                                    # belabored.
    )
    if "block_s" in res:
        rec["block_s"] = res["block_s"]
    return rec


# --------------------------------------------------------------------------- #
# Environment manifest (version pin)                                           #
# --------------------------------------------------------------------------- #
def env_manifest():
    """Record version pins for reproducibility (Triton kernels are fragile)."""
    import importlib.metadata as md
    import platform

    def _v(pkg):
        try:
            return md.version(pkg)
        except Exception:
            return None

    env = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=_v("numpy"),
        scipy=_v("scipy"),
        stim=_v("stim"),
        ldpc=_v("ldpc"),
        relay_bp=_v("relay-bp"),
        tesseract_decoder=_v("tesseract-decoder"),
        torch=_v("torch"),
        triton=_v("triton"),
        qldpc_ext=_v("qldpc"),  # may be None when measured from the repo process
    )
    # torch/CUDA + GPU details (guarded import; CUDA-less Mac -> Nones).
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            try:
                import subprocess
                drv = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=20)
                env["gpu_driver"] = drv.stdout.strip().splitlines()[0] \
                    if drv.returncode == 0 and drv.stdout.strip() else None
            except Exception:
                env["gpu_driver"] = None
            env["gpu_clock"] = gpu_clock_state()
        else:
            env["gpu_name"] = None
            env["gpu_driver"] = None
            env["gpu_clock"] = None
    except Exception:
        env["cuda"] = None
        env["gpu_name"] = None
        env["gpu_driver"] = None
        env["gpu_clock"] = None
    return env


def gpu_clock_state():
    """Snapshot the GPU clock conditions for the manifest / figure caption.

    CLOCK NOTE (Coda fallback): clock-locking (``nvidia-smi -lgc``) is NOT
    permitted on this RunPod container ("user does not have permission"). The GPU
    runs at applications-clock 1980 MHz and boosts to 1980 MHz under load (idle
    345 MHz). We therefore (a) STATE these conditions here, and (b) run a
    variance-over-runs check (``measure_clock_variance``) to confirm thermal
    stability despite unlocked clocks. Returns the queried clocks + the note."""
    out = dict(
        clocks_locked=False,
        lock_note=("clock-locking not permitted on RunPod container "
                   "(user does not have permission); clocks left at default"),
        applications_clock_mhz=None,
        gr_clock_mhz=None,
        max_gr_clock_mhz=None,
        idle_gr_clock_mhz=345,
        boost_clock_mhz=1980,
        runtime_conditions="single-tenant pod, sustained-load bench",
    )
    try:
        import subprocess
        q = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.applications.graphics,clocks.gr,clocks.max.gr",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        if q.returncode == 0 and q.stdout.strip():
            a, g, m = (x.strip() for x in
                       q.stdout.strip().splitlines()[0].split(","))
            out["applications_clock_mhz"] = int(float(a)) if a.isdigit() or \
                a.replace(".", "").isdigit() else None
            out["gr_clock_mhz"] = int(float(g)) if g.replace(".", "").isdigit() \
                else None
            out["max_gr_clock_mhz"] = int(float(m)) if m.replace(".", "").isdigit() \
                else None
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# Both-workloads table: realistic (operational p) vs uniform-random (OSD       #
# worst-case) syndromes for the OSD/LSD decoders -- quantify the ~26x          #
# BP-OSD-10 inflation. Realistic is the headline; both are reported.           #
# --------------------------------------------------------------------------- #
# The OSD/LSD family whose latency is workload-SENSITIVE (the combination sweep
# walks a residual support that grows with syndrome density). Pure BP and
# Tesseract are workload-insensitive (BP flooding is data-independent; Tesseract
# beam-search cost is dominated by the beam) -- excluded with a note.
BOTH_WORKLOADS_DECODERS = ("BPOSD-0", "BPOSD-10", "BPLSD")
BOTH_WORKLOADS_REFERENCE_NOTE = (
    "Pure-BP and Tesseract are workload-insensitive (BP flooding min-sum is "
    "data-independent in latency; Tesseract's cost is beam-dominated) -- not "
    "measured here. The OSD/LSD combination sweep is the workload-sensitive "
    "family: its residual support (and thus GE work) grows with syndrome density.")


def measure_both_workloads(dem, *, decoders=BOTH_WORKLOADS_DECODERS,
                           realistic_dets, n_det, batch=128, seed=0,
                           n_warmup=DEFAULT_WARMUP, n_reps=DEFAULT_REPS):
    """Latency under REALISTIC vs UNIFORM-RANDOM syndromes for OSD/LSD decoders.

    realistic = circuit syndromes at the grid p (operational; the headline).
    uniform   = dense p=0.5-per-bit syndromes (OSD worst-case).
    Both timed with the SAME batch/discipline so the inflation ratio is fair.
    Returns a dict keyed by decoder -> {realistic, uniform, inflation_x}."""
    from qldpc.zoo import adapters as A
    makers = {
        "BPOSD-0": lambda: A.make_bposd0(dem),
        "BPOSD-10": lambda: A.make_bposd10(dem),
        "BPLSD": lambda: A.make_bplsd(dem),
    }
    real = np.asarray(realistic_dets, dtype=bool)
    b = min(int(batch), real.shape[0])
    real = real[:b]
    uni = uniform_random_detectors(n_det, b, seed=seed)

    out = {}
    for name in decoders:
        if name not in makers:
            continue
        try:
            adapter = makers[name]()
            rec_real = time_cpu_decoder(adapter, real,
                                        n_warmup=n_warmup, n_reps=n_reps)
            rec_uni = time_cpu_decoder(adapter, uni,
                                       n_warmup=n_warmup, n_reps=n_reps)
            inflation = (rec_uni["us_per_syndrome"] / rec_real["us_per_syndrome"]
                         if rec_real["us_per_syndrome"] > 0 else None)
            out[name] = dict(
                decoder=name,
                realistic=rec_real,
                uniform_random=rec_uni,
                inflation_x=float(inflation) if inflation else None,
            )
        except Exception as e:
            import traceback
            out[name] = dict(decoder=name, error=repr(e),
                             traceback=traceback.format_exc()[-800:])
    return out


# --------------------------------------------------------------------------- #
# Variance-over-runs check (Coda fallback for "can't lock clocks"): repeat the #
# GPU bench N>=3x across the session, report per-decoder run-to-run CV.        #
# --------------------------------------------------------------------------- #
def measure_clock_variance(dem, *, device="cuda", n_runs=3,
                           batches=(1, 16384), which=("bp_triton", "bp_gpu"),
                           seed=0, n_warmup=DEFAULT_WARMUP, n_reps=DEFAULT_REPS):
    """Repeat the GPU latency bench ``n_runs`` times; report run-to-run CV.

    Confirms thermal stability despite unlocked clocks (clock-locking not
    permitted on the RunPod container). For each (decoder, batch) we record the
    mean us/syndrome of each run and the coefficient of variation
    (std/mean across runs). A small CV (<~few %) means the unlocked clock did not
    drift over the session -- Coda's stated fallback when clocks can't be locked.

    Batches chosen: 1 (single-shot, latency-bound, most overhead-sensitive) and
    16384 (throughput, fully-amortized) -- the two regimes the figure reports."""
    runs = {}
    for w in which:
        per_batch = {}
        for b in batches:
            means = []
            recs = []
            for r in range(int(n_runs)):
                try:
                    rec = time_gpu_decoder(w, dem, batch=int(b), device=device,
                                           n_warmup=n_warmup, n_reps=n_reps,
                                           seed=seed + r)
                    means.append(rec["us_per_syndrome"])
                    recs.append(dict(run=r, us_per_syndrome=rec["us_per_syndrome"],
                                     mean_ms=rec["mean_ms"],
                                     p99_9_ms=rec["p99_9_ms"]))
                except Exception as e:
                    recs.append(dict(run=r, error=repr(e)))
            if means:
                m = float(np.mean(means))
                s = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
                cv = float(s / m) if m > 0 else None
            else:
                m = s = cv = None
            per_batch[str(b)] = dict(
                batch=int(b), n_runs=int(n_runs),
                run_means_us=means, runs=recs,
                mean_us=m, std_us=s, cv=cv,
            )
        runs[w] = per_batch
    return runs


# --------------------------------------------------------------------------- #
# Top-level driver: measure every decoder, emit the latency manifest          #
# --------------------------------------------------------------------------- #
def run_all(device="cuda", cpu_batch=DEFAULT_BATCH, gpu_batches=GPU_BATCH_SWEEP,
            seed=0, n_warmup=DEFAULT_WARMUP, n_reps=DEFAULT_REPS,
            rounds=CANON_ROUNDS, p=CANON_P, basis=CANON_BASIS,
            include_gpu=True, include_sw=True, ext_dir=None, det_beam=64,
            both_workloads=True, clock_variance_runs=3,
            both_workloads_batch=64, both_workloads_warmup=3,
            both_workloads_reps=30):
    """Measure latency for every available decoder on the canonical DEM.

    Returns the full results manifest (dict). Each decoder that cannot be timed
    (missing package, no CUDA) is recorded with a ``skipped``/``error`` note --
    never fabricated."""
    import hashlib

    circ, dem = canonical_dem(rounds=rounds, p=p, basis=basis)
    dem_sha = hashlib.sha256(str(dem).encode()).hexdigest()

    # Realistic detector syndromes from the circuit at the grid p (the fair
    # latency workload). Sampled ONCE; every CPU decoder times the SAME shots
    # (matched-protocol consistency, mirroring the LER grid). Max CPU cap drives
    # how many we need.
    max_cpu = max([cpu_batch] + list(CPU_BATCH_CAPS.values()))
    all_dets = sample_detectors(circ, max_cpu, seed=seed)

    def _cpu_dets(name):
        b = min(CPU_BATCH_CAPS.get(name, cpu_batch), all_dets.shape[0])
        return all_dets[:b]

    results = {}

    # --- CPU adapters (ldpc family, relay-bp, tesseract) -------------------- #
    from qldpc.zoo import adapters as A

    # Core ldpc + Tesseract anchor.
    cpu_specs = [
        ("BP", lambda: A.make_bp(dem)),
        ("BPOSD-0", lambda: A.make_bposd0(dem)),
        ("BPOSD-10", lambda: A.make_bposd10(dem)),
        ("BPLSD", lambda: A.make_bplsd(dem)),
        ("Tesseract", lambda: A.make_tesseract(dem, det_beam=det_beam)),
    ]
    if A.relay_bp_available():
        cpu_specs.append(("RelayBP", lambda: A.make_relay_bp(dem)))
    else:
        results["RelayBP"] = dict(decoder="RelayBP", skipped="relay-bp unavailable")

    for name, make in cpu_specs:
        try:
            adapter = make()
            rec = time_cpu_decoder(adapter, _cpu_dets(name),
                                   n_warmup=n_warmup, n_reps=n_reps)
            results[name] = rec
        except Exception as e:  # never fabricate -- record the failure
            import traceback
            results[name] = dict(decoder=name, error=repr(e),
                                 traceback=traceback.format_exc()[-800:])

    # --- Sliding-window (fair, in-process) ---------------------------------- #
    if include_sw:
        try:
            results["SlidingWindow"] = time_sliding_window(
                dem, rounds=rounds, dets=_cpu_dets("SlidingWindow"),
                ext_dir=ext_dir, n_warmup=n_warmup, n_reps=n_reps)
        except Exception as e:
            import traceback
            results["SlidingWindow"] = dict(
                decoder="SlidingWindow", error=repr(e),
                traceback=traceback.format_exc()[-800:])

    # --- GPU decoders (Triton kernel + torch baseline), batch sweep --------- #
    if include_gpu:
        cuda_ok = False
        try:
            import torch
            cuda_ok = (str(device).startswith("cuda")
                       and torch.cuda.is_available())
        except Exception:
            cuda_ok = False
        for which in ("bp_triton", "bp_gpu"):
            sweeps = []
            err = None
            for b in gpu_batches:
                try:
                    sweeps.append(time_gpu_decoder(
                        which, dem, batch=b, device=device,
                        n_warmup=n_warmup, n_reps=n_reps, seed=seed))
                except Exception as e:
                    import traceback
                    err = repr(e)
                    sweeps.append(dict(batch=int(b), error=repr(e),
                                       traceback=traceback.format_exc()[-400:]))
            # representative point = largest successful batch (best amortized
            # THROUGHPUT) -- kept as ``representative`` for cycle_time_budget
            # back-compat. single_shot = batch-1 (REAL-TIME latency, no
            # amortization) -- the headline latency story. Both reported.
            ok_sweeps = [s for s in sweeps if "error" not in s]
            rep = ok_sweeps[-1] if ok_sweeps else None
            single = next((s for s in ok_sweeps
                           if s.get("batch") == SINGLE_SHOT_BATCH), None)
            results[which] = dict(
                decoder=which,
                timing=("cuda_event" if cuda_ok else "perf_counter_cpu_fallback"),
                cuda=cuda_ok,
                batch_sweep=sweeps,
                representative=rep,        # throughput (batch 16384)
                single_shot=single,        # real-time latency (batch 1)
                error=err if rep is None else None,
            )

    # --- Both-workloads (realistic vs uniform-random) for OSD/LSD ----------- #
    both_wl = None
    if both_workloads:
        try:
            bw_dets = sample_detectors(circ, both_workloads_batch, seed=seed)
            both_wl = dict(
                kind="t9-both-workloads",
                code="[[72,12,6]] BB (BBCode default)",
                rounds=rounds, p=p, basis=basis, noise=CANON_NOISE,
                dem_sha256=dem_sha,
                batch=int(both_workloads_batch),
                discipline=dict(n_warmup=int(both_workloads_warmup),
                                n_reps=int(both_workloads_reps),
                                drop_frac=DROP_FRAC, seed=seed,
                                note=("smaller batch/reps than the main figure "
                                      "discipline: the uniform-random OSD "
                                      "worst-case is ms/shot; the inflation "
                                      "RATIO is robust at fewer reps and is the "
                                      "reported quantity (not a figure point)")),
                workloads=dict(
                    realistic=("circuit detector syndromes at the operational "
                               f"grid p={p} (sparse; the HEADLINE workload)"),
                    uniform_random=("dense p=0.5-per-bit syndromes (OSD "
                                    "worst-case; the adversarial bound)"),
                ),
                reference_note=BOTH_WORKLOADS_REFERENCE_NOTE,
                results=measure_both_workloads(
                    dem, realistic_dets=bw_dets, n_det=dem.num_detectors,
                    batch=both_workloads_batch, seed=seed,
                    n_warmup=int(both_workloads_warmup),
                    n_reps=int(both_workloads_reps)),
            )
        except Exception as e:
            import traceback
            both_wl = dict(kind="t9-both-workloads", error=repr(e),
                           traceback=traceback.format_exc()[-800:])

    # --- Variance-over-runs (unlocked-clock thermal-stability check) -------- #
    clock_var = None
    if include_gpu and clock_variance_runs and clock_variance_runs >= 1:
        cuda_ok2 = False
        try:
            import torch
            cuda_ok2 = (str(device).startswith("cuda")
                        and torch.cuda.is_available())
        except Exception:
            cuda_ok2 = False
        if cuda_ok2:
            try:
                clock_var = dict(
                    kind="t9-clock-variance",
                    purpose=("Coda fallback for 'can't lock clocks' on RunPod: "
                             "repeat the GPU bench N>=3x across the session; a "
                             "small run-to-run CV confirms no thermal drift "
                             "despite unlocked clocks."),
                    clock_state=gpu_clock_state(),
                    n_runs=int(clock_variance_runs),
                    batches=[SINGLE_SHOT_BATCH, gpu_batches[-1]],
                    runs=measure_clock_variance(
                        dem, device=device, n_runs=int(clock_variance_runs),
                        batches=(SINGLE_SHOT_BATCH, gpu_batches[-1]),
                        seed=seed, n_warmup=n_warmup, n_reps=n_reps),
                )
            except Exception as e:
                import traceback
                clock_var = dict(kind="t9-clock-variance", error=repr(e),
                                 traceback=traceback.format_exc()[-800:])

    manifest = dict(
        kind="t9-decoder-latency",
        code="[[72,12,6]] BB (BBCode default)",
        rounds=rounds, p=p, basis=basis, noise=CANON_NOISE,
        dem_n_det=dem.num_detectors, dem_n_obs=dem.num_observables,
        dem_sha256=dem_sha,
        discipline=dict(n_warmup=n_warmup, n_reps=n_reps, drop_frac=DROP_FRAC,
                        cpu_batch=cpu_batch, gpu_batches=list(gpu_batches),
                        seed=seed, tesseract_det_beam=det_beam,
                        single_shot_batch=SINGLE_SHOT_BATCH,
                        throughput_batch=gpu_batches[-1]),
        bootstrap=dict(n_resamples=BOOTSTRAP_N, alpha=BOOTSTRAP_ALPHA,
                       seed=BOOTSTRAP_SEED,
                       protocol=("percentile bootstrap on the kept per-rep "
                                 "measure window (drop-first-10% applied): "
                                 "resample-with-replacement the per-rep batch "
                                 "times, amortize each resample's mean to "
                                 "us/syndrome, take 2.5/97.5 percentiles -> "
                                 "95% CI on the MEAN per-syndrome latency")),
        sw_latency_method=(
            "(a) in-process isolated importlib load of /workspace/qldpc_ext "
            "SlidingWindowDecoder; only compiled.decode_shots is timed, the "
            "subprocess spawn (~1-1.3 s, an integration artifact) is EXCLUDED. "
            "Fair = it is the genuine per-batch decode cost."),
        env=env_manifest(),
        results=results,
    )
    # Attach the sub-artifacts so a single run produces all three JSONs.
    manifest["_both_workloads"] = both_wl
    manifest["_clock_variance"] = clock_var
    return manifest


def _print_table(manifest):
    """Print a human-readable per-decoder latency table."""
    print(f"\nT9 latency -- {manifest['code']} d={manifest['rounds']} "
          f"R={manifest['rounds']} p={manifest['p']} basis={manifest['basis']} "
          f"{manifest['noise']}")
    print(f"DEM: {manifest['dem_n_det']} det x {manifest['dem_n_obs']} obs, "
          f"sha {manifest['dem_sha256'][:12]}")
    d = manifest["discipline"]
    print(f"discipline: warmup={d['n_warmup']} reps={d['n_reps']} "
          f"drop={int(DROP_FRAC*100)}% cpu_batch={d['cpu_batch']}")
    print("-" * 92)
    print(f"{'decoder':<16}{'batch':>7}{'mean_ms':>11}{'p99.9_ms':>11}"
          f"{'us/syn':>11}{'us/syn_p99.9':>14}{'shots/s':>14}")
    print("-" * 92)

    def row(name, rec):
        if not rec or rec.get("error") or rec.get("skipped"):
            note = rec.get("error") or rec.get("skipped") if rec else "n/a"
            print(f"{name:<16}  -- {str(note)[:70]}")
            return
        print(f"{name:<16}{rec['batch']:>7}{rec['mean_ms']:>11.4f}"
              f"{rec['p99_9_ms']:>11.4f}{rec['us_per_syndrome']:>11.4f}"
              f"{rec['us_per_syndrome_p99_9']:>14.4f}"
              f"{rec['throughput_shots_per_s']:>14.1f}")

    order = ["bp_triton", "bp_gpu", "BP", "BPOSD-0", "BPOSD-10", "BPLSD",
             "RelayBP", "SlidingWindow", "Tesseract"]
    res = manifest["results"]
    for name in order:
        if name not in res:
            continue
        rec = res[name]
        if name in ("bp_triton", "bp_gpu"):
            print(f"[{name}] batch sweep (CUDA-event):")
            for s in rec.get("batch_sweep", []):
                if "error" in s:
                    print(f"  batch {s['batch']:>6}: ERROR {s['error'][:60]}")
                else:
                    print(f"  batch {s['batch']:>6}: mean {s['mean_ms']:.4f} ms "
                          f"p99.9 {s['p99_9_ms']:.4f} ms  "
                          f"{s['us_per_syndrome']:.4f} us/syn  "
                          f"{s['throughput_shots_per_s']:.0f} shots/s")
        else:
            row(name, rec)
    print("-" * 92)


def main(argv=None):
    ap = argparse.ArgumentParser(description="T9 decoder latency harness")
    ap.add_argument("--out", default=os.path.join(_HERE, "latency_results.json"))
    ap.add_argument("--both-workloads-out",
                    default=os.path.join(_HERE, "both_workloads.json"))
    ap.add_argument("--clock-variance-out",
                    default=os.path.join(_HERE, "clock_variance.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu-batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--gpu-batches", type=int, nargs="+",
                    default=list(GPU_BATCH_SWEEP))
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=float, default=CANON_P)
    ap.add_argument("--basis", default=CANON_BASIS)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--no-sw", action="store_true")
    ap.add_argument("--no-both-workloads", action="store_true")
    ap.add_argument("--clock-variance-runs", type=int, default=3)
    ap.add_argument("--both-workloads-batch", type=int, default=64)
    ap.add_argument("--both-workloads-warmup", type=int, default=3)
    ap.add_argument("--both-workloads-reps", type=int, default=30)
    ap.add_argument("--ext-dir", default=None)
    ap.add_argument("--det-beam", type=int, default=64)
    args = ap.parse_args(argv)

    manifest = run_all(
        device=args.device, cpu_batch=args.cpu_batch,
        gpu_batches=tuple(args.gpu_batches), seed=args.seed,
        n_warmup=args.warmup, n_reps=args.reps, p=args.p, basis=args.basis,
        include_gpu=not args.no_gpu, include_sw=not args.no_sw,
        ext_dir=args.ext_dir, det_beam=args.det_beam,
        both_workloads=not args.no_both_workloads,
        clock_variance_runs=args.clock_variance_runs,
        both_workloads_batch=args.both_workloads_batch,
        both_workloads_warmup=args.both_workloads_warmup,
        both_workloads_reps=args.both_workloads_reps)

    # Split the sub-artifacts into their own committed JSON files.
    both_wl = manifest.pop("_both_workloads", None)
    clock_var = manifest.pop("_clock_variance", None)

    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    _print_table(manifest)
    print(f"\nwrote {args.out}")

    if both_wl is not None:
        with open(args.both_workloads_out, "w") as f:
            json.dump(both_wl, f, indent=2)
        print(f"wrote {args.both_workloads_out}")
        _print_both_workloads(both_wl)
    if clock_var is not None:
        with open(args.clock_variance_out, "w") as f:
            json.dump(clock_var, f, indent=2)
        print(f"wrote {args.clock_variance_out}")
        _print_clock_variance(clock_var)
    return manifest


def _print_both_workloads(bw):
    if not bw or bw.get("error"):
        print(f"both-workloads: {bw.get('error') if bw else 'n/a'}")
        return
    print("\nboth-workloads (realistic vs uniform-random; OSD/LSD):")
    print(f"{'decoder':<12}{'realistic_us':>15}{'uniform_us':>15}"
          f"{'inflation_x':>14}")
    for name, rec in bw.get("results", {}).items():
        if rec.get("error"):
            print(f"{name:<12}  ERROR {rec['error'][:50]}")
            continue
        r = rec["realistic"]["us_per_syndrome"]
        u = rec["uniform_random"]["us_per_syndrome"]
        infl = rec.get("inflation_x")
        print(f"{name:<12}{r:>15.4f}{u:>15.4f}"
              f"{(infl if infl else 0):>14.2f}")


def _print_clock_variance(cv):
    if not cv or cv.get("error"):
        print(f"clock-variance: {cv.get('error') if cv else 'n/a'}")
        return
    cs = cv.get("clock_state", {})
    print(f"\nclock-variance ({cv.get('n_runs')} runs; clocks_locked="
          f"{cs.get('clocks_locked')}, app_clk={cs.get('applications_clock_mhz')} "
          f"MHz, gr_clk={cs.get('gr_clock_mhz')} MHz):")
    print(f"{'decoder':<12}{'batch':>7}{'mean_us':>14}{'cv':>10}")
    for which, per_batch in cv.get("runs", {}).items():
        for b, rec in per_batch.items():
            m = rec.get("mean_us")
            c = rec.get("cv")
            print(f"{which:<12}{rec['batch']:>7}"
                  f"{(m if m else 0):>14.4f}"
                  f"{(c if c is not None else float('nan')):>10.4f}")


if __name__ == "__main__":
    main()
