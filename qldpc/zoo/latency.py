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
# we sweep a few batches for the GPU/Triton path so the figure can show the
# batch dependence and pick a representative cell.
DEFAULT_BATCH = 1024
GPU_BATCH_SWEEP = (256, 1024, 4096, 16384)

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


def summarize(times_ms, batch):
    """Reduce kept per-batch times (ms) to the reported latency record."""
    mean_ms = float(np.mean(times_ms))
    p999_ms = float(np.percentile(times_ms, 99.9))
    median_ms = float(np.median(times_ms))
    thr = float(batch / (mean_ms / 1e3)) if mean_ms > 0 else 0.0
    us_per_syn = float(mean_ms * 1e3 / batch) if batch > 0 else 0.0
    us_per_syn_p999 = float(p999_ms * 1e3 / batch) if batch > 0 else 0.0
    return dict(
        batch=int(batch),
        n_kept=int(len(times_ms)),
        mean_ms=mean_ms,
        median_ms=median_ms,
        p99_9_ms=p999_ms,
        throughput_shots_per_s=thr,
        us_per_syndrome=us_per_syn,
        us_per_syndrome_p99_9=us_per_syn_p999,
    )


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
    mean_ms = res["mean_ms"]
    p999_ms = res["p99_9_ms"]
    rec = dict(
        decoder=which,
        timing=("cuda_event" if res.get("used_cuda_events") else "perf_counter"),
        batch=int(batch),
        n_kept=int(res["n_iter"]),
        mean_ms=mean_ms,
        p99_9_ms=p999_ms,
        throughput_shots_per_s=res["throughput_shots_per_s"],
        us_per_syndrome=float(mean_ms * 1e3 / batch) if batch else 0.0,
        us_per_syndrome_p99_9=float(p999_ms * 1e3 / batch) if batch else 0.0,
        device=res["device"],
        max_iter=res["max_iter"],
        used_cuda_events=bool(res.get("used_cuda_events")),
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
        else:
            env["gpu_name"] = None
            env["gpu_driver"] = None
    except Exception:
        env["cuda"] = None
        env["gpu_name"] = None
        env["gpu_driver"] = None
    return env


# --------------------------------------------------------------------------- #
# Top-level driver: measure every decoder, emit the latency manifest          #
# --------------------------------------------------------------------------- #
def run_all(device="cuda", cpu_batch=DEFAULT_BATCH, gpu_batches=GPU_BATCH_SWEEP,
            seed=0, n_warmup=DEFAULT_WARMUP, n_reps=DEFAULT_REPS,
            rounds=CANON_ROUNDS, p=CANON_P, basis=CANON_BASIS,
            include_gpu=True, include_sw=True, ext_dir=None, det_beam=64):
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
            # representative point: largest successful batch (best amortized)
            ok_sweeps = [s for s in sweeps if "error" not in s]
            rep = ok_sweeps[-1] if ok_sweeps else None
            results[which] = dict(
                decoder=which,
                timing=("cuda_event" if cuda_ok else "perf_counter_cpu_fallback"),
                cuda=cuda_ok,
                batch_sweep=sweeps,
                representative=rep,
                error=err if rep is None else None,
            )

    manifest = dict(
        kind="t9-decoder-latency",
        code="[[72,12,6]] BB (BBCode default)",
        rounds=rounds, p=p, basis=basis, noise=CANON_NOISE,
        dem_n_det=dem.num_detectors, dem_n_obs=dem.num_observables,
        dem_sha256=dem_sha,
        discipline=dict(n_warmup=n_warmup, n_reps=n_reps, drop_frac=DROP_FRAC,
                        cpu_batch=cpu_batch, gpu_batches=list(gpu_batches),
                        seed=seed, tesseract_det_beam=det_beam),
        sw_latency_method=(
            "(a) in-process isolated importlib load of /workspace/qldpc_ext "
            "SlidingWindowDecoder; only compiled.decode_shots is timed, the "
            "subprocess spawn (~1-1.3 s, an integration artifact) is EXCLUDED. "
            "Fair = it is the genuine per-batch decode cost."),
        env=env_manifest(),
        results=results,
    )
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
    ap.add_argument("--ext-dir", default=None)
    ap.add_argument("--det-beam", type=int, default=64)
    args = ap.parse_args(argv)

    manifest = run_all(
        device=args.device, cpu_batch=args.cpu_batch,
        gpu_batches=tuple(args.gpu_batches), seed=args.seed,
        n_warmup=args.warmup, n_reps=args.reps, p=args.p, basis=args.basis,
        include_gpu=not args.no_gpu, include_sw=not args.no_sw,
        ext_dir=args.ext_dir, det_beam=args.det_beam)

    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    _print_table(manifest)
    print(f"\nwrote {args.out}")
    return manifest


if __name__ == "__main__":
    main()
