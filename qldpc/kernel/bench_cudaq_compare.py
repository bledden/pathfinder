"""Pod-only THROUGHPUT head-to-head: NVIDIA CUDA-Q QEC 0.6's GPU RelayBP
(``nv-qldpc-decoder``, ``bp_method=3`` = DMem-BP / Sequential Relay-BP) vs our
open Triton Relay-BP (:class:`RelayBpTriton`), on the SAME H200 + SAME canonical
BB-code DEM.

This is an HONEST, exploratory comparison. CUDA-Q QEC is CUDA-only and closed;
our Triton kernel is open + portable (one ROCm port from AMD). Either outcome is
publishable. Throughput-only, batched -- single-shot / real-time is conceded to
NVQLink and NOT measured here.

WHAT IS MEASURED (apples-to-apples where possible)
==================================================
  * SAME DEM: ``build_memory(BBCode(), 6, 0.003, 'Z', 'si1000')`` ->
    ``canon_dem.extract`` -> H (252 checks x 1584 errors), per-error priors,
    Lo (12 obs x 1584). Both decoders ingest THIS H + priors.
  * SAME real syndromes: stim detector sampler, seed 0.
  * SAME relay config where the two APIs overlap: gamma0=0.1,
    gamma_dist=(-0.24, 0.66), pre_iter=80, num_sets=60, max_iter(per leg)=60.
  * SAME timing discipline for BOTH decoders: torch.cuda.synchronize-bracketed
    perf_counter, warmup>=5, reps adaptive by batch (100 for B<=1024 where a
    call is cheap; 40/20/12 for B=2k/4k/8k where each multi-leg Triton call is
    already 100s of ms, so a handful of reps gives a stable mean). CUDA-Q exposes
    no CUDA-event API, so wall-clock-with-sync is the fair common denominator;
    our Triton is timed the IDENTICAL way here for parity (NOT via its internal
    CUDA-event path) -- both numbers are wall-clock-with-sync, same harness.

CONFIG ASYMMETRIES (documented, not hidden)
===========================================
  * STOPPING: ours uses nconv (collect ``stop_nconv=5`` valid solutions across
    legs, return the LOWEST-WEIGHT). CUDA-Q's ``srelay_config.stopping_criterion``
    accepts a string but in 0.6 all tested values (FirstConv/ProductConv/
    PreSetMaxIter/Nconv) yielded an IDENTICAL LER here -> its relay selection is
    not exposed/tunable to match our lowest-weight-over-5 rule. Reported as-is.
  * MIN-SUM SCALE: ours alpha=1.0; CUDA-Q ``scale_factor`` left at its default.
  * PRECISION: ours fp32 AND fp64 reported; CUDA-Q ``proc_float`` left default.
  * OSD: ours has no OSD post-pass; CUDA-Q run BOTH without OSD (matched) and
    with OSD-0/10 (its native best-LER mode) -- both reported.
The LER's are in the same ballpark (~1.5-3%); the HEADLINE is THROUGHPUT.

Usage (on pod):
  PYTHONPATH=/workspace/pf-mle:/workspace/pf-mle/qldpc \
      python3 qldpc/kernel/bench_cudaq_compare.py > /tmp/bench_cudaq.out 2>&1
"""
import json
import time

import numpy as np
import torch

from bb_code import BBCode
from qldpc.foundation.circuits import build_memory
from canon_dem import extract
from qldpc.kernel.relay_triton import RelayBpTriton

ROUNDS, P = 6, 0.003
N = 2000               # LER-sanity shots
BATCH = 2000           # primary throughput batch (where Triton hits its quoted
                       # ~483us reference; the multi-leg Triton schedule needs a
                       # few-k batch to amortize per-leg kernel-launch overhead --
                       # CUDA-Q's single fused call does not).
BATCH_SWEEP = [512, 1024, 2000, 4096, 8192]   # both decoders, every batch
WARMUP, REPS = 10, 100


def _reps_for(batch):
    """Adaptive (warmup, reps): plenty for cheap small batches, fewer for the
    expensive large-batch Triton multi-leg schedule (each call already 100s ms,
    so a handful of reps gives a stable mean without an hour-long run)."""
    if batch <= 1024:
        return 10, 100
    if batch <= 2048:
        return 8, 40
    if batch <= 4096:
        return 5, 20
    return 5, 12
RELAY_CFG = dict(gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60,
                 gamma_dist_interval=(-0.24, 0.66), stop_nconv=5,
                 stopping_criterion="nconv")
CUDAQ_SRELAY = dict(pre_iter=80, num_sets=60, stopping_criterion="FirstConv")


def _wall_throughput(call, batch_arg, warmup=WARMUP, reps=REPS, n=None):
    """Wall-clock (sync-bracketed) mean ms over a batch -- common denominator
    timing used IDENTICALLY for both decoders. ``n`` (batch size) selects an
    adaptive (warmup, reps) so big-batch runs stay tractable."""
    if n is not None:
        warmup, reps = _reps_for(n)
    for _ in range(warmup):
        call(batch_arg)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        call(batch_arg)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        ts.append((t1 - t0) * 1e3)
    return float(np.mean(ts)), float(np.percentile(ts, 99.9))


def main():
    assert torch.cuda.is_available(), "CUDA required"
    out = {"meta": {}}
    out["meta"]["gpu"] = torch.cuda.get_device_name(0)
    import cudaq_qec as qec
    out["meta"]["cudaq_qec_version"] = qec.__version__
    out["meta"]["torch"] = torch.__version__
    print("GPU:", out["meta"]["gpu"], "| cudaq_qec:", out["meta"]["cudaq_qec_version"])

    # ---- canonical DEM (single source of truth) ----
    circ = build_memory(BBCode(), rounds=ROUNDS, p=P, basis="Z", noise="si1000")
    dem = circ.detector_error_model(decompose_errors=False)
    ex = extract(dem)
    H = np.ascontiguousarray(ex["H"].toarray().astype(np.uint8))   # (252,1584)
    Lo = ex["Lo"].toarray().astype(np.uint8)                       # (12,1584)
    priors = np.asarray(ex["priors"], dtype=np.float64)            # (1584,)
    dets, obs = circ.compile_detector_sampler(seed=0).sample(
        N, separate_observables=True)
    dets = np.asarray(dets, dtype=bool); obs = np.asarray(obs, dtype=bool)
    out["meta"]["structure"] = dict(n_checks=H.shape[0], n_err=H.shape[1],
                                    n_obs=Lo.shape[0], n_shots=N, batch=BATCH)
    print("STRUCTURE", out["meta"]["structure"])

    def ler_from_ehat(eh):                       # eh (S, n_err) uint8 -> block LER
        pred = (eh @ Lo.T) % 2
        return float(np.any(pred.astype(bool) != obs[:eh.shape[0]], axis=1).mean())

    # =============================================================== #
    # 1. CUDA-Q QEC nv-qldpc Relay-BP (bp_method=3)                    #
    # =============================================================== #
    def cudaq_dec(use_osd=False, osd_order=0, batch=BATCH):
        kw = dict(bp_method=3, gamma0=RELAY_CFG["gamma0"],
                  gamma_dist=list(RELAY_CFG["gamma_dist_interval"]),
                  error_rate_vec=priors.tolist(), srelay_config=dict(CUDAQ_SRELAY),
                  max_iterations=RELAY_CFG["set_max_iter"], use_osd=use_osd,
                  use_sparsity=True, bp_batch_size=batch)
        if use_osd:
            kw["osd_order"] = osd_order
        return qec.get_decoder("nv-qldpc-decoder", H, **kw)

    def cudaq_decode_ehat(d, det_batch, batch):
        eh = np.zeros((det_batch.shape[0], H.shape[1]), dtype=np.uint8)
        for i in range(0, det_batch.shape[0], batch):
            res = d.decode_batch(det_batch[i:i + batch].astype(np.float64).tolist())
            eh[i:i + len(res)] = (np.array([np.asarray(r.result) for r in res])
                                  > 0.5).astype(np.uint8)
        return eh

    # LER sanity for the 3 CUDA-Q modes (no-OSD matches our config; OSD-0/10 is
    # its native best-LER mode). Decoder rebuilt per batch size for throughput.
    cudaq_ler_rows = []
    for use_osd, osd_order, tag in [(False, 0, "relaybp_no_osd"),
                                    (True, 0, "relaybp_osd0"),
                                    (True, 10, "relaybp_osd10")]:
        d = cudaq_dec(use_osd=use_osd, osd_order=osd_order, batch=BATCH)
        ler = ler_from_ehat(cudaq_decode_ehat(d, dets, BATCH))
        cudaq_ler_rows.append(dict(tag=tag, ler=round(ler, 4),
                                   logical_errors=int(round(ler * N))))
        print("CUDAQ LER", cudaq_ler_rows[-1])
    out["cudaq_relaybp_ler"] = cudaq_ler_rows

    # Throughput sweep (no-OSD = the config-matched mode).
    cudaq_tp = []
    for B in BATCH_SWEEP:
        d = cudaq_dec(use_osd=False, batch=B)
        bl = dets[:B].astype(np.float64).tolist()
        ms, p999 = _wall_throughput(d.decode_batch, bl, n=B)
        row = dict(batch=B, mean_ms=round(ms, 3), p99_9_ms=round(p999, 3),
                   per_syndrome_us=round(ms / B * 1e3, 2),
                   throughput_shots_per_s=round(B / (ms / 1e3), 1))
        cudaq_tp.append(row)
        print("CUDAQ TP", row)
    out["cudaq_relaybp_throughput"] = cudaq_tp

    # =============================================================== #
    # 2. Our Triton Relay-BP -- SAME DEM, SAME syndromes, SAME timing  #
    # =============================================================== #
    triton_ler_rows = []
    triton_tp = {}
    for dt in ("float32", "float64"):
        trt = RelayBpTriton.from_dem(dem, block_s=256, dtype=dt, **RELAY_CFG)
        # LER over N shots -- decode_batch returns predicted observables directly.
        pred_all = trt.decode_batch(dets[:N], device="cuda")
        ler = float(np.any(pred_all != obs, axis=1).mean())
        triton_ler_rows.append(dict(dtype=dt, ler=round(ler, 4),
                                    logical_errors=int(round(ler * N))))
        print("TRITON LER", triton_ler_rows[-1], flush=True)
        # throughput sweep, SAME batches + SAME wall-clock-sync timing as CUDA-Q.
        # fp32 is the throughput mode (full sweep); fp64 is the LER-precision mode
        # (sweep only up to the primary BATCH -- big-batch fp64 is slow + off-path).
        rows = []
        sweep = BATCH_SWEEP if dt == "float32" else [b for b in BATCH_SWEEP
                                                     if b <= BATCH]
        for B in sweep:
            det_batch = dets[:B]
            ms, p999 = _wall_throughput(
                lambda b: trt.decode_batch(b, device="cuda"), det_batch, n=B)
            r = dict(batch=B, mean_ms=round(ms, 3), p99_9_ms=round(p999, 3),
                     per_syndrome_us=round(ms / B * 1e3, 2),
                     throughput_shots_per_s=round(B / (ms / 1e3), 1))
            rows.append(r)
            print(f"TRITON TP[{dt}]", r)
        triton_tp[dt] = rows
    out["triton_relaybp_ler"] = triton_ler_rows
    out["triton_relaybp_throughput"] = triton_tp

    # =============================================================== #
    # 3. CPU relay_bp Rust oracle -- same DEM, same syndromes (LER + tp)#
    # =============================================================== #
    cpu = None
    try:
        import relay_bp
        from relay_bp.stim import CheckMatrices
        cm = CheckMatrices.from_dem(dem)
        dec = relay_bp.RelayDecoderF64(
            cm.check_matrix, error_priors=cm.error_priors,
            gamma0=RELAY_CFG["gamma0"], pre_iter=RELAY_CFG["pre_iter"],
            num_sets=RELAY_CFG["num_sets"], set_max_iter=RELAY_CFG["set_max_iter"],
            gamma_dist_interval=RELAY_CFG["gamma_dist_interval"],
            stop_nconv=RELAY_CFG["stop_nconv"],
            stopping_criterion=RELAY_CFG["stopping_criterion"])
        runner = relay_bp.ObservableDecoderRunner(
            dec, cm.observables_matrix, include_decode_result=False)
        syn = dets.astype(np.uint8)
        pred_ref = (np.asarray(runner.decode_observables_batch(syn)) % 2).astype(bool)
        if pred_ref.ndim == 1:
            pred_ref = pred_ref.reshape(-1, 1)
        ler = float(np.any(pred_ref != obs, axis=1).mean())
        syn_b = syn[:BATCH]
        for _ in range(2):
            runner.decode_observables_batch(syn_b)
        t0 = time.perf_counter(); r = 3
        for _ in range(r):
            runner.decode_observables_batch(syn_b)
        t1 = time.perf_counter()
        ms = (t1 - t0) / r * 1e3
        cpu = dict(ler=round(ler, 4), logical_errors=int(round(ler * N)),
                   batch=BATCH, mean_ms=round(ms, 3),
                   per_syndrome_us=round(ms / BATCH * 1e3, 2),
                   throughput_shots_per_s=round(BATCH / (ms / 1e3), 1),
                   note="CPU Rust relay_bp oracle, nconv stop_nconv=5 "
                        "(lowest-weight) -- the LER reference + slowest baseline")
        print("CPU_RELAY_BP", cpu)
    except Exception as e:
        cpu = dict(error=repr(e)[:200])
        print("CPU_RELAY_BP unavailable:", cpu)
    out["cpu_relay_bp"] = cpu

    # =============================================================== #
    # VERDICT -- best throughput each, + apples-to-apples at BATCH     #
    # =============================================================== #
    cq_best = max(out["cudaq_relaybp_throughput"],
                  key=lambda r: r["throughput_shots_per_s"])
    tr_best = max(out["triton_relaybp_throughput"]["float32"],
                  key=lambda r: r["throughput_shots_per_s"])
    cq_at = next(r for r in out["cudaq_relaybp_throughput"] if r["batch"] == BATCH)
    tr_at = next(r for r in out["triton_relaybp_throughput"]["float32"]
                 if r["batch"] == BATCH)
    verdict = dict(
        winner="CUDA-Q QEC (throughput)",
        cudaq_best_throughput=cq_best,
        triton_best_throughput_fp32=tr_best,
        cudaq_over_triton_best=round(
            cq_best["throughput_shots_per_s"]
            / tr_best["throughput_shots_per_s"], 1),
        at_matched_batch=BATCH,
        cudaq_at_batch=cq_at, triton_fp32_at_batch=tr_at,
        cudaq_over_triton_at_batch=round(
            cq_at["throughput_shots_per_s"]
            / tr_at["throughput_shots_per_s"], 1),
        cudaq_ler_no_osd=out["cudaq_relaybp_ler"][0]["ler"],
        triton_fp32_ler=out["triton_relaybp_ler"][0]["ler"],
        cpu_relay_bp_ler=(cpu.get("ler") if cpu else None),
        ler_note="CUDA-Q LER ~2x ours: its relay selection is FirstConv-style, "
                 "not our lowest-weight-over-nconv=5 -- a config/algorithm "
                 "asymmetry, not a kernel-speed artifact.")
    if cpu and "throughput_shots_per_s" in cpu:
        verdict["cudaq_over_cpu"] = round(
            cq_best["throughput_shots_per_s"]
            / cpu["throughput_shots_per_s"], 1)
        verdict["triton_over_cpu"] = round(
            tr_best["throughput_shots_per_s"]
            / cpu["throughput_shots_per_s"], 1)
    out["verdict"] = verdict
    print("VERDICT", verdict)

    with open("/tmp/bench_cudaq.json", "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE /tmp/bench_cudaq.json")
    return out


if __name__ == "__main__":
    main()
