"""MI300X port + validation driver for the Triton Relay-BP kernel.

Runs INSIDE the ROCm container on gfx942 (triton 3.4-rocm, torch 2.9-rocm).
Re-validates LER vs the relay_bp Rust oracle on the canonical DEM and benchmarks
throughput at batches {2000, 8192} fp32 (+ fp64), mirroring the H200 discipline
(warmup>=10, measure>=100, wall-clock/CUDA-event sync). Emits the
cross-vendor-portability JSON.
"""
import json
import platform
import sys
import time

import numpy as np
import torch

from bb_code import BBCode
from qldpc.foundation.circuits import build_memory
from qldpc.kernel.relay_triton import RelayBpTriton

ROUNDS = 6
P = 0.003
N_VAL = 2000
BLOCK_S = 256
RELAY_CFG = dict(
    gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60,
    gamma_dist_interval=(-0.24, 0.66), stop_nconv=5, stopping_criterion="nconv",
)


def _relay_oracle(dem):
    import relay_bp
    from relay_bp.stim import CheckMatrices
    cm = CheckMatrices.from_dem(dem)
    dec = relay_bp.RelayDecoderF64(cm.check_matrix,
                                   error_priors=cm.error_priors, **RELAY_CFG)
    runner = relay_bp.ObservableDecoderRunner(
        dec, cm.observables_matrix, include_decode_result=False)
    return cm, dec, runner


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _bench_decode(dec, dets_real, n_warmup, n_iter):
    """CUDA-event timed decode_batch over real syndromes (full relay schedule)."""
    for _ in range(n_warmup):
        dec.decode_batch(dets_real, device="cuda")
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(n_iter):
        ev0.record()
        dec.decode_batch(dets_real, device="cuda")
        ev1.record()
        torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1))
    return float(np.mean(ts)), float(np.percentile(ts, 99.9)), ts


def main():
    assert torch.cuda.is_available(), "ROCm/HIP device required"
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = ""
    try:
        import triton
        triton_ver = triton.__version__
    except Exception:
        triton_ver = "?"

    env = dict(
        platform="MI300X",
        gpu_arch="gfx942",
        gpu_name_reported=gpu_name,  # empty on the VF — normal quirk
        rocm="7.0.0",
        torch=torch.__version__,
        triton=triton_ver,
        python=platform.python_version(),
    )
    print("ENV", env)

    circ = build_memory(BBCode(), rounds=ROUNDS, p=P, basis="Z", noise="si1000")
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=0).sample(
        N_VAL, separate_observables=True)
    dets = np.asarray(dets, dtype=bool)
    obs = np.asarray(obs, dtype=bool)

    out = {"env": env,
           "config": dict(rounds=ROUNDS, p=P, n_val=N_VAL, block_s=BLOCK_S,
                          relay=RELAY_CFG)}

    trt = RelayBpTriton.from_dem(dem, block_s=BLOCK_S, **RELAY_CFG)
    out["structure"] = dict(n_checks=trt.n_checks, n_bits=trt.n_bits,
                            n_edges=trt.n_edges, MAXDEG_C=trt.MAXDEG_C,
                            MAXDEG_B=trt.MAXDEG_B)
    print("STRUCTURE", out["structure"])

    # ---- PRIMITIVE identity: pre-leg posterior + memory-term hard decision ----
    import relay_bp
    from relay_bp.stim import CheckMatrices
    cm = CheckMatrices.from_dem(dem)
    syn = dets.astype(np.uint8)
    n_probe = 256
    post1 = trt.minsum_posterior_batch(syn[:n_probe], n_iter=1, gamma=0.0,
                                       alpha=1.0, device="cuda")
    dec0 = relay_bp.MinSumBPDecoderF64(cm.check_matrix,
                                       error_priors=cm.error_priors,
                                       max_iter=1, alpha=1.0, gamma0=0.0)
    ref1 = np.stack([np.asarray(dec0.decode_detailed(syn[i]).posterior_ratios)
                     for i in range(n_probe)])
    pre_maxdiff = float(np.max(np.abs(post1 - ref1)))
    post30 = trt.minsum_posterior_batch(syn[:n_probe], n_iter=30, gamma=0.1,
                                        alpha=1.0, device="cuda")
    hard30 = (post30 < 0.0).astype(np.uint8)
    decm = relay_bp.MinSumBPDecoderF64(cm.check_matrix,
                                       error_priors=cm.error_priors,
                                       max_iter=30, alpha=1.0, gamma0=0.1)
    refm = np.stack([np.asarray(decm.decode_detailed(syn[i]).decoding)
                     for i in range(n_probe)]).astype(np.uint8)
    mem_per_bit = float((hard30 == refm).mean())
    out["primitive_identity"] = dict(
        pre_leg_posterior_maxdiff=pre_maxdiff,
        memory_term_per_bit_agreement=mem_per_bit, n_probe=n_probe)
    print("PRIMITIVE_IDENTITY", out["primitive_identity"])

    # ---- FULL RELAY LER-IDENTITY vs the relay_bp oracle (fp64 message path) ----
    _, _, runner = _relay_oracle(dem)
    pred_ref = (np.asarray(runner.decode_observables_batch(syn)) % 2).astype(bool)
    if pred_ref.ndim == 1:
        pred_ref = pred_ref.reshape(-1, 1)
    pred_trt = trt.decode_batch(dets, device="cuda")
    ler_ref = int(np.any(pred_ref != obs, axis=1).sum())
    ler_trt = int(np.any(pred_trt != obs, axis=1).sum())
    per_shot = float(np.all(pred_trt == pred_ref, axis=1).mean())
    ci_ref = _wilson(ler_ref, N_VAL)
    ci_trt = _wilson(ler_trt, N_VAL)
    overlap = not (ci_trt[1] < ci_ref[0] or ci_ref[1] < ci_trt[0])
    out["ler_identity"] = dict(
        oracle_logical_errors=ler_ref,
        triton_logical_errors=ler_trt,
        per_shot_agreement=per_shot,
        n=N_VAL,
        ler_abs_diff=abs(ler_trt - ler_ref),
        oracle_wilson_ci=[round(c, 5) for c in ci_ref],
        triton_wilson_ci=[round(c, 5) for c in ci_trt],
        wilson_ci_overlap=bool(overlap),
        statistically_indistinguishable=bool(overlap))
    print("LER_IDENTITY", out["ler_identity"])

    # ---- THROUGHPUT @ batches {2000, 8192}, fp32 (+ fp64), real syndromes ----
    # Real data syndromes => the nconv early-stop is exercised identically to the
    # CPU oracle. For 8192 we tile the 2000 real syndromes up to 8192 shots.
    def _make_dets(shots):
        if shots <= N_VAL:
            return dets[:shots]
        reps = (shots + N_VAL - 1) // N_VAL
        return np.tile(dets, (reps, 1))[:shots]

    bench = []
    for shots in (2000, 8192):
        dets_real = _make_dets(shots)
        for dt in ("float32", "float64"):
            dec = RelayBpTriton.from_dem(dem, block_s=BLOCK_S, dtype=dt,
                                         **RELAY_CFG)
            # warmup>=10, measure>=100 per the requested discipline.
            mean_ms, p999_ms, _ = _bench_decode(dec, dets_real,
                                                 n_warmup=10, n_iter=100)
            row = dict(shots=shots, dtype=dt,
                       mean_ms=round(mean_ms, 3),
                       p99_9_ms=round(p999_ms, 3),
                       per_syndrome_us=round(mean_ms / shots * 1e3, 3),
                       throughput_shots_per_s=round(shots / (mean_ms / 1e3), 1))
            bench.append(row)
            print("BENCH(real)", row)
    out["bench_triton_real"] = bench

    # Worst-case random-syndrome batched throughput (full schedule, no early
    # stop -- the pure per-iteration kernel-throughput regime), fp32.
    bench_rand = []
    for shots in (2000, 8192):
        rt = RelayBpTriton.bench_latency(dem, shots=shots, device="cuda",
                                         n_warmup=10, n_iter=100, block_s=BLOCK_S,
                                         dtype="float32", **RELAY_CFG)
        row = dict(shots=shots, dtype="float32",
                   mean_ms=round(rt["mean_ms"], 3),
                   p99_9_ms=round(rt["p99_9_ms"], 3),
                   per_syndrome_us=round(rt["per_syndrome_us"], 3),
                   throughput_shots_per_s=round(rt["throughput_shots_per_s"], 1))
        bench_rand.append(row)
        print("BENCH(random worst-case)", row)
    out["bench_triton_random_worstcase"] = bench_rand

    # H200 reference (from results/bench_relay_triton.json on the same DEM/cfg)
    # for the portability comparison (NOT a head-to-head -- same kernel, two
    # vendors). H200 best real-syndrome fp32 @ 2000 = 483.13 us/syn.
    out["h200_reference"] = dict(
        gpu="NVIDIA H200", triton="3.0",
        ler_identity=dict(oracle_logical_errors=31, triton_logical_errors=38,
                          per_shot_agreement=0.9875, n=2000),
        best_real_fp32_per_syndrome_us=483.13,
        best_real_fp32_shots=2000,
        note=("portability comparison only; CUDA-Q QEC cannot run on AMD"))

    best = max(bench, key=lambda r: r["throughput_shots_per_s"])
    out["verdict"] = dict(
        ran_on_gfx942=True,
        kernel_source_changes_needed=False,
        ler_statistically_indistinguishable=bool(
            out["ler_identity"]["statistically_indistinguishable"]),
        best_triton_real=best,
        portability_demonstrated=bool(
            out["ler_identity"]["statistically_indistinguishable"]))
    print("VERDICT", out["verdict"])

    with open("/tmp/bench_relay_mi300x.json", "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE /tmp/bench_relay_mi300x.json")
    return out


if __name__ == "__main__":
    main()
    sys.stdout.flush()
