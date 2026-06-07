"""Pod-only: validate + benchmark the Triton BP kernel vs the torch baseline.

Runs on the H200 (requires triton + CUDA). Emits a JSON note with:
  - one-iteration hard-decision agreement (Triton vs CPU baseline),
  - full-decode per-shot agreement + LER (Triton vs CPU baseline vs torch GPU),
  - head-to-head latency/throughput at matched shots (256/1024/4096/16384),
    max_iter=30, using each class's own bench_latency (same timing protocol).

Usage (on pod):
  PYTHONPATH=/workspace/pf-mle:/workspace/pf-mle/qldpc python3 \
      qldpc/kernel/bench_triton.py > /tmp/bench_triton.out 2>&1
"""
import json
import sys
import numpy as np
import torch

from bb_code import BBCode
from qldpc.foundation.circuits import build_memory
from qldpc.kernel.bp_baseline import BpBaseline
from qldpc.kernel.bp_gpu import BpGpu
from qldpc.kernel.bp_triton import BpTriton

MS = 0.625
ROUNDS = 6
P = 0.003
SHOTS_GRID = [256, 1024, 4096, 16384]
MAX_ITER = 30
BLOCK_S = 256


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("GPU:", torch.cuda.get_device_name(0))
    circ = build_memory(BBCode(), rounds=ROUNDS, p=P, basis="Z", noise="si1000")
    dem = circ.detector_error_model(decompose_errors=False)
    n_val = 2000
    dets, obs = circ.compile_detector_sampler(seed=0).sample(
        n_val, separate_observables=True)
    dets = np.asarray(dets, dtype=bool)
    obs = np.asarray(obs, dtype=bool)

    out = {"config": dict(rounds=ROUNDS, p=P, ms=MS, max_iter=MAX_ITER,
                          block_s=BLOCK_S, n_val=n_val)}

    # ---- structure ----
    trt = BpTriton.from_dem(dem, max_iter=MAX_ITER, ms_scaling_factor=MS,
                            block_s=BLOCK_S)
    out["structure"] = dict(n_checks=trt.n_checks, n_bits=trt.n_bits,
                            n_edges=trt.n_edges, MAXDEG_C=trt.MAXDEG_C,
                            MAXDEG_B=trt.MAXDEG_B)
    print("STRUCTURE", out["structure"])

    # ---- one-iteration hard agreement (Triton vs CPU baseline) ----
    cpu1 = BpBaseline.from_dem(dem, max_iter=1, ms_scaling_factor=MS)
    trt1 = BpTriton.from_dem(dem, max_iter=1, ms_scaling_factor=MS,
                             block_s=BLOCK_S)
    syn = dets.astype(np.uint8)
    post1 = trt1.run_iterations_batch(syn, n_iter=1, device="cuda")
    hard_trt1 = (post1 < 0.0).astype(np.uint8)
    hard_cpu1 = np.empty_like(hard_trt1)
    for i in range(syn.shape[0]):
        hard_cpu1[i] = cpu1.run_iterations(syn[i], n_iter=1)["hard"]
    agree1 = float((hard_trt1 == hard_cpu1).mean())
    out["one_iter_hard_agreement"] = agree1
    print("ONE_ITER hard agreement (Triton vs CPU):", agree1)

    # ---- full decode agreement + LER ----
    cpu = BpBaseline.from_dem(dem, max_iter=MAX_ITER, ms_scaling_factor=MS)
    gpu = BpGpu.from_dem(dem, max_iter=MAX_ITER, ms_scaling_factor=MS)
    pred_cpu = cpu.decode_batch(dets)
    pred_gpu = gpu.decode_batch(dets, device="cuda")
    pred_trt = trt.decode_batch(dets, device="cuda")
    frac_trt_cpu = float(np.all(pred_trt == pred_cpu, axis=1).mean())
    frac_trt_gpu = float(np.all(pred_trt == pred_gpu, axis=1).mean())
    ler_cpu = int(np.any(pred_cpu != obs, axis=1).sum())
    ler_gpu = int(np.any(pred_gpu != obs, axis=1).sum())
    ler_trt = int(np.any(pred_trt != obs, axis=1).sum())
    out["full_decode"] = dict(
        frac_exact_trt_vs_cpu=frac_trt_cpu,
        frac_exact_trt_vs_gpu=frac_trt_gpu,
        ler_cpu=ler_cpu, ler_gpu=ler_gpu, ler_triton=ler_trt,
        n=n_val)
    print("FULL_DECODE", out["full_decode"])

    # ---- head-to-head: END-TO-END decode_batch (bench_latency harness) ----
    bench = []
    for shots in SHOTS_GRID:
        n_iter = 50 if shots <= 4096 else 20
        rt = BpTriton.bench_latency(dem, shots=shots, device="cuda",
                                    n_warmup=10, n_iter=n_iter,
                                    max_iter=MAX_ITER, ms_scaling_factor=MS,
                                    block_s=BLOCK_S)
        rg = BpGpu.bench_latency(dem, shots=shots, device="cuda",
                                 n_warmup=10, n_iter=n_iter,
                                 max_iter=MAX_ITER, ms_scaling_factor=MS)
        speedup = rg["mean_ms"] / rt["mean_ms"] if rt["mean_ms"] > 0 else 0.0
        row = dict(
            shots=shots,
            triton_mean_ms=round(rt["mean_ms"], 4),
            triton_p99_9_ms=round(rt["p99_9_ms"], 4),
            triton_throughput=round(rt["throughput_shots_per_s"], 1),
            torch_mean_ms=round(rg["mean_ms"], 4),
            torch_p99_9_ms=round(rg["p99_9_ms"], 4),
            torch_throughput=round(rg["throughput_shots_per_s"], 1),
            speedup_triton_over_torch=round(speedup, 3),
        )
        bench.append(row)
        print("BENCH(decode_batch)", row)
    out["bench_decode_batch"] = bench

    # ---- FAIR head-to-head: BP-CORE ONLY (run_iterations_batch -> numpy) ----
    # Isolates the kernel from the observable projection: BOTH classes return the
    # (S, n_bits) posterior to host the same way, so this is the honest kernel
    # speedup (no GPU-vs-CPU Lo-matmul confound).
    def _bench_core(dec, shots, n=30, warm=8):
        rng = np.random.default_rng(0)
        syn = rng.integers(0, 2, size=(shots, dec.n_checks)).astype(np.uint8)
        for _ in range(warm):
            dec.run_iterations_batch(syn, n_iter=MAX_ITER, device="cuda")
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(n):
            ev0.record()
            dec.run_iterations_batch(syn, n_iter=MAX_ITER, device="cuda")
            ev1.record(); torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1))
        return float(np.mean(ts))

    core = []
    for shots in SHOTS_GRID:
        tt = _bench_core(trt, shots)
        tg = _bench_core(gpu, shots)
        row = dict(shots=shots, triton_mean_ms=round(tt, 4),
                   torch_mean_ms=round(tg, 4),
                   speedup_triton_over_torch=round(tg / tt, 3) if tt > 0 else 0.0)
        core.append(row)
        print("BENCH(bp_core)", row)
    out["bench_bp_core"] = core

    best_e2e = max(r["speedup_triton_over_torch"] for r in bench)
    best_core = max(r["speedup_triton_over_torch"] for r in core)
    out["verdict"] = dict(
        triton_beats_torch=bool(best_core > 1.0),
        best_speedup_e2e_decode_batch=best_e2e,
        best_speedup_bp_core_only=best_core,
        note=("BP-core-only is the honest kernel speedup; end-to-end additionally "
              "moves the observable projection to the GPU."),
    )
    print("VERDICT", out["verdict"])

    with open("/tmp/bench_triton.json", "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE /tmp/bench_triton.json")
    return out


if __name__ == "__main__":
    main()
    sys.stdout.flush()
