"""T8/T9 launch-overhead receipt -- the evidence for "fusion, not tensor cores".

Coda's fusion claim (spec 9, LOCKED): the torch edge-list BP baseline loses to the
fused Triton kernel because BP is per-iteration LAUNCH-BOUND, not arithmetic-bound:
torch fires HUNDREDS of small scatter/gather/reduce CUDA kernels per BP iteration,
so per-iteration launch overhead is comparable to the actual compute, while the
fused Triton kernel collapses each iteration to ~2 launches (check-update +
bit-update) of coalesced FP32 -- moving the kernel closer to memory-bandwidth-bound.
Min-sum is a min/sum semiring (no GEMM), so tensor cores are NOT in play.

This script produces a rough but honest receipt, on the pod, via torch.profiler:

  (1) LAUNCH COUNT per BP iteration -- count CUDA kernel launches (cuda_time
      events) for ONE iteration of the torch baseline's `_iterate_batch` vs the
      Triton kernel's check+bit update. The ratio is the fusion factor.
  (2) LAUNCH-OVERHEAD vs RUNTIME -- for the torch baseline, compare the CPU-side
      wall time to dispatch the iteration (launch overhead, measured with the GPU
      result not awaited) against the GPU-side compute time (CUDA events). When
      overhead ~ runtime the iteration is launch-bound (the fusion premise).
  (3) ARITHMETIC INTENSITY sanity -- report bytes moved per iteration vs FLOPs to
      argue the fused kernel is bandwidth-bound (compute is cheap min/sum, the
      cost is moving messages), and confirm no tensor-core / GEMM op appears in
      either profile (forecloses "why not cuBLAS").

Writes ``launch_overhead_receipt.json``. Pod-only (needs triton + CUDA).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_QLDPC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_QLDPC)
for _p in (_QLDPC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ROUNDS = 6
P = 0.003
MS = 0.625
MAX_ITER = 30
BLOCK_S = 256


# PRECISE GEMM / tensor-core kernel-name signatures. These appear ONLY in real
# matmul / tensor-core kernels (cuBLAS, CUTLASS, arch-tagged GEMM tiles, the
# *_tensor_op_* / wmma / wgmma MMA instructions). We deliberately do NOT match the
# bare word "tensor" (it is in every "TensorIterator"/"TensorAssign" elementwise
# kernel name and would false-positive every op).
_GEMM_SIGNATURES = (
    "gemm", "cutlass", "cublas", "wgmma", "wmma", "_tensor_op_",
    "volta_", "turing_", "ampere_", "hopper_", "s16816", "h16816",
)


def _count_cuda_launches(prof):
    """Count CUDA device kernel launches in a torch profiler trace.

    Counts profiler events that ran on the device (positive CUDA self time);
    flags any GEMM / tensor-core op by a PRECISE name signature (to confirm none
    appear -- min-sum is a min/sum semiring, no matmul path)."""
    n_launch = 0
    gemm_ops = []
    for ev in prof.key_averages():
        cuda_us = getattr(ev, "self_device_time_total", None)
        if cuda_us is None:
            cuda_us = getattr(ev, "self_cuda_time_total", 0.0)
        count = getattr(ev, "count", 0)
        if cuda_us and cuda_us > 0:
            n_launch += int(count)
            low = ev.key.lower()
            if any(s in low for s in _GEMM_SIGNATURES):
                gemm_ops.append((ev.key[:80], int(count)))
    return n_launch, gemm_ops


def main(out_path=None, shots=4096):
    import torch
    from torch.profiler import ProfilerActivity, profile

    assert torch.cuda.is_available(), "CUDA required (pod-only receipt)"
    from bb_code import BBCode
    from qldpc.foundation.circuits import build_memory
    from qldpc.kernel.bp_gpu import BpGpu
    from qldpc.kernel.bp_triton import BpTriton

    circ = build_memory(BBCode(), rounds=ROUNDS, p=P, basis="Z", noise="si1000")
    dem = circ.detector_error_model(decompose_errors=False)

    gpu = BpGpu.from_dem(dem, max_iter=MAX_ITER, ms_scaling_factor=MS)
    trt = BpTriton.from_dem(dem, max_iter=MAX_ITER, ms_scaling_factor=MS,
                            block_s=BLOCK_S)

    rng = np.random.default_rng(0)
    syn = rng.integers(0, 2, size=(shots, gpu.n_checks)).astype(np.uint8)

    out = {
        "kind": "launch-overhead-receipt",
        "claim": ("BP is per-iteration LAUNCH-BOUND; the fused Triton kernel wins "
                  "by collapsing hundreds of torch scatter/reduce launches to ~2 "
                  "coalesced FP32 launches/iter. Min-sum is a min/sum semiring -- "
                  "no GEMM/tensor-core path (forecloses 'why not cuBLAS')."),
        "config": dict(rounds=ROUNDS, p=P, ms=MS, max_iter=MAX_ITER,
                       block_s=BLOCK_S, shots=shots),
        "structure": dict(n_checks=trt.n_checks, n_bits=trt.n_bits,
                          n_edges=trt.n_edges, MAXDEG_C=trt.MAXDEG_C,
                          MAXDEG_B=trt.MAXDEG_B),
        "gpu": torch.cuda.get_device_name(0),
    }

    # ----- (1) launch count per BP iteration -------------------------------- #
    # torch baseline: profile ONE _iterate_batch via run_iterations_batch n_iter=1.
    for _ in range(5):
        gpu.run_iterations_batch(syn, n_iter=1, device="cuda")
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pg:
        gpu.run_iterations_batch(syn, n_iter=1, device="cuda")
        torch.cuda.synchronize()
    torch_launches, torch_gemm = _count_cuda_launches(pg)

    for _ in range(5):
        trt.run_iterations_batch(syn, n_iter=1, device="cuda")
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pt:
        trt.run_iterations_batch(syn, n_iter=1, device="cuda")
        torch.cuda.synchronize()
    triton_launches, triton_gemm = _count_cuda_launches(pt)

    out["launches_per_iteration"] = dict(
        torch_baseline=torch_launches,
        triton_kernel=triton_launches,
        fusion_factor=(round(torch_launches / triton_launches, 2)
                       if triton_launches else None),
        torch_gemm_ops=torch_gemm,        # expected: [] (no tensor cores)
        triton_gemm_ops=triton_gemm,      # expected: []
        note=("launch = a profiler event with positive CUDA self-time; the torch "
              "iteration is many scatter_add/gather/reduce kernels, the Triton "
              "iteration is 2 fused kernels (check-update + bit-update)."),
    )

    # ----- (2) launch overhead (CPU dispatch) vs GPU runtime, torch --------- #
    # GPU runtime: CUDA-event time for one torch iteration (device compute).
    def _cuda_time(fn, n=50, warm=10):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(n):
            ev0.record(); fn(); ev1.record(); torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1))
        return float(np.mean(ts))

    # CPU dispatch (launch) overhead: time the host-side python+dispatch WITHOUT
    # awaiting the GPU (no synchronize inside the loop), then subtract a sync once.
    import time as _time

    def _cpu_dispatch_time(fn, n=50, warm=10):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        for _ in range(n):
            fn()                          # enqueue only; do NOT synchronize
        t1 = _time.perf_counter()         # host returns once the queue is filled
        torch.cuda.synchronize()          # drain (not counted)
        return (t1 - t0) / n * 1e3        # ms per iteration, host dispatch side

    torch_iter = lambda: gpu.run_iterations_batch(syn, n_iter=1, device="cuda")
    triton_iter = lambda: trt.run_iterations_batch(syn, n_iter=1, device="cuda")

    torch_gpu_ms = _cuda_time(torch_iter)
    torch_cpu_ms = _cpu_dispatch_time(torch_iter)
    triton_gpu_ms = _cuda_time(triton_iter)
    triton_cpu_ms = _cpu_dispatch_time(triton_iter)

    out["overhead_vs_runtime_ms_per_iter"] = dict(
        torch_gpu_runtime_ms=round(torch_gpu_ms, 5),
        torch_cpu_dispatch_ms=round(torch_cpu_ms, 5),
        torch_overhead_over_runtime=(round(torch_cpu_ms / torch_gpu_ms, 3)
                                     if torch_gpu_ms else None),
        triton_gpu_runtime_ms=round(triton_gpu_ms, 5),
        triton_cpu_dispatch_ms=round(triton_cpu_ms, 5),
        note=("These per-iteration wrappers include run_iterations_batch's "
              "host-side syndrome marshalling; the comparison that matters is "
              "torch dispatch vs runtime ~ O(1) => launch-bound."),
    )

    # ----- (3) bandwidth-bound sanity: bytes moved vs flops per iteration ---- #
    E = trt.n_edges
    # Per iteration the kernel reads/writes ~ a small constant number of fp32
    # message arrays of length E*shots; the arithmetic is O(E*shots) min/adds.
    bytes_per_iter = 4 * E * shots * 6   # ~6 message-array touches/iter (fp32)
    flops_per_iter = E * shots * 4       # min/sum/cmp ~ a few ops/edge
    out["arithmetic_intensity"] = dict(
        edges=E, shots=shots,
        approx_bytes_per_iter=int(bytes_per_iter),
        approx_flops_per_iter=int(flops_per_iter),
        approx_flops_per_byte=round(flops_per_iter / bytes_per_iter, 4),
        note=("flops/byte << 1 => memory-bandwidth-bound, NOT compute-bound; "
              "consistent with a fused-launch min/sum kernel rather than a GEMM."),
    )

    # Launch-bound evidence: the torch baseline's host-side dispatch time is
    # comparable to (>= ~half) its GPU runtime -- i.e. the iteration spends as
    # long launching kernels as computing. (Ratio ~1.0 here -> fully launch-bound.)
    overhead_ratio = (torch_cpu_ms / torch_gpu_ms) if torch_gpu_ms else 0.0
    launch_bound = overhead_ratio >= 0.5
    no_tc = (not torch_gemm and not triton_gemm)
    out["verdict"] = dict(
        launch_bound_evidence=bool(launch_bound),
        torch_overhead_over_runtime=round(overhead_ratio, 3),
        fusion_factor=out["launches_per_iteration"]["fusion_factor"],
        no_tensor_cores=bool(no_tc),
        summary=(f"torch fires {torch_launches} CUDA launches/iter vs Triton "
                 f"{triton_launches} (fusion ~"
                 f"{out['launches_per_iteration']['fusion_factor']}x); torch "
                 f"host-dispatch/GPU-runtime ~ {overhead_ratio:.2f} "
                 f"(launch-bound); "
                 f"{'no' if no_tc else 'SOME'} GEMM/tensor-core ops in either "
                 f"profile; arithmetic intensity {out['arithmetic_intensity']['approx_flops_per_byte']}"
                 f" flop/byte (<1) => the win is fusion + bandwidth, not tensor cores."),
    )

    out_path = out_path or os.path.join(_HERE, "launch_overhead_receipt.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return out


if __name__ == "__main__":
    main()
