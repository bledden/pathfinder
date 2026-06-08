"""Pod-only: validate (LER-identity) + benchmark the Triton Relay-BP kernel.

Runs on the H200 (requires triton + CUDA + relay_bp). Emits a JSON with:
  - LER-IDENTITY: RelayBpTriton vs the relay_bp Rust oracle (RelayDecoderF64 via
    the same wiring as the zoo RelayBPAdapter) on N>=2000 canonical shots --
    logical-error COUNT match + per-shot observable agreement %.
  - PRE-LEG / MEMORY-TERM identity: posterior fp-match vs MinSumBPDecoderF64.
  - THROUGHPUT: CUDA-event batched throughput (shots/s, us/syndrome) for
    RelayBpTriton vs the CPU relay_bp reference, warmup>=10 / measure>=100.

The CPU relay_bp is the slowest practical decoder (~1557 us/syn). RelayBP is
many-iteration; the win is per-iteration throughput, not single-shot latency --
both regimes reported where feasible.

Usage (on pod):
  PYTHONPATH=/workspace/pf-mle:/workspace/pf-mle/qldpc python3 \
      qldpc/kernel/bench_relay_triton.py > /tmp/bench_relay_triton.out 2>&1
"""
import json
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


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print("GPU:", torch.cuda.get_device_name(0))
    circ = build_memory(BBCode(), rounds=ROUNDS, p=P, basis="Z", noise="si1000")
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=0).sample(
        N_VAL, separate_observables=True)
    dets = np.asarray(dets, dtype=bool)
    obs = np.asarray(obs, dtype=bool)

    out = {"config": dict(rounds=ROUNDS, p=P, n_val=N_VAL, block_s=BLOCK_S,
                          relay=RELAY_CFG)}

    trt = RelayBpTriton.from_dem(dem, block_s=BLOCK_S, **RELAY_CFG)
    out["structure"] = dict(n_checks=trt.n_checks, n_bits=trt.n_bits,
                            n_edges=trt.n_edges, MAXDEG_C=trt.MAXDEG_C,
                            MAXDEG_B=trt.MAXDEG_B)
    print("STRUCTURE", out["structure"])

    # ---- PRE-LEG / MEMORY-TERM identity vs MinSumBPDecoderF64 ----
    import relay_bp
    from relay_bp.stim import CheckMatrices
    cm = CheckMatrices.from_dem(dem)
    syn = dets.astype(np.uint8)
    n_probe = 256
    # pre-leg (gamma=0, 1 iter): posterior fp match
    post1 = trt.minsum_posterior_batch(syn[:n_probe], n_iter=1, gamma=0.0,
                                       alpha=1.0, device="cuda")
    dec0 = relay_bp.MinSumBPDecoderF64(cm.check_matrix,
                                       error_priors=cm.error_priors,
                                       max_iter=1, alpha=1.0, gamma0=0.0)
    ref1 = np.stack([np.asarray(dec0.decode_detailed(syn[i]).posterior_ratios)
                     for i in range(n_probe)])
    pre_maxdiff = float(np.max(np.abs(post1 - ref1)))
    # memory term (gamma=0.1, 30 iters): hard decision per-bit
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

    # ---- FULL RELAY LER-IDENTITY vs the relay_bp oracle ----
    _, _, runner = _relay_oracle(dem)
    pred_ref = (np.asarray(runner.decode_observables_batch(syn)) % 2).astype(bool)
    if pred_ref.ndim == 1:
        pred_ref = pred_ref.reshape(-1, 1)
    pred_trt = trt.decode_batch(dets, device="cuda")
    ler_ref = int(np.any(pred_ref != obs, axis=1).sum())
    ler_trt = int(np.any(pred_trt != obs, axis=1).sum())
    per_shot = float(np.all(pred_trt == pred_ref, axis=1).mean())
    out["ler_identity"] = dict(
        oracle_logical_errors=ler_ref,
        triton_logical_errors=ler_trt,
        per_shot_agreement=per_shot,
        n=N_VAL,
        ler_abs_diff=abs(ler_trt - ler_ref))
    print("LER_IDENTITY", out["ler_identity"])

    # ---- THROUGHPUT: RelayBpTriton (CUDA-event) vs CPU relay_bp ----
    # Triton batched throughput at several shot counts.
    bench = []
    for shots in [256, 1024, 4096]:
        n_iter = 30 if shots <= 1024 else 10
        rt = RelayBpTriton.bench_latency(dem, shots=shots, device="cuda",
                                         n_warmup=10, n_iter=n_iter,
                                         block_s=BLOCK_S, **RELAY_CFG)
        row = dict(shots=shots,
                   triton_mean_ms=round(rt["mean_ms"], 3),
                   triton_per_syndrome_us=round(rt["per_syndrome_us"], 3),
                   triton_throughput_shots_per_s=round(
                       rt["throughput_shots_per_s"], 1))
        bench.append(row)
        print("BENCH(relay_triton)", row)
    out["bench_triton"] = bench

    # CPU relay_bp reference: time the oracle runner over a representative batch.
    cpu_shots = 512
    _, _, runner_b = _relay_oracle(dem)
    syn_b = syn[:cpu_shots]
    for _ in range(2):  # warmup
        runner_b.decode_observables_batch(syn_b)
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        runner_b.decode_observables_batch(syn_b)
    t1 = time.perf_counter()
    cpu_ms = (t1 - t0) / reps * 1e3
    cpu_per_syn_us = cpu_ms / cpu_shots * 1e3
    cpu_throughput = cpu_shots / (cpu_ms / 1e3)
    out["cpu_relay_bp"] = dict(
        shots=cpu_shots, mean_ms=round(cpu_ms, 3),
        per_syndrome_us=round(cpu_per_syn_us, 3),
        throughput_shots_per_s=round(cpu_throughput, 1))
    print("CPU_RELAY_BP", out["cpu_relay_bp"])

    # Speedup at the largest triton shot count vs CPU per-syndrome rate.
    best = max(bench, key=lambda r: r["triton_throughput_shots_per_s"])
    speedup = best["triton_throughput_shots_per_s"] / cpu_throughput
    out["verdict"] = dict(
        best_triton_throughput=best["triton_throughput_shots_per_s"],
        best_triton_per_syndrome_us=best["triton_per_syndrome_us"],
        cpu_relay_bp_per_syndrome_us=out["cpu_relay_bp"]["per_syndrome_us"],
        throughput_speedup_vs_cpu_relay_bp=round(speedup, 1),
        ler_identity_match=bool(out["ler_identity"]["ler_abs_diff"]
                                <= max(5, int(0.01 * N_VAL))))
    print("VERDICT", out["verdict"])

    with open("/tmp/bench_relay_triton.json", "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE /tmp/bench_relay_triton.json")
    return out


if __name__ == "__main__":
    main()
    sys.stdout.flush()
