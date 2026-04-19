"""Benchmark H=192 distilled + run ensemble test (H=128 distilled + PM)."""
import sys, time, copy, os
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace")
import torch, numpy as np
import pymatching
from model import NeuralDecoder
from triton_directional import swap_to_triton
from data import SyndromeDataset, DataConfig

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def load_fp16(path):
    ck = torch.load(path, weights_only=False, map_location="cuda")
    m = NeuralDecoder(ck["config"]).cuda().eval()
    m.load_state_dict(ck["model_state_dict"])
    return m.half()


def bench_latency(model, B, trials=5, iters=500, warmup=100):
    x = torch.randint(0, 2, (B, 1, 7, 7, 7), dtype=torch.float16, device="cuda")
    torch._dynamo.reset()
    mc = torch.compile(model, mode="max-autotune")
    with torch.no_grad():
        for _ in range(warmup): _ = mc(x)
    torch.cuda.synchronize()
    vals = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters): _ = mc(x)
        torch.cuda.synchronize()
        vals.append((time.perf_counter() - t0) * 1e6 / iters)
    torch._dynamo.reset()
    return min(vals)


print("=== H=192 distilled latency ===")
m = load_fp16("/workspace/pathfinder/train/checkpoints/d7_h192_distill/best_model.pt")
n_params = sum(p.numel() for p in m.parameters())
print(f"params: {n_params:,}")

for B in [1, 64, 1024]:
    us = bench_latency(m, B)
    print(f"  B={B:>4} (no Triton): {us:>8.2f} us/call  ({us/B:.3f} us/syn)")

# With Triton
m_triton = copy.deepcopy(m).float()  # Need fp32 to swap, then re-half
swap_to_triton(m_triton)
m_triton = m_triton.half().eval()

for B in [1, 64, 1024]:
    us = bench_latency(m_triton, B)
    print(f"  B={B:>4} (Triton):    {us:>8.2f} us/call  ({us/B:.3f} us/syn)")

# Eval H=192 LER at multiple rates
print("")
print("=== H=192 distilled LER across noise rates (20K shots each) ===")
for p in [0.003, 0.005, 0.007, 0.010]:
    ds = SyndromeDataset(DataConfig(distance=7, rounds=7, physical_error_rate=p))
    err = total = 0
    for i in range(40):
        torch.manual_seed(2000+i)
        synd, lab = ds.sample(500)
        synd, lab = synd.cuda().half(), lab.cuda()
        with torch.no_grad():
            preds = (m(synd) > 0).float()
        err += (preds != lab).any(dim=1).sum().item()
        total += 500
    print(f"  p={p:.3f}: LER={err/total:.5f}")

# Ensemble test: H=128 distilled + PyMatching
print("")
print("=== Ensemble: H=128 distilled + PyMatching (d=7, 20K shots) ===")
m128 = load_fp16("/workspace/pathfinder/train/checkpoints/d7_distill/best_model.pt")

print(f"{'p':>6} {'neural':>9} {'PM':>9} {'agree%':>8} {'OR':>9} {'confT=2':>10} {'confT=5':>10}")
for p in [0.003, 0.005, 0.007, 0.010]:
    ds = SyndromeDataset(DataConfig(distance=7, rounds=7, physical_error_rate=p))
    circuit = ds.circuit
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()

    neural_errs = 0
    pm_errs = 0
    or_oracle_errs = 0
    and_oracle_errs = 0
    agree = 0
    ens2_errs = 0
    ens5_errs = 0
    total = 0

    for _ in range(40):
        det, obs = sampler.sample(500, separate_observables=True)
        det_u8 = det.astype(np.uint8)
        obs_u8 = obs.astype(np.uint8)

        # Neural: convert detector events to tensor via dataset helper
        synd_tensor = ds.detectors_to_tensor(det_u8).unsqueeze(1).float().cuda().half()
        with torch.no_grad():
            logits = m128(synd_tensor).cpu().float().numpy()
        preds_neural = (logits > 0).astype(np.uint8)

        # PM
        preds_pm = matching.decode_batch(det_u8)

        for i in range(500):
            n_right = (preds_neural[i] == obs_u8[i]).all()
            p_right = (preds_pm[i] == obs_u8[i]).all()
            if not n_right: neural_errs += 1
            if not p_right: pm_errs += 1
            if (preds_neural[i] == preds_pm[i]).all(): agree += 1
            if not (n_right or p_right): or_oracle_errs += 1
            if not (n_right and p_right): and_oracle_errs += 1
            # Confidence ensembles: pick neural if |logit|>thresh else PM
            conf = abs(logits[i][0])
            chosen2 = preds_neural[i] if conf > 2.0 else preds_pm[i]
            chosen5 = preds_neural[i] if conf > 5.0 else preds_pm[i]
            if not (chosen2 == obs_u8[i]).all(): ens2_errs += 1
            if not (chosen5 == obs_u8[i]).all(): ens5_errs += 1
        total += 500

    print(f"{p:>6.3f} {neural_errs/total:>9.5f} {pm_errs/total:>9.5f} {100*agree/total:>7.2f}% {or_oracle_errs/total:>9.5f} {ens2_errs/total:>10.5f} {ens5_errs/total:>10.5f}")
