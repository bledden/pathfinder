"""
Final post-distillation benchmark suite.
Runs:
  1. Clean isolated latency of full H=256 model with/without Triton kernel
  2. Clean isolated latency of narrow H=128 (regular + distilled) with/without Triton kernel
  3. LER of distilled narrow across all noise rates
  4. Apples-to-apples comparison table
"""
import sys, time, json, copy
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace")
import torch
from model import NeuralDecoder
from triton_directional import swap_to_triton
from data import SyndromeDataset, DataConfig

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def load_fp16(ckpt_path):
    ck = torch.load(ckpt_path, weights_only=False, map_location="cuda")
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
    return min(vals), sum(vals)/len(vals)


def eval_ler(model, p, n_shots=20000):
    ds = SyndromeDataset(DataConfig(distance=7, rounds=7, physical_error_rate=p))
    err = total = 0
    for i in range(n_shots // 500):
        synd, lab = ds.sample(500)
        synd, lab = synd.cuda().half(), lab.cuda()
        with torch.no_grad():
            preds = (model(synd) > 0).float()
        err += (preds != lab).any(dim=1).sum().item()
        total += 500
    return err / total


results = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "configs": []}


def run_config(name, ckpt_path, use_triton):
    print(f"\n=== {name} (Triton={use_triton}) ===")
    m = load_fp16(ckpt_path)
    if use_triton:
        swap_to_triton(m)
        m = m.cuda().half().eval()
    cfg = {"name": name, "triton": use_triton, "path": ckpt_path, "params": sum(p.numel() for p in m.parameters())}

    # Latency at representative batch sizes
    for B in [1, 64, 1024]:
        us_min, us_mean = bench_latency(m, B)
        cfg[f"B{B}_min_us"] = us_min
        cfg[f"B{B}_mean_us"] = us_mean
        print(f"  B={B:>4}: {us_min:>8.2f} us/call  ({us_min/B:.3f} us/syn)")

    # LER across noise rates (for narrow variants)
    if "narrow" in name or "distill" in name or "full" in name.lower():
        for p in [0.003, 0.007, 0.010]:
            ler = eval_ler(m, p, n_shots=20000)
            cfg[f"ler_p{p}"] = ler
            print(f"  LER@p={p:.3f}: {ler:.5f}")

    results["configs"].append(cfg)


# Full H=256 model
run_config("full_H256", "/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt", use_triton=False)
run_config("full_H256_triton", "/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt", use_triton=True)

# Narrow H=128 (non-distilled)
run_config("narrow_H128", "/workspace/pathfinder/train/checkpoints/d7_narrow/best_model.pt", use_triton=False)
run_config("narrow_H128_triton", "/workspace/pathfinder/train/checkpoints/d7_narrow/best_model.pt", use_triton=True)

# Narrow distilled
import os
if os.path.exists("/workspace/pathfinder/train/checkpoints/d7_distill/best_model.pt"):
    run_config("narrow_H128_distill", "/workspace/pathfinder/train/checkpoints/d7_distill/best_model.pt", use_triton=False)
    run_config("narrow_H128_distill_triton", "/workspace/pathfinder/train/checkpoints/d7_distill/best_model.pt", use_triton=True)

with open("/workspace/final_benchmark.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n=== wrote /workspace/final_benchmark.json ===")
