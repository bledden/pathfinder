"""Verify numerical equivalence between original and stacked, then benchmark."""
import sys, time
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace")
import torch, torch.nn as nn
from model import NeuralDecoder, DirectionalConv3d
from pathfinder_opt import StackedDirectionalConv3d, swap_to_stacked

# ---- unit test: DirectionalConv3d vs Stacked ----
torch.manual_seed(0)
C_in, C_out = 64, 64
orig = DirectionalConv3d(C_in, C_out).cuda().eval()
stacked = StackedDirectionalConv3d(C_in, C_out).cuda().eval()
orig_sd = {f"w_{d}.weight": getattr(orig, f"w_{d}").weight.data for d in ["self","tp","tm","rp","rm","cp","cm"]}
stacked.load_from_original(orig_sd)

for (B,T,R,Co) in [(1,3,3,3), (1,7,7,7), (4,7,7,7), (64,5,5,5)]:
    x = torch.randn(B, C_in, T, R, Co, device="cuda")
    with torch.no_grad():
        y_orig = orig(x)
        y_new = stacked(x)
    diff = (y_orig - y_new).abs().max().item()
    rel = diff / y_orig.abs().max().item()
    print(f"B={B} T={T} R={R} Co={Co}: max|diff|={diff:.3e}  relative={rel:.3e}  {'PASS' if rel < 1e-4 else 'FAIL'}")

# ---- end-to-end model test ----
print("\n=== end-to-end model (d=7) ===")
ckpt = torch.load("/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt", weights_only=False, map_location="cuda")
model_orig = NeuralDecoder(ckpt["config"]).cuda().eval()
model_orig.load_state_dict(ckpt["model_state_dict"])

import copy
model_stacked = copy.deepcopy(model_orig)
swap_to_stacked(model_stacked)
model_stacked.eval()

for B in [1, 16, 256]:
    x = torch.randint(0, 2, (B, 1, 7, 7, 7), dtype=torch.float32, device="cuda")
    with torch.no_grad():
        y1 = model_orig(x)
        y2 = model_stacked(x)
    diff = (y1 - y2).abs().max().item()
    print(f"B={B}: max|diff|={diff:.3e}  {'PASS' if diff < 1e-3 else 'FAIL'}")

# ---- latency benchmark at batch=1, 64, 1024 in FP16 ----
print("\n=== batch=1 FP16 latency ===")
m_orig_fp16 = model_orig.half()
m_stacked_fp16 = model_stacked.half()

def bench(model, x, iters=500, warmup=50):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters

for B in [1, 16, 64, 256, 1024]:
    x = torch.randint(0, 2, (B, 1, 7, 7, 7), dtype=torch.float16, device="cuda")
    us_orig = bench(m_orig_fp16, x)
    us_stacked = bench(m_stacked_fp16, x)
    speedup = us_orig / us_stacked
    print(f"B={B:>4}: original={us_orig:7.2f} us/call, stacked={us_stacked:7.2f} us/call, speedup={speedup:.2f}x")

# ---- with torch.compile reduce-overhead ----
print("\n=== FP16 + torch.compile(reduce-overhead) ===")
m_orig_c = torch.compile(m_orig_fp16, mode="reduce-overhead")
m_stacked_c = torch.compile(m_stacked_fp16, mode="reduce-overhead")
for B in [1, 16, 64, 256, 1024]:
    x = torch.randint(0, 2, (B, 1, 7, 7, 7), dtype=torch.float16, device="cuda")
    us_orig = bench(m_orig_c, x)
    us_stacked = bench(m_stacked_c, x)
    speedup = us_orig / us_stacked
    print(f"B={B:>4}: original={us_orig:7.2f} us/call, stacked={us_stacked:7.2f} us/call, speedup={speedup:.2f}x")
