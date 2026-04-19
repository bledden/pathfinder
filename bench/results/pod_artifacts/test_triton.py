"""Verify Triton DirectionalConv3d matches reference, benchmark end-to-end."""
import sys, time, copy
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace")
import torch
from model import NeuralDecoder, DirectionalConv3d
from triton_directional import TritonDirectionalConv3d, swap_to_triton

# --- Unit test ---
torch.manual_seed(0)
for (B, T, R, Co, Cin, Cout) in [(1, 3, 3, 3, 64, 64), (1, 7, 7, 7, 64, 64), (64, 7, 7, 7, 64, 64), (1024, 7, 7, 7, 64, 64)]:
    orig = DirectionalConv3d(Cin, Cout).cuda().eval()
    new = TritonDirectionalConv3d(Cin, Cout).cuda().eval()
    sd = {f"w_{d}.weight": getattr(orig, f"w_{d}").weight.data for d in ["self","tp","tm","rp","rm","cp","cm"]}
    new.load_from_original(sd)

    # Test in FP32
    x = torch.randn(B, Cin, T, R, Co, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        y_orig = orig(x)
        y_new = new(x)
    diff = (y_orig - y_new).abs().max().item()
    rel = diff / y_orig.abs().max().item() if y_orig.abs().max() > 0 else 0
    print(f"FP32 B={B} T={T} R={R} Co={Co}: max|diff|={diff:.3e}  rel={rel:.3e}  {'PASS' if rel < 1e-4 else 'FAIL'}")

    # Test in FP16
    orig16 = orig.half()
    new16 = new.half()
    x16 = x.half()
    with torch.no_grad():
        y_orig16 = orig16(x16)
        y_new16 = new16(x16)
    diff16 = (y_orig16 - y_new16).abs().max().item()
    rel16 = diff16 / y_orig16.abs().max().item() if y_orig16.abs().max() > 0 else 0
    print(f"FP16 B={B} T={T} R={R} Co={Co}: max|diff|={diff16:.3e}  rel={rel16:.3e}  {'PASS' if rel16 < 1e-2 else 'FAIL'}")
    print()

# --- End-to-end test at d=7 ---
print("=== end-to-end d=7 ===")
ck = torch.load("/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt", weights_only=False, map_location="cuda")
m_orig = NeuralDecoder(ck["config"]).cuda().eval()
m_orig.load_state_dict(ck["model_state_dict"])
m_triton = copy.deepcopy(m_orig)
swap_to_triton(m_triton)
m_triton.eval()

m_orig = m_orig.half()
m_triton = m_triton.half()

for B in [1, 64, 1024]:
    x = torch.randint(0, 2, (B, 1, 7, 7, 7), dtype=torch.float16, device="cuda")
    with torch.no_grad():
        y1 = m_orig(x)
        y2 = m_triton(x)
    diff = (y1 - y2).abs().max().item()
    print(f"B={B}: max|diff|={diff:.3e}  {'PASS' if diff < 5e-3 else 'FAIL'}")

# --- Benchmark ---
print("\n=== benchmarks ===")

def bench(fn, iters=500, warmup=50):
    for _ in range(warmup): _ = fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): _ = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters

for B in [1, 64, 1024]:
    x = torch.randint(0, 2, (B, 1, 7, 7, 7), dtype=torch.float16, device="cuda")
    # Eager baselines
    with torch.no_grad():
        us_orig_eager = bench(lambda: m_orig(x))
        us_triton_eager = bench(lambda: m_triton(x))
    # Compiled
    torch._dynamo.reset()
    mc_orig = torch.compile(m_orig, mode="max-autotune")
    mc_triton = torch.compile(m_triton, mode="max-autotune")
    with torch.no_grad():
        us_orig_c = bench(lambda: mc_orig(x))
        us_triton_c = bench(lambda: mc_triton(x))
    print(f"B={B:>4}: orig_eager={us_orig_eager:>8.2f}  triton_eager={us_triton_eager:>8.2f}  orig_compiled={us_orig_c:>8.2f}  triton_compiled={us_triton_c:>8.2f}")
    torch._dynamo.reset()
