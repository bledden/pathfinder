"""Alternating-pairs benchmark: run orig and triton back-to-back multiple times.
Even with background GPU contention, the ratio should be stable.
"""
import sys, time, copy
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace")
import torch
from model import NeuralDecoder
from triton_directional import swap_to_triton

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

ck = torch.load("/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt", weights_only=False, map_location="cuda")
m_orig = NeuralDecoder(ck["config"]).cuda().eval()
m_orig.load_state_dict(ck["model_state_dict"])
m_triton = copy.deepcopy(m_orig)
swap_to_triton(m_triton)
m_triton.eval()

m_orig = m_orig.half()
m_triton = m_triton.half()

for B in [1, 1024]:
    x = torch.randint(0, 2, (B, 1, 7, 7, 7), dtype=torch.float16, device="cuda")
    torch._dynamo.reset()
    mc_orig = torch.compile(m_orig, mode="max-autotune")
    mc_triton = torch.compile(m_triton, mode="max-autotune")

    # Warmup both
    with torch.no_grad():
        for _ in range(100): _ = mc_orig(x); _ = mc_triton(x)
    torch.cuda.synchronize()

    # Alternating measurements
    def timed(fn, n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n): _ = fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e6 / n

    orig_times, triton_times = [], []
    for trial in range(10):
        orig_times.append(timed(lambda: mc_orig(x), 100))
        triton_times.append(timed(lambda: mc_triton(x), 100))

    orig_med = sorted(orig_times)[5]
    triton_med = sorted(triton_times)[5]
    ratio = triton_med / orig_med
    speedup = 1 / ratio
    print(f"B={B:>4}: orig_median={orig_med:>8.2f} us/call  triton_median={triton_med:>8.2f} us/call  "
          f"ratio={ratio:.3f}  speedup={speedup:.2f}x  ({triton_med/B:.3f} us/syn vs {orig_med/B:.3f})")
    print(f"       orig samples: min={min(orig_times):.1f} max={max(orig_times):.1f}")
    print(f"       triton samples: min={min(triton_times):.1f} max={max(triton_times):.1f}")
    torch._dynamo.reset()
