"""Full-model integration test and benchmark."""
import sys, time, copy
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace")
import torch
from model import NeuralDecoder
from triton_block import swap_to_full_triton
from data import SyndromeDataset, DataConfig

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

ck = torch.load("/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt", weights_only=False, map_location="cuda")
m_orig = NeuralDecoder(ck["config"]).cuda().eval()
m_orig.load_state_dict(ck["model_state_dict"])
m_triton = copy.deepcopy(m_orig)
swap_to_full_triton(m_triton)
m_triton.eval()

m_orig = m_orig.half()
m_triton = m_triton.half()

# --- LER equivalence ---
print("=== LER equivalence check ===")
for p in [0.003, 0.007, 0.010]:
    ds = SyndromeDataset(DataConfig(distance=7, rounds=7, physical_error_rate=p))
    disagree = 0
    err_orig = err_triton = total = 0
    for i in range(20):
        torch.manual_seed(1000 + i)
        synd, lab = ds.sample(500)
        synd, lab = synd.cuda(), lab.cuda()
        with torch.no_grad():
            po = (m_orig(synd.half()) > 0).float()
            pt = (m_triton(synd.half()) > 0).float()
        disagree += (po != pt).any(dim=1).sum().item()
        err_orig += (po != lab).any(dim=1).sum().item()
        err_triton += (pt != lab).any(dim=1).sum().item()
        total += 500
    print(f"p={p:.3f}: LER orig={err_orig/total:.5f}  triton={err_triton/total:.5f}  disagree={disagree}/{total} ({100*disagree/total:.3f}%)")

# --- Latency ---
print("")
print("=== latency (alternating pairs, 5 trials per batch) ===")

def timed(fn, n):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n): _ = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / n

for B in [1, 64, 1024]:
    x = torch.randint(0, 2, (B, 1, 7, 7, 7), dtype=torch.float16, device="cuda")
    torch._dynamo.reset()
    mc_orig = torch.compile(m_orig, mode="max-autotune")
    mc_triton = torch.compile(m_triton, mode="max-autotune")

    # Warmup
    with torch.no_grad():
        for _ in range(100):
            _ = mc_orig(x); _ = mc_triton(x)
    torch.cuda.synchronize()

    os_list = []; ts_list = []
    for _ in range(5):
        os_list.append(timed(lambda: mc_orig(x), 100))
        ts_list.append(timed(lambda: mc_triton(x), 100))
    om = sorted(os_list)[len(os_list)//2]
    tm = sorted(ts_list)[len(ts_list)//2]
    speedup = om / tm
    print(f"B={B:>4}: orig_median={om:>8.2f} us  triton_median={tm:>8.2f} us  speedup={speedup:.2f}x  ({tm/B:.3f} vs {om/B:.3f} us/syn)")
    torch._dynamo.reset()
