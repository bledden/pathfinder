"""Quick latency-only benchmark for the narrow H=128 d=7 model, to verify it's worth finishing training."""
import sys, time
sys.path.insert(0, "/workspace/pathfinder/train")
import torch
from model import NeuralDecoder

torch.set_float32_matmul_precision("high")


def bench(fn, iters=500, warmup=50):
    for _ in range(warmup): _ = fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): _ = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


ckpts = [
    ("original (H=256, L=7, 500K params)", "/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt"),
    ("narrow   (H=128, L=7, 126K params)", "/workspace/pathfinder/train/checkpoints/d7_narrow/best_model.pt"),
]

print("GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__)
for name, path in ckpts:
    try:
        ck = torch.load(path, weights_only=False, map_location="cuda")
        m = NeuralDecoder(ck["config"]).cuda()
        m.eval()
        m.load_state_dict(ck["model_state_dict"])
        m = m.half()
        n = sum(p.numel() for p in m.parameters())
        print(f"\n=== {name} ===")
        print(f"   params: {n:,}")
        for B in [1, 64, 1024]:
            x = torch.randint(0, 2, (B, 1, 7, 7, 7), dtype=torch.float16, device="cuda")
            with torch.no_grad():
                us_eager = bench(lambda: m(x))
            mc = torch.compile(m, mode="reduce-overhead")
            with torch.no_grad():
                us_c = bench(lambda: mc(x))
            print(f"   B={B:<4}  eager={us_eager:>8.2f} us/call    compile(ro)={us_c:>8.2f} us/call    throughput_us_per_syn={us_c/B:>6.3f}")
            torch._dynamo.reset()
    except FileNotFoundError as e:
        print(f"   SKIP {name}: checkpoint not yet available ({e.filename})")
