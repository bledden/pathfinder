"""
FP8 quantization test on H200 using torchao.
Applies FP8 dynamic quantization to Linear/Conv modules, benchmarks latency,
validates LER on a small syndrome sample.
"""
import sys, time, json, os
sys.path.insert(0, "/workspace/pathfinder/train")
import torch
from model import NeuralDecoder
from data import SyndromeDataset, DataConfig


def bench(fn, iters=500, warmup=50):
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def evaluate_ler(model, distance, noise_rate, n_shots=10000):
    """Pathfinder LER on fresh syndromes."""
    cfg = DataConfig(distance=distance, rounds=distance, physical_error_rate=noise_rate)
    ds = SyndromeDataset(cfg)
    errors = 0
    total = 0
    batch = 500
    for _ in range(n_shots // batch):
        synd, labels = ds.sample(batch)
        synd = synd.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            logits = model(synd.half())
            preds = (logits > 0).float()
        errors += (preds != labels).any(dim=1).sum().item()
        total += batch
    return errors / max(total, 1)


def load_model(d, r, ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cuda")
    m = NeuralDecoder(ckpt["config"]).cuda()
    m.eval()
    m.load_state_dict(ckpt["model_state_dict"])
    return m


def run_fp8(ckpt_path, d, r):
    print(f"\n=== d={d} FP8 on H200 ===")

    # Baseline: FP16 compiled
    m = load_model(d, r, ckpt_path).half()
    x = torch.randint(0, 2, (1, 1, r, d, d), dtype=torch.float16, device="cuda")
    mc_fp16 = torch.compile(m, mode="reduce-overhead")
    us_fp16 = bench(lambda: mc_fp16(x))
    print(f"  FP16 compile(reduce-oh) B=1:     {us_fp16:8.2f} us/call")
    ler_fp16 = evaluate_ler(m, d, 0.007, n_shots=5000)
    print(f"  FP16 LER @ p=0.007 (5K shots):   {ler_fp16:.5f}")
    torch._dynamo.reset()

    # FP8 via torchao float8 dynamic
    from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
    m_fp8 = load_model(d, r, ckpt_path).half()
    try:
        def _filter(mod, fqn):
            return isinstance(mod, torch.nn.Linear) and mod.in_features % 16 == 0 and mod.out_features % 16 == 0
        quantize_(m_fp8, float8_dynamic_activation_float8_weight(), filter_fn=_filter)
        print(f"  FP8 quantized: OK")
    except Exception as e:
        print(f"  FP8 quantize FAIL: {type(e).__name__}: {e}")
        return

    # Benchmark FP8 eager
    with torch.no_grad():
        us_fp8_eager = bench(lambda: m_fp8(x))
    print(f"  FP8 eager B=1:                   {us_fp8_eager:8.2f} us/call")

    # Benchmark FP8 compiled
    try:
        mc_fp8 = torch.compile(m_fp8, mode="reduce-overhead")
        with torch.no_grad():
            us_fp8_comp = bench(lambda: mc_fp8(x))
        print(f"  FP8 compile(reduce-oh) B=1:      {us_fp8_comp:8.2f} us/call")
    except Exception as e:
        print(f"  FP8 compile FAIL: {type(e).__name__}: {str(e)[:200]}")
        us_fp8_comp = None
    torch._dynamo.reset()

    # Validate LER on FP8 model
    try:
        ler_fp8 = evaluate_ler(m_fp8, d, 0.007, n_shots=5000)
        print(f"  FP8 LER @ p=0.007 (5K shots):    {ler_fp8:.5f}   (delta vs FP16: {(ler_fp8 - ler_fp16)*100:+.3f}%pt)")
    except Exception as e:
        print(f"  FP8 LER FAIL: {type(e).__name__}: {e}")
        ler_fp8 = None

    # Benchmark at batch=1024 too
    x_big = torch.randint(0, 2, (1024, 1, r, d, d), dtype=torch.float16, device="cuda")
    m_fp16 = load_model(d, r, ckpt_path).half()
    mc_fp16_big = torch.compile(m_fp16, mode="reduce-overhead")
    us_fp16_big = bench(lambda: mc_fp16_big(x_big), iters=100, warmup=10)
    per_fp16 = us_fp16_big / 1024
    print(f"  FP16 compile B=1024:             {us_fp16_big:8.2f} us/call ({per_fp16:.2f} us/syn)")
    torch._dynamo.reset()

    try:
        mc_fp8_big = torch.compile(m_fp8, mode="reduce-overhead")
        us_fp8_big = bench(lambda: mc_fp8_big(x_big), iters=100, warmup=10)
        per_fp8 = us_fp8_big / 1024
        print(f"  FP8 compile B=1024:              {us_fp8_big:8.2f} us/call ({per_fp8:.2f} us/syn)")
    except Exception as e:
        print(f"  FP8 B=1024 FAIL: {type(e).__name__}: {str(e)[:200]}")
    torch._dynamo.reset()


def main():
    torch.backends.cudnn.benchmark = True
    print("torch:", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
    configs = [
        ("/workspace/pathfinder/train/checkpoints/d5_muon/best_model.pt", 5, 5),
        ("/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt", 7, 7),
    ]
    for ckpt, d, r in configs:
        run_fp8(ckpt, d, r)


if __name__ == "__main__":
    main()
