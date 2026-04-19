"""H200 latency benchmark for Pathfinder."""
import sys, time, json
sys.path.insert(0, "/workspace/pathfinder/train")
import torch
from model import NeuralDecoder


def bench(model, x, iters=300, warmup=50):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def run_config(ckpt_path, distance, rounds, batch_sizes, dtypes, compile_modes):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cuda")
    base_model = NeuralDecoder(ckpt["config"]).cuda()
    base_model.eval()
    base_model.load_state_dict(ckpt["model_state_dict"])
    n_params = sum(p.numel() for p in base_model.parameters())
    results = []
    print("")
    print("=== d=%d, rounds=%d, params=%s ===" % (distance, rounds, format(n_params, ",")))
    print("%6s %17s %6s %9s %11s" % ("dtype", "compile", "batch", "us/syn", "syn/s"))
    for dtype in dtypes:
        dtype_name = "fp16" if dtype == torch.float16 else "fp32"
        for compile_mode in compile_modes:
            cm_str = str(compile_mode) if compile_mode else "eager"
            m = base_model.half() if dtype == torch.float16 else base_model.float()
            if compile_mode:
                m = torch.compile(m, mode=compile_mode)
            for bs in batch_sizes:
                try:
                    x = torch.randint(0, 2, (bs, 1, rounds, distance, distance), dtype=dtype, device="cuda")
                    ms_per_batch = bench(m, x)
                    us_per_syn = ms_per_batch * 1000 / bs
                    throughput = bs / (ms_per_batch / 1000)
                    print("%6s %17s %6d %9.2f %11.0f" % (dtype_name, cm_str, bs, us_per_syn, throughput))
                    results.append({
                        "distance": distance, "rounds": rounds, "params": n_params,
                        "dtype": dtype_name, "compile": cm_str, "batch": bs,
                        "us_per_syn": us_per_syn, "throughput": throughput,
                        "ms_per_batch": ms_per_batch,
                    })
                except Exception as e:
                    print("  SKIP bs=%d: %s: %s" % (bs, type(e).__name__, e))
            if compile_mode:
                torch._dynamo.reset()
    return results


def main():
    torch.backends.cudnn.benchmark = True
    gpu_name = torch.cuda.get_device_name(0)
    print("GPU:", gpu_name)
    print("PyTorch:", torch.__version__)
    all_results = []
    configs = [
        ("/workspace/pathfinder/train/checkpoints/best_model.pt", 3, 3),
        ("/workspace/pathfinder/train/checkpoints/d5_muon/best_model.pt", 5, 5),
        ("/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt", 7, 7),
    ]
    batch_sizes = [1, 16, 64, 256, 1024]
    dtypes = [torch.float32, torch.float16]
    compile_modes = [None, "default", "reduce-overhead"]
    for ckpt, d, r in configs:
        results = run_config(ckpt, d, r, batch_sizes, dtypes, compile_modes)
        all_results.extend(results)
    out = "/workspace/h200_latency_results.json"
    with open(out, "w") as f:
        json.dump({"gpu": gpu_name, "torch": torch.__version__, "results": all_results}, f, indent=2)
    print("")
    print("=== wrote %s ===" % out)


if __name__ == "__main__":
    main()
