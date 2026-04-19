"""
Batch=1 real-time latency optimization for Pathfinder.
Compares eager vs torch.compile vs manual CUDA Graphs capture.
The goal: push batch=1 latency as close to pure compute as possible.
"""
import sys, time, json
sys.path.insert(0, "/workspace/pathfinder/train")
import torch
from model import NeuralDecoder


def bench_eager(model, x, iters=1000, warmup=100):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters  # us/call


def bench_cuda_graph(model, x, iters=1000, warmup=100):
    """Capture forward pass in a CUDA graph, replay via graph.replay() only."""
    # Warmup on a separate stream (required by CUDA Graphs)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    static_x = x.clone()
    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            static_y = model(static_x)

    # Warmup graph replays
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()

    # Time pure graph replay
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def run(ckpt_path, d, r):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cuda")
    base = NeuralDecoder(ckpt["config"]).cuda()
    base.eval()
    base.load_state_dict(ckpt["model_state_dict"])
    results = {"distance": d, "rounds": r, "params": sum(p.numel() for p in base.parameters())}
    print(f"\n=== d={d}, rounds={r}, params={results['params']:,} ===")

    for dtype in [torch.float32, torch.float16]:
        dname = "fp16" if dtype == torch.float16 else "fp32"
        m = base.half() if dtype == torch.float16 else base.float()
        x = torch.randint(0, 2, (1, 1, r, d, d), dtype=dtype, device="cuda")

        # 1. Eager
        us_eager = bench_eager(m, x)
        print(f"  {dname} eager:                    {us_eager:8.2f} us/call")

        # 2. Compiled default
        try:
            mc = torch.compile(m, mode="default")
            us_compiled = bench_eager(mc, x)
            print(f"  {dname} compile(default):          {us_compiled:8.2f} us/call")
            torch._dynamo.reset()
        except Exception as e:
            print(f"  {dname} compile(default): FAIL {e}")

        # 3. Compiled reduce-overhead
        try:
            mc = torch.compile(m, mode="reduce-overhead")
            us_ro = bench_eager(mc, x)
            print(f"  {dname} compile(reduce-overhead): {us_ro:8.2f} us/call")
            torch._dynamo.reset()
        except Exception as e:
            print(f"  {dname} compile(reduce-overhead): FAIL {e}")

        # 4. CUDA Graph (manual)
        try:
            us_graph = bench_cuda_graph(m, x)
            print(f"  {dname} CUDA Graph (manual):      {us_graph:8.2f} us/call  <-- pure replay")
            results[f"{dname}_cuda_graph_us"] = us_graph
        except Exception as e:
            print(f"  {dname} CUDA Graph: FAIL {type(e).__name__}: {e}")
        results[f"{dname}_eager_us"] = us_eager

    return results


def main():
    torch.backends.cudnn.benchmark = True
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    all_results = []
    configs = [
        ("/workspace/pathfinder/train/checkpoints/best_model.pt", 3, 3),
        ("/workspace/pathfinder/train/checkpoints/d5_muon/best_model.pt", 5, 5),
        ("/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt", 7, 7),
    ]
    for ckpt, d, r in configs:
        all_results.append(run(ckpt, d, r))
    out = "/workspace/h200_cudagraph_results.json"
    with open(out, "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "results": all_results}, f, indent=2)
    print(f"\n=== wrote {out} ===")


if __name__ == "__main__":
    main()
