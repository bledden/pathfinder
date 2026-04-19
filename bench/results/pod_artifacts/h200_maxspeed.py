"""
Maximum speed batch=1 benchmark.
Strategy: get Inductor's kernel fusion, then wrap the entire compiled forward pass
in a single manual CUDA graph capture for one-shot replay.
"""
import sys, time, json
sys.path.insert(0, "/workspace/pathfinder/train")
import torch
from model import NeuralDecoder


def bench_loop(fn, iters=2000, warmup=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def build_graph_of_compiled(model, x):
    """Compile + warm up on side stream, then capture the whole forward as one graph."""
    with torch.no_grad():
        # Warmup compile on side stream (forces Inductor to emit kernels)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(20):  # enough warmup for Inductor to specialize
                _ = model(x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Static buffer for graph input
        static_x = x.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_y = model(static_x)
    return g, static_x, static_y


def run(ckpt_path, d, r):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cuda")
    base = NeuralDecoder(ckpt["config"]).cuda()
    base.eval()
    base.load_state_dict(ckpt["model_state_dict"])
    params = sum(p.numel() for p in base.parameters())
    print(f"\n=== d={d}, rounds={r}, params={params:,} ===")
    out = {"distance": d, "rounds": r, "params": params}

    for dtype in [torch.float16]:
        dname = "fp16"
        m = base.half() if dtype == torch.float16 else base.float()
        x = torch.randint(0, 2, (1, 1, r, d, d), dtype=dtype, device="cuda")

        # Baseline: compile reduce-overhead (re-measure cleanly)
        m1 = torch.compile(m, mode="reduce-overhead")
        with torch.no_grad():
            us_ro = bench_loop(lambda: m1(x))
        print(f"  {dname} compile(reduce-overhead):            {us_ro:8.2f} us/call")
        out[f"{dname}_compile_reduce_overhead_us"] = us_ro
        torch._dynamo.reset()

        # Try max-autotune
        try:
            m2 = torch.compile(m, mode="max-autotune")
            with torch.no_grad():
                us_ma = bench_loop(lambda: m2(x))
            print(f"  {dname} compile(max-autotune):               {us_ma:8.2f} us/call")
            out[f"{dname}_compile_max_autotune_us"] = us_ma
            torch._dynamo.reset()
        except Exception as e:
            print(f"  {dname} max-autotune FAIL: {type(e).__name__}: {e}")

        # Compiled (default) wrapped in a manual CUDA graph
        try:
            m3 = torch.compile(m, mode="default")
            g, sx, sy = build_graph_of_compiled(m3, x)
            us_cg = bench_loop(lambda: g.replay())
            print(f"  {dname} compile(default) + manual graph:   {us_cg:8.2f} us/call  <-- pure replay")
            out[f"{dname}_compile_default_then_graph_us"] = us_cg
            torch._dynamo.reset()
        except Exception as e:
            print(f"  {dname} compile+graph FAIL: {type(e).__name__}: {e}")

        # Eager model wrapped in manual CUDA graph (baseline for just launch-overhead removal)
        try:
            g2, sx2, sy2 = build_graph_of_compiled(m, x)
            us_eg = bench_loop(lambda: g2.replay())
            print(f"  {dname} eager + manual graph:             {us_eg:8.2f} us/call")
            out[f"{dname}_eager_then_graph_us"] = us_eg
        except Exception as e:
            print(f"  {dname} eager+graph FAIL: {type(e).__name__}: {e}")

    return out


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    results = []
    for ckpt, d, r in [
        ("/workspace/pathfinder/train/checkpoints/best_model.pt", 3, 3),
        ("/workspace/pathfinder/train/checkpoints/d5_muon/best_model.pt", 5, 5),
        ("/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt", 7, 7),
    ]:
        results.append(run(ckpt, d, r))
    with open("/workspace/h200_maxspeed_results.json", "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "results": results}, f, indent=2)
    print("\n=== wrote /workspace/h200_maxspeed_results.json ===")


if __name__ == "__main__":
    main()
