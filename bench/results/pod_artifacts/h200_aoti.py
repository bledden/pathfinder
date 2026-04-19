"""
AOT Inductor compilation test on PyTorch 2.4.
Goal: eliminate Python runtime overhead at batch=1.
"""
import sys, time, os
sys.path.insert(0, "/workspace/pathfinder/train")
import torch
from torch._export import aot_compile
from model import NeuralDecoder

torch.set_float32_matmul_precision("high")

def bench(fn, iters=500, warmup=50):
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def run(ckpt_path, d, r, batch=1, dtype=torch.float16):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cuda")
    m = NeuralDecoder(ckpt["config"]).cuda()
    m.eval()
    m.load_state_dict(ckpt["model_state_dict"])
    if dtype == torch.float16:
        m = m.half()

    x = torch.randint(0, 2, (batch, 1, r, d, d), dtype=dtype, device="cuda")

    # Baseline eager
    with torch.no_grad():
        us_eager = bench(lambda: m(x))
    print(f"  d={d} B={batch}: eager                    = {us_eager:8.2f} us/call")

    # torch.compile reduce-overhead
    mc = torch.compile(m, mode="reduce-overhead")
    with torch.no_grad():
        us_compiled = bench(lambda: mc(x))
    print(f"  d={d} B={batch}: torch.compile(reduce-oh) = {us_compiled:8.2f} us/call")
    torch._dynamo.reset()

    # AOT compile — this path returns a path to a .so that we load with a runner
    out_dir = f"/workspace/aoti_d{d}_b{batch}_{'fp16' if dtype==torch.float16 else 'fp32'}"
    os.makedirs(out_dir, exist_ok=True)
    try:
        with torch.no_grad():
            so_path = aot_compile(
                m,
                (x,),
                options={
                    "aot_inductor.output_path": f"{out_dir}/model.so",
                    "max_autotune": True,
                    "max_autotune_gemm": True,
                },
            )
        print(f"  d={d} B={batch}: AOT compiled -> {so_path}")

        # Load the runner
        from torch._inductor.package import AOTICompiledModel, load_package
        # Fallback: use the lowlevel runner
        
        import torch._C as _tc; runner = _tc._aoti.AOTIModelContainerRunnerCuda(so_path, 1, "cuda:0")
        # AOTI runner signature: accepts list of tensors, returns list
        with torch.no_grad():
            us_aoti = bench(lambda: runner.run([x]))
        print(f"  d={d} B={batch}: AOTI runner             = {us_aoti:8.2f} us/call")
    except Exception as e:
        print(f"  d={d} B={batch}: AOTI FAIL: {type(e).__name__}: {e}")

    return None


def main():
    torch.backends.cudnn.benchmark = True
    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print()
    configs = [
        ("/workspace/pathfinder/train/checkpoints/best_model.pt", 3, 3),
        ("/workspace/pathfinder/train/checkpoints/d5_muon/best_model.pt", 5, 5),
        ("/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt", 7, 7),
    ]
    for ckpt, d, r in configs:
        print(f"=== d={d} ===")
        for B in [1, 16, 256]:
            run(ckpt, d, r, batch=B)
        print()


if __name__ == "__main__":
    main()
