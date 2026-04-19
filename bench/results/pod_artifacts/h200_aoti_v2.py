"""
AOT Inductor benchmark using PyTorch 2.6 modern API.
Goal: eliminate Python runtime overhead at batch=1.
"""
import sys, time, os
sys.path.insert(0, "/workspace/pathfinder/train")
import torch
from torch._inductor import aoti_compile_and_package, aoti_load_package
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


def compile_aoti(model, example, output_path, max_autotune=False):
    with torch.no_grad():
        exported = torch.export.export(model, (example,))
        options = {"max_autotune": True, "max_autotune_gemm": True} if max_autotune else {}
        pt2 = aoti_compile_and_package(exported, package_path=output_path, inductor_configs=options)
    return pt2


def run(ckpt_path, d, r, batch, dtype):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cuda")
    m = NeuralDecoder(ckpt["config"]).cuda()
    m.eval()
    m.load_state_dict(ckpt["model_state_dict"])
    if dtype == torch.float16:
        m = m.half()

    x = torch.randint(0, 2, (batch, 1, r, d, d), dtype=dtype, device="cuda")

    with torch.no_grad():
        us_eager = bench(lambda: m(x))
    print(f"  d={d} B={batch:<4} {'fp16' if dtype==torch.float16 else 'fp32'}: eager                    = {us_eager:8.2f} us/call")

    try:
        mc = torch.compile(m, mode="reduce-overhead")
        with torch.no_grad():
            us_ro = bench(lambda: mc(x))
        print(f"  d={d} B={batch:<4} {'fp16' if dtype==torch.float16 else 'fp32'}: torch.compile(reduce-oh) = {us_ro:8.2f} us/call")
        torch._dynamo.reset()
    except Exception as e:
        print(f"  compile FAIL: {e}")

    # AOTI without max-autotune (fast compile)
    try:
        out_path = f"/workspace/aoti_cache/d{d}_b{batch}_{'fp16' if dtype==torch.float16 else 'fp32'}.pt2"
        os.makedirs("/workspace/aoti_cache", exist_ok=True)
        compile_aoti(m, x, out_path, max_autotune=False)
        loaded = aoti_load_package(out_path)
        with torch.no_grad():
            us_aoti = bench(lambda: loaded(x))
        print(f"  d={d} B={batch:<4} {'fp16' if dtype==torch.float16 else 'fp32'}: AOTI (default)           = {us_aoti:8.2f} us/call")
    except Exception as e:
        print(f"  AOTI default FAIL: {type(e).__name__}: {str(e)[:200]}")


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
        for B in [1, 64, 1024]:
            for dtype in [torch.float16]:
                run(ckpt, d, r, batch=B, dtype=dtype)
        print()


if __name__ == "__main__":
    main()
