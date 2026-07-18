"""Corrected latency measurement: real-pipeline tensor shapes (J1, Joel Pendleton's catch).

The committed 6.12/7.86 us headline numbers were measured on synthetic (B,1,d,d,d)
inputs, but the real Stim pipeline (CDS mapping, run_final_eval.py / clean_d7_eval.py)
produces (B,1,r+1,d+1,d+1) grids from detector coordinates — at d=7 r=7 that is
(8,8,8) = 512 voxels vs the synthetic 343 (1.49x). This script measures BOTH shapes
in one session so the correction receipt shows the delta directly. Latency is
weight-value-independent (random bits), but shape is load-bearing.

Usage (H200 pod, torch-2.6.0 stack):
    python bench/h200_latency_realshape.py --out bench/results/h200_latency_realshape.json
"""
import sys, os, time, json, argparse
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "train"))
sys.path.insert(0, os.path.join(REPO, "bench"))
import torch
from model import NeuralDecoder
try:
    from triton_directional import swap_to_triton
    HAVE_TRITON = True
except Exception as e:
    print("triton swap unavailable (Inductor-only rows will still run):", e)
    HAVE_TRITON = False

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

CKPTS = [
    ("d3_full_H256",   f"{REPO}/bench/results/h200_main/tuned/finetune_d3/best_model.pt", True),
    ("d5_full_H256",   f"{REPO}/bench/results/h200_main/tuned/finetune_d5/best_model.pt", True),
    ("d7_full_H256",   f"{REPO}/bench/results/h200_main/tuned/finetune_d7/best_model.pt", True),
    ("d7_H384_pfwl3s", f"{REPO}/bench/results/h200_main/tierC1/pathfinder_wide_long_d7_seed0/best_model.pt", True),
]

FALLBACK_GRID = {3: (4, 4, 4), 5: (6, 6, 6), 7: (8, 8, 8)}


def real_grid(d, r=None):
    """(T,H,W) exactly as the CDS mapping derives it from stim detector coords."""
    r = r or d
    try:
        import stim, numpy as np
        c = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=d, rounds=r,
            after_clifford_depolarization=0.003, before_measure_flip_probability=0.003,
            after_reset_flip_probability=0.003)
        coords = c.get_detector_coordinates()
        ac = np.array([coords[i] for i in range(c.num_detectors)])
        sp, tm = ac[:, :-1], ac[:, -1]
        tu = np.unique(tm); xu = np.unique(sp[:, 0])
        yu = np.unique(sp[:, 1]) if sp.shape[1] > 1 else np.array([0.0])
        return (len(tu), len(yu), len(xu))
    except Exception as e:
        print(f"stim unavailable ({e}); using fallback grid for d={d}")
        return FALLBACK_GRID[d]


def load_fp16(path):
    ck = torch.load(path, weights_only=False, map_location="cuda")
    m = NeuralDecoder(ck["config"]).cuda().eval()
    m.load_state_dict(ck["model_state_dict"])
    d = getattr(ck["config"], "distance", 7)
    return m.half(), d


def bench(model, shape, B, trials=5, iters=500, warmup=100):
    T, H, W = shape
    x = torch.randint(0, 2, (B, 1, T, H, W), dtype=torch.float16, device="cuda")
    torch._dynamo.reset()
    mc = torch.compile(model, mode="max-autotune")
    with torch.no_grad():
        for _ in range(warmup): _ = mc(x)
    torch.cuda.synchronize()
    vals = []
    for _ in range(trials):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters): _ = mc(x)
        torch.cuda.synchronize(); vals.append((time.perf_counter() - t0) * 1e6 / iters)
    torch._dynamo.reset()
    return min(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/h200_latency_realshape.json")
    a = ap.parse_args()
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "cuda": torch.version.cuda,
           "note": "realshape = (r+1, d+1, d+1) from stim detector coords (CDS mapping); "
                   "synthetic = legacy (d,d,d) convention of the committed 6.12/7.86 headline",
           "rows": []}
    print(f"stack: torch {torch.__version__} / CUDA {torch.version.cuda} / {out['gpu']}", flush=True)
    for label, path, want_tri in CKPTS:
        if not os.path.exists(path):
            print(f"[skip] {label}: missing {path}", flush=True); continue
        modes = [False, True] if (want_tri and HAVE_TRITON) else [False]
        for tri in modes:
            m, d = load_fp16(path)
            if tri:
                swap_to_triton(m); m = m.cuda().half().eval()
            shapes = {"synthetic": (d, d, d), "realshape": real_grid(d)}
            row = {"label": label, "triton": tri, "distance": d,
                   "grids": {k: list(v) for k, v in shapes.items()},
                   "params": sum(p.numel() for p in m.parameters())}
            for sk, shp in shapes.items():
                for B in [1, 1024]:
                    us = bench(m, shp, B)
                    row[f"{sk}_B{B}_us_call"] = round(us, 2)
                    row[f"{sk}_B{B}_us_syn"] = round(us / B, 3)
                    print(f"  {label} triton={tri} {sk}{shp} B={B}: "
                          f"{us:.2f} us/call ({us/B:.3f} us/syn)", flush=True)
            out["rows"].append(row)
            json.dump(out, open(a.out, "w"), indent=2)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
