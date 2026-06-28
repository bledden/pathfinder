"""Clean single-stack latency re-measurement for §5.3 Tables 3a/3b/3d/8 (and the
§5.13 H=384 latency table). Run this on ONE torch-2.6.0 + CUDA H200 stack so every
row shares a measurement environment — this resolves the §5.3 "version-sensitivity /
pre-submission re-measurement" caveat. The JSON self-documents the stack (torch + CUDA
+ GPU) so the artifact is auditable.

Latency is weight-value-independent; we still load the real checkpoints so the
architecture (distance, hidden_dim, depth) is exactly the deployed one. Input-tensor
shape (B,1,d,d,d) matches the convention used for the committed 6.12/7.86 µs headline.

Usage (on the H200 pod, torch 2.6.0 stack):
    python bench/h200_latency_clean.py --out bench/results/h200_latency_clean.json
Edit CKPTS below if your pod's checkpoint paths differ. Rows whose checkpoint is
missing are skipped (printed), so it degrades gracefully.
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

# (label, checkpoint path, also-measure-with-Triton). Adjust paths to your pod layout.
CKPTS = [
    ("d3_full_H256",   f"{REPO}/bench/results/h200_main/tuned/finetune_d3/best_model.pt", True),
    ("d5_full_H256",   f"{REPO}/bench/results/h200_main/tuned/finetune_d5/best_model.pt", True),
    ("d7_full_H256",   f"{REPO}/bench/results/h200_main/tuned/finetune_d7/best_model.pt", True),
    ("d7_narrow_H128", f"{REPO}/train/checkpoints/d7_narrow/best_model.pt",               True),
    ("d7_H192",        f"{REPO}/train/checkpoints/d7_distill/best_model.pt",              True),
    ("d7_H384_pfwl3s", f"{REPO}/bench/results/h200_main/tierC1/pathfinder_wide_long_d7_seed0/best_model.pt", True),
]


def load_fp16(path):
    ck = torch.load(path, weights_only=False, map_location="cuda")
    m = NeuralDecoder(ck["config"]).cuda().eval()
    m.load_state_dict(ck["model_state_dict"])
    d = getattr(ck["config"], "distance", 7)
    return m.half(), d


def bench(model, d, B, trials=5, iters=500, warmup=100):
    x = torch.randint(0, 2, (B, 1, d, d, d), dtype=torch.float16, device="cuda")
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
    ap.add_argument("--out", default="bench/results/h200_latency_clean.json")
    a = ap.parse_args()
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "cuda": torch.version.cuda, "rows": []}
    print(f"stack: torch {torch.__version__} / CUDA {torch.version.cuda} / {out['gpu']}", flush=True)
    for label, path, want_tri in CKPTS:
        if not os.path.exists(path):
            print(f"[skip] {label}: missing {path}", flush=True); continue
        modes = [False, True] if (want_tri and HAVE_TRITON) else [False]
        for tri in modes:
            m, d = load_fp16(path)
            if tri:
                swap_to_triton(m); m = m.cuda().half().eval()
            row = {"label": label, "triton": tri, "distance": d,
                   "params": sum(p.numel() for p in m.parameters())}
            for B in [1, 64, 1024]:
                us = bench(m, d, B)
                row[f"B{B}_us_call"] = round(us, 2)
                row[f"B{B}_us_syn"] = round(us / B, 3)
                print(f"  {label} triton={tri} B={B}: {us:.2f} us/call ({us/B:.3f} us/syn)", flush=True)
            out["rows"].append(row)
            json.dump(out, open(a.out, "w"), indent=2)  # checkpoint after each row
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
