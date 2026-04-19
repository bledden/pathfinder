import sys, os
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace")
import torch, numpy as np
import pymatching
from model import NeuralDecoder
from data import SyndromeDataset, DataConfig


def load_fp16(path):
    ck = torch.load(path, weights_only=False, map_location="cuda")
    m = NeuralDecoder(ck["config"]).cuda().eval()
    m.load_state_dict(ck["model_state_dict"])
    return m.half()


m128 = load_fp16("/workspace/pathfinder/train/checkpoints/d7_distill/best_model.pt")
m192 = load_fp16("/workspace/pathfinder/train/checkpoints/d7_h192_distill/best_model.pt")
mfull = load_fp16("/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt")

print("=== Ensemble: narrow_distill (H=128) + PM ===")
print(f"{'p':>6} {'neural':>9} {'PM':>9} {'agree%':>8} {'OR':>9} {'confT=2':>10} {'confT=5':>10}")


def run_ensemble(model, p, n_shots=20000):
    ds = SyndromeDataset(DataConfig(distance=7, rounds=7, physical_error_rate=p))
    circuit = ds.circuit
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()

    ne, pe, oe, ae, e2, e5, total = 0, 0, 0, 0, 0, 0, 0
    batch = 500
    for _ in range(n_shots // batch):
        det, obs = sampler.sample(batch, separate_observables=True)
        det_u8 = det.astype(np.uint8)
        obs_u8 = obs.astype(np.uint8)
        synd_tensor = ds.detectors_to_tensor(det_u8).float().cuda().half()
        with torch.no_grad():
            logits = model(synd_tensor).cpu().float().numpy()
        preds_n = (logits > 0).astype(np.uint8)
        preds_p = matching.decode_batch(det_u8)

        for i in range(batch):
            n_right = (preds_n[i] == obs_u8[i]).all()
            p_right = (preds_p[i] == obs_u8[i]).all()
            if not n_right: ne += 1
            if not p_right: pe += 1
            if (preds_n[i] == preds_p[i]).all(): ae += 1
            if not (n_right or p_right): oe += 1
            conf = abs(logits[i][0])
            c2 = preds_n[i] if conf > 2.0 else preds_p[i]
            c5 = preds_n[i] if conf > 5.0 else preds_p[i]
            if not (c2 == obs_u8[i]).all(): e2 += 1
            if not (c5 == obs_u8[i]).all(): e5 += 1
        total += batch
    return dict(n=ne/total, p=pe/total, agree=ae/total*100, or_o=oe/total, e2=e2/total, e5=e5/total, total=total)


for p in [0.003, 0.005, 0.007, 0.010]:
    r = run_ensemble(m128, p)
    print(f"{p:>6.3f} {r['n']:>9.5f} {r['p']:>9.5f} {r['agree']:>7.2f}% {r['or_o']:>9.5f} {r['e2']:>10.5f} {r['e5']:>10.5f}")

print()
print("=== Ensemble: H=192 distilled + PM ===")
print(f"{'p':>6} {'neural':>9} {'PM':>9} {'agree%':>8} {'OR':>9} {'confT=2':>10} {'confT=5':>10}")
for p in [0.003, 0.005, 0.007, 0.010]:
    r = run_ensemble(m192, p)
    print(f"{p:>6.3f} {r['n']:>9.5f} {r['p']:>9.5f} {r['agree']:>7.2f}% {r['or_o']:>9.5f} {r['e2']:>10.5f} {r['e5']:>10.5f}")

print()
print("=== Ensemble: FULL H=256 + PM (replicate paper 5.6) ===")
print(f"{'p':>6} {'neural':>9} {'PM':>9} {'agree%':>8} {'OR':>9} {'confT=2':>10} {'confT=5':>10}")
for p in [0.005, 0.007]:
    r = run_ensemble(mfull, p)
    print(f"{p:>6.3f} {r['n']:>9.5f} {r['p']:>9.5f} {r['agree']:>7.2f}% {r['or_o']:>9.5f} {r['e2']:>10.5f} {r['e5']:>10.5f}")
