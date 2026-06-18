"""U3 (audit pre-submission gate): re-evaluate the Table 7 narrow-H128 LER column
and the Table 8 H=192-distilled / narrow-distilled cells, which had no committed
backing JSON.

Uses the CANONICAL eval path (the same stim circuit + detector->grid tensor +
single-checkpoint LER as run_final_eval.py, which produced final_eval.json /
Table 1). NOT bench/h200_final_benchmark.py's eval_ler, whose SyndromeDataset
noise overestimates LER ~1.6x vs the canonical eval.

Run from repo root on a CUDA box.
"""
import os, sys, json, math
import numpy as np
import torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "train"))
from model import NeuralDecoder

device = torch.device("cuda")
CT = "surface_code:rotated_memory_z"
D = R = 7
P_GRID = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015]
SHOTS = 20000   # match the paper's Table 7 "20K-shot evaluation" protocol

CKPTS = {
    "full_H256":      "train/checkpoints/d7_final/best_model.pt",
    "narrow_H128":    "train/checkpoints/d7_narrow/best_model.pt",
    "narrow_distill": "train/checkpoints/d7_distill/best_model.pt",
    "h192_distill":   "train/checkpoints/d7_h192_distill/best_model.pt",
}

import stim


def wilson(k, n, z=1.96):
    p = k/n; den = 1+z*z/n
    c = (p+z*z/2/n)/den; h = z*math.sqrt(p*(1-p)/n+z*z/4/n/n)/den
    return (c-h, c+h)


class Dataset:
    """Canonical detector->(T,H,W) grid mapper, identical to run_final_eval.py."""
    def __init__(self, p):
        self.circ = stim.Circuit.generated(
            CT, distance=D, rounds=R,
            after_clifford_depolarization=p, before_measure_flip_probability=p,
            after_reset_flip_probability=p)
        self.samp = self.circ.compile_detector_sampler()
        self.nd = self.circ.num_detectors
        coords = self.circ.get_detector_coordinates()
        ac = np.array([coords[i] for i in range(self.nd)])
        sp, tm = ac[:, :-1], ac[:, -1]
        tu = np.sort(np.unique(tm)); xu = np.sort(np.unique(sp[:, 0]))
        yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
        self.grid = (len(tu), len(yu), len(xu))
        tm_m = {v: i for i, v in enumerate(tu)}
        xm = {v: i for i, v in enumerate(xu)}; ym = {v: i for i, v in enumerate(yu)}
        self.d2g = {did: (tm_m[coords[did][-1]],
                          ym.get(coords[did][1], 0) if len(coords[did]) > 2 else 0,
                          xm[coords[did][0]]) for did in range(self.nd)}

    def tensor(self, det):
        B = det.shape[0]; T, H, W = self.grid
        t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        for did, (gi, gj, gk) in self.d2g.items():
            if gi < T and gj < H and gk < W and did < det.shape[1]:
                t[:, 0, gi, gj, gk] = torch.from_numpy(det[:, did].astype(np.float32))
        return t

    def raw(self, n):
        return self.samp.sample(shots=n, separate_observables=True)


def load(path):
    ck = torch.load(os.path.join(REPO, path), weights_only=False, map_location="cpu")
    m = NeuralDecoder(ck["config"]).to(device).eval()
    m.load_state_dict(ck["model_state_dict"])
    return m, sum(p.numel() for p in m.parameters())


def ler(model, ds, det, obs, n):
    e = 0
    for i in range(0, n, 1000):
        bd, bo = det[i:i+1000], obs[i:i+1000]
        syn = ds.tensor(bd).to(device)
        lab = torch.from_numpy(bo.astype(np.float32)).to(device)
        with torch.no_grad():
            e += ((model(syn) > 0).float() != lab).any(dim=1).sum().item()
    return e


out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "shots": SHOTS, "noise": "3-parameter circuit-level (canonical, == run_final_eval.py)",
       "configs": {}}
models = {name: load(path) for name, path in CKPTS.items()}
for name, (m, np_) in models.items():
    print(f"\n=== {name} ({np_:,} params) ===")
for p in P_GRID:
    ds = Dataset(p)
    det, obs = ds.raw(SHOTS)
    det = np.asarray(det); obs = np.asarray(obs)
    for name, (m, np_) in models.items():
        k = ler(m, ds, det, obs, SHOTS)
        lo, hi = wilson(k, SHOTS)
        out["configs"].setdefault(name, {"params": np_, "path": CKPTS[name], "rates": {}})
        out["configs"][name]["rates"][f"p{p}"] = {"ler": k/SHOTS, "errors": k, "n": SHOTS, "ci": [lo, hi]}
        print(f"  p={p:.3f} {name:16s} LER={k/SHOTS*100:7.4f}% [{lo*100:.4f},{hi*100:.4f}] ({k})")

json.dump(out, open(os.path.join(REPO, "coda_experiments", "u3_table7_8_ler.json"), "w"), indent=2)
print("\nsaved coda_experiments/u3_table7_8_ler.json")
