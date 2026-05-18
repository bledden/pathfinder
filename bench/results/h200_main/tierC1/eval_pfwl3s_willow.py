"""Eval PFWL3S + PM on Willow real-hardware d=7 r13 detection events.

Willow uses compound detectors (L=6 coords = round-comparison; L=9/L=15 = boundary
combos). For Pathfinder's input tensor we use the FIRST 3 coords (x, y, t) of
each detector. Acknowledged limitation: Pathfinder was trained on standard
Stim `surface_code:rotated_memory_z` detectors (all L=3), so the Willow circuit's
compound-detector format is out-of-distribution. Plus Pathfinder trained at
R=d=7 (T=8 timepoints); Willow data is R=13 (T=14 timepoints) — we truncate to
T=8 for in-distribution input shape.
"""
import sys, os, json
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace/GNN_decoder")
import numpy as np
import torch, stim, pymatching
from model import NeuralDecoder
from ensemble_pf_lange import wilson

device = torch.device("cuda")

PFW_CKPTS = [
    "/workspace/persist/checkpoints/pathfinder_wide_long_d7/best_model.pt",
    "/workspace/persist/checkpoints/pathfinder_wide_long_d7_seed1/best_model.pt",
    "/workspace/persist/checkpoints/pathfinder_wide_long_d7_seed2/best_model.pt",
]
DATA = "/workspace/persist/willow_data/willow_105Q/d7_z_r13"

def load_pf(paths):
    models = []
    for p in paths:
        ck = torch.load(p, weights_only=False, map_location=device)
        m = NeuralDecoder(ck["config"]).to(device); m.load_state_dict(ck["model_state_dict"]); m.eval()
        models.append(m)
    return models

def pf_predict_avg(models, syn):
    with torch.no_grad():
        avg = None
        for m in models:
            lg = m(syn).cpu().numpy()
            avg = lg if avg is None else avg + lg
        return ((avg / len(models)) > 0).astype(np.uint8)

circ = stim.Circuit.from_file(DATA + "/circuit_noisy_si1000.stim")
det_events = stim.read_shot_data_file(path=DATA + "/detection_events.b8",
                                       format="b8", num_detectors=circ.num_detectors,
                                       num_observables=0, bit_packed=False)
obs_flips = stim.read_shot_data_file(path=DATA + "/obs_flips_actual.b8",
                                      format="b8", num_observables=circ.num_observables,
                                      num_detectors=0, bit_packed=False)
det = det_events.astype(np.uint8)
obs = obs_flips.astype(np.uint8)
N = det.shape[0]
print(f"Loaded Willow real-hw d=7 r13 Z basis: {N} shots, {det.shape[1]} detectors, {obs.shape[1]} observables")

# PM on full r13 data
dem = circ.detector_error_model(decompose_errors=True)
pm = pymatching.Matching.from_detector_error_model(dem)
pm_pred = pm.decode_batch(det).astype(np.uint8)
pm_wrong = np.any(pm_pred != obs, axis=1)
pm_ler, pm_lo, pm_hi = wilson(int(pm_wrong.sum()), N)
print(f"PyMatching on real Willow d=7 r13 ({N} shots): LER = {pm_ler*100:.4f}% [{pm_lo*100:.4f}, {pm_hi*100:.4f}]")

# Pathfinder spatial mapper — use first 3 coords of each detector
coords = circ.get_detector_coordinates()
nd = circ.num_detectors
det_xyt = np.zeros((nd, 3), dtype=np.float32)
for i in range(nd):
    c = coords[i]
    det_xyt[i] = [c[0], c[1], c[2]]

xu = np.sort(np.unique(det_xyt[:, 0]))
yu = np.sort(np.unique(det_xyt[:, 1]))
tu = np.sort(np.unique(det_xyt[:, 2]))
print(f"Willow grid: T={len(tu)} timepoints, H={len(yu)} y, W={len(xu)} x")

x_map = {v: i for i, v in enumerate(xu)}
y_map = {v: i for i, v in enumerate(yu)}
t_map = {v: i for i, v in enumerate(tu)}
det_idx = np.zeros((nd, 3), dtype=np.int64)
for i in range(nd):
    det_idx[i] = [t_map[det_xyt[i, 2]], y_map[det_xyt[i, 1]], x_map[det_xyt[i, 0]]]

# Truncate to first T_train = 8 timepoints (matches R=d=7 training)
T_train = 8
keep = det_idx[:, 0] < T_train
print(f"Truncating to first T={T_train}: keeping {int(keep.sum())}/{nd} detectors")
det_t = det[:, keep]
det_idx_t = det_idx[keep]
H, W = len(yu), len(xu)

def build_syndrome_tensor(det_subset, det_idx_subset, T, H, W):
    B = det_subset.shape[0]
    t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
    nd = det_subset.shape[1]
    for i in range(nd):
        t[:, 0, det_idx_subset[i, 0], det_idx_subset[i, 1], det_idx_subset[i, 2]] = torch.from_numpy(det_subset[:, i].astype(np.float32))
    return t

print(f"Loading PFWL3S 3-seed avg...")
pf_models = load_pf(PFW_CKPTS)

pf_pred = np.zeros((N, 1), dtype=np.uint8)
CHUNK = 500
for i in range(0, N, CHUNK):
    syn = build_syndrome_tensor(det_t[i:i+CHUNK], det_idx_t, T_train, H, W).to(device)
    pf_pred[i:i+CHUNK] = pf_predict_avg(pf_models, syn)

pf_wrong = np.any(pf_pred != obs, axis=1)
pf_ler, pf_lo, pf_hi = wilson(int(pf_wrong.sum()), N)
print(f"PFWL3S (3-seed avg, truncated T={T_train}) on real Willow d=7: LER = {pf_ler*100:.4f}% [{pf_lo*100:.4f}, {pf_hi*100:.4f}]")

# Triad: PF + PM (no Lange yet)
both_pred = ((pf_pred.astype(int) + pm_pred.astype(int)) >= 1).astype(np.uint8)  # OR-oracle
triad_2way_pred = np.where(pf_pred == pm_pred, pf_pred, pm_pred)  # if disagree, PM tie-breaks
oracle_wrong = np.any(both_pred != obs, axis=1) & np.any(pm_pred != obs, axis=1) & np.any(pf_pred != obs, axis=1)
both_wrong = pf_wrong & pm_wrong
or_lb = int(both_wrong.sum())
or_ler, or_lo, or_hi = wilson(or_lb, N)
print(f"OR-oracle (both PF and PM wrong): {or_ler*100:.4f}% [{or_lo*100:.4f}, {or_hi*100:.4f}]")

results = {
    "dataset": "Willow d=7 r13 Z basis (real Sycamore hardware data, Nature 2024)",
    "n_shots": int(N),
    "circuit_path": DATA + "/circuit_noisy_si1000.stim",
    "pm_ler": float(pm_ler), "pm_ci": [float(pm_lo), float(pm_hi)],
    "pfwl3s_ler": float(pf_ler), "pfwl3s_ci": [float(pf_lo), float(pf_hi)],
    "or_oracle_ler": float(or_ler), "or_oracle_ci": [float(or_lo), float(or_hi)],
    "pf_truncated_to_T": T_train,
    "willow_R": 13, "pf_train_R": 7,
    "note": ("PFWL3S trained at R=d=7 (T=8 timepoints) and on standard Stim "
             "surface_code:rotated_memory_z 4-parameter noise. Willow data is "
             "R=13 (T=14) on Sycamore SI1000 noise with compound detectors "
             "(L=6 round-comparison). Truncated to first T=8 timepoints for "
             "in-distribution input shape; used first 3 coords of each "
             "detector as (x,y,t) mapping. Both Pathfinder's noise-model and "
             "detector-format are out-of-distribution relative to training."),
}
os.makedirs("/workspace/persist/results", exist_ok=True)
with open("/workspace/persist/results/willow_eval_d7.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved /workspace/persist/results/willow_eval_d7.json")
