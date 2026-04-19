"""Empirical test: is the memory-Z logical observable invariant under row/col flips
of the syndrome tensor? Test by comparing label rates across flipped and unflipped."""
import sys, numpy as np, stim, torch
sys.path.insert(0, "/workspace/pathfinder/train")

# Get a syndrome batch from Stim's memory-Z circuit
d = 5; p = 0.007
circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
    after_clifford_depolarization=p, before_measure_flip_probability=p,
    after_reset_flip_probability=p, before_round_data_depolarization=p)
sampler = circuit.compile_detector_sampler(seed=42)
det, obs = sampler.sample(shots=100000, separate_observables=True)
det = det.astype(np.uint8)
obs = obs.astype(np.uint8)

# Map detectors to 3D grid (Pathfinder-style)
from data import SyndromeDataset, DataConfig

# Build the tensor ourselves
coords = circuit.get_detector_coordinates()
nd = circuit.num_detectors
ac = np.array([coords[i] for i in range(nd)])
sp, tm = ac[:, :-1], ac[:, -1]
tu = np.sort(np.unique(tm))
xu = np.sort(np.unique(sp[:, 0]))
yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
T, H, W = len(tu), len(yu), len(xu)
tm_m = {v: i for i, v in enumerate(tu)}; xm = {v: i for i, v in enumerate(xu)}; ym = {v: i for i, v in enumerate(yu)}
det_idx = np.zeros((nd, 3), dtype=np.int64)
for did in range(nd):
    c = coords[did]
    det_idx[did] = [tm_m[c[-1]], ym.get(c[1], 0) if len(c) > 2 else 0, xm[c[0]]]
print(f"Grid shape: T={T}, H={H}, W={W}, n_det={nd}")

# Now: for a given syndrome s, flip it along R and C axes.
# Is the resulting flipped syndrome's correct logical label the same as original?
# To check, we need another way to compute the label — use PyMatching as a proxy.
import pymatching
dem = circuit.detector_error_model(decompose_errors=True)
pm = pymatching.Matching.from_detector_error_model(dem)

# Original predictions
preds_orig = pm.decode_batch(det)

# NOW — to test the geometric flip hypothesis, I need to:
# 1. Reshape det into 3D syndrome tensor [B, T, H, W]
# 2. Flip along axes
# 3. Un-reshape back to flat detector order
# 4. Decode with PM
# 5. Compare to obs

# But this only works if the flipped syndrome is STILL a valid Stim detection event 
# for THE SAME circuit. If the flip maps to a different circuit's syndrome,
# then the PM baseline for comparison is wrong.

# A cleaner test: check if Pathfinder's prediction is invariant under flip.
from model import NeuralDecoder
ck = torch.load("/workspace/pathfinder/train/checkpoints/d5_muon/best_model.pt", weights_only=False, map_location="cuda")
m = NeuralDecoder(ck["config"]).cuda()
m.load_state_dict(ck["model_state_dict"])
m.eval()

# Build 3D syndrome for Pathfinder
def to_tensor(det_batch):
    B = det_batch.shape[0]
    t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
    d = torch.from_numpy(det_batch.astype(np.float32))
    for i in range(nd):
        t[:, 0, det_idx[i, 0], det_idx[i, 1], det_idx[i, 2]] = d[:, i]
    return t

# Get Pathfinder logits
bs = 1000
det_small = det[:bs]
obs_small = obs[:bs]

syn = to_tensor(det_small).cuda()
with torch.no_grad():
    lg_orig = m(syn).cpu().numpy()
preds_orig_pf = (lg_orig > 0).astype(np.uint8)

# Flip along R axis (axis -2 of syn)
syn_r = torch.flip(syn, dims=[-2])
with torch.no_grad():
    lg_r = m(syn_r).cpu().numpy()
preds_r_pf = (lg_r > 0).astype(np.uint8)

syn_c = torch.flip(syn, dims=[-1])
with torch.no_grad():
    lg_c = m(syn_c).cpu().numpy()
preds_c_pf = (lg_c > 0).astype(np.uint8)

syn_rc = torch.flip(syn, dims=[-2, -1])
with torch.no_grad():
    lg_rc = m(syn_rc).cpu().numpy()
preds_rc_pf = (lg_rc > 0).astype(np.uint8)

print("\n=== Pathfinder (unaugmented v1) predictions under flips ===")
print(f"Accuracy orig:   {100 * np.mean(preds_orig_pf.flatten() == obs_small.flatten()):.2f}%")
print(f"Accuracy R-flip: {100 * np.mean(preds_r_pf.flatten() == obs_small.flatten()):.2f}%   (would be same if task invariant)")
print(f"Accuracy C-flip: {100 * np.mean(preds_c_pf.flatten() == obs_small.flatten()):.2f}%")
print(f"Accuracy both:   {100 * np.mean(preds_rc_pf.flatten() == obs_small.flatten()):.2f}%")

# Prediction agreement: how often does the flipped pred match the original
print(f"\nPrediction agreement (unflipped vs flipped prediction, IF task is invariant should be ~100%):")
print(f"orig vs R-flip:  {100 * np.mean(preds_orig_pf == preds_r_pf):.2f}%")
print(f"orig vs C-flip:  {100 * np.mean(preds_orig_pf == preds_c_pf):.2f}%")
print(f"orig vs both:    {100 * np.mean(preds_orig_pf == preds_rc_pf):.2f}%")

# A cleaner test: if the task is invariant, logit magnitude should be similar
print(f"\nLogit statistics:")
print(f"orig:   mean={lg_orig.mean():.4f}  |lg| mean={np.abs(lg_orig).mean():.4f}")
print(f"R-flip: mean={lg_r.mean():.4f}  |lg| mean={np.abs(lg_r).mean():.4f}")
print(f"C-flip: mean={lg_c.mean():.4f}  |lg| mean={np.abs(lg_c).mean():.4f}")
print(f"both:   mean={lg_rc.mean():.4f}  |lg| mean={np.abs(lg_rc).mean():.4f}")
