"""Re-decode IBM d=3 r=3 data with proper PFWL3S 3-seed ensemble (H=384, rescue seeds).

The earlier decode used finetune_d3 (H=256, single seed) which is NOT PFWL3S.
This uses the actual PFWL3S 3-seed rescue checkpoints at the matching distribution
(d=3 r=3), so we can fairly compare PF vs PM on real IBM hardware data.
"""
import sys, json, math
import numpy as np
import stim
import pymatching
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'train')

from decode_ibm_result import bitarray_packed_to_bools, wilson_ci, run_decoders
from model import NeuralDecoder

# rescue seeds — fully converged at LER ~3% on Stim, H=384, d=3 r=3
PFWL3S_SEEDS = [
    'bench/results/h200_main/tierC1/pathfinder_wide_long_d3_rescue_seed0/best_model.pt',
    'bench/results/h200_main/tierC1/pathfinder_wide_long_d3_rescue_seed1/best_model.pt',
    'bench/results/h200_main/tierC1/pathfinder_wide_long_d3_rescue_seed2/best_model.pt',
]

D, R = 3, 3


class PathfinderMapper:
    """Map Stim detectors → (T, H, W) tensor for the CNN."""
    def __init__(self, circuit):
        nd = circuit.num_detectors
        coords = circuit.get_detector_coordinates()
        ac = np.array([coords[i] for i in range(nd)])
        sp, tm = ac[:, :-1], ac[:, -1]
        tu = np.sort(np.unique(tm))
        xu = np.sort(np.unique(sp[:, 0]))
        yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
        self.grid = (len(tu), len(yu), len(xu))
        tm_m = {v: i for i, v in enumerate(tu)}
        xm = {v: i for i, v in enumerate(xu)}
        ym = {v: i for i, v in enumerate(yu)}
        di = np.zeros((nd, 3), dtype=np.int64)
        for did in range(nd):
            c = coords[did]
            di[did] = [tm_m[c[-1]], ym.get(c[1], 0) if len(c) > 2 else 0, xm[c[0]]]
        self.det_idx = di
        self.nd = nd

    def to_tensor(self, det):
        B = det.shape[0]
        T, H, W = self.grid
        t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        d = torch.from_numpy(det.astype(np.float32))
        for i in range(self.nd):
            t[:, 0, self.det_idx[i, 0], self.det_idx[i, 1], self.det_idx[i, 2]] = d[:, i]
        return t


def run_pf_seed(circ, detectors, observables, ckpt_path):
    device = torch.device('cpu')
    pfm = PathfinderMapper(circ)
    ck = torch.load(ckpt_path, weights_only=False, map_location=device)
    model = NeuralDecoder(ck['config']).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.train(False)
    n_shots = detectors.shape[0]
    pf_pred = np.zeros_like(observables)
    chunk = 500
    det_u8 = detectors.astype(np.uint8)
    logits_all = np.zeros(n_shots, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n_shots, chunk):
            end = min(i+chunk, n_shots)
            t = pfm.to_tensor(det_u8[i:end]).to(device)
            logits = model(t).cpu().numpy().squeeze(-1)
            logits_all[i:end] = logits
            pf_pred[i:end, 0] = (logits > 0).astype(np.uint8)
    return pf_pred, logits_all


def main():
    obj = json.load(open('ibm_d5r1_result.json'.replace('d5r1', f'd{D}r{R}')))
    r = obj["result"]
    packed = np.array(r["shots_array_packed"], dtype=np.uint8)
    measurements = bitarray_packed_to_bools(packed, r["n_shots"], r["n_bits"])
    print(f"IBM d={D} r={R}: {measurements.shape[0]} shots, {measurements.shape[1]} bits")

    clean = stim.Circuit.generated('surface_code:rotated_memory_z', distance=D, rounds=R)
    m2d = clean.compile_m2d_converter()
    det, obs = m2d.convert(measurements=measurements, separate_observables=True)
    n_shots = det.shape[0]
    print(f"detectors: {det.shape}, observables: {obs.shape}")
    print(f"detector_flip_rate: {det.mean():.4f}  observable_flip_rate: {obs.mean():.4f}")

    # PM baseline (with DEM at effective IBM noise — match per-cell calibration)
    # Use p that approximately matches the chip's detector rate
    for p_dem in [0.005, 0.010, 0.015, 0.020]:
        dem_circ = stim.Circuit.generated(
            'surface_code:rotated_memory_z', distance=D, rounds=R,
            after_clifford_depolarization=p_dem,
            before_measure_flip_probability=p_dem,
            after_reset_flip_probability=p_dem,
            before_round_data_depolarization=p_dem,
        )
        dem = dem_circ.detector_error_model(decompose_errors=True)
        pm = pymatching.Matching.from_detector_error_model(dem)
        pm_pred = pm.decode_batch(det).astype(np.uint8)
        pm_wrong = np.any(pm_pred != obs, axis=1)
        ler, lo, hi = wilson_ci(int(pm_wrong.sum()), n_shots)
        print(f"  PM (p_dem={p_dem:.3f}): LER={ler*100:6.3f}% CI=[{lo*100:.3f}%, {hi*100:.3f}%]")

    # use p=0.010 (closest to chip det rate of 27.9%)
    p_dem = 0.010
    dem_circ = stim.Circuit.generated(
        'surface_code:rotated_memory_z', distance=D, rounds=R,
        after_clifford_depolarization=p_dem,
        before_measure_flip_probability=p_dem,
        after_reset_flip_probability=p_dem,
        before_round_data_depolarization=p_dem,
    )
    dem = dem_circ.detector_error_model(decompose_errors=True)
    pm = pymatching.Matching.from_detector_error_model(dem)
    pm_pred = pm.decode_batch(det).astype(np.uint8)
    pm_wrong = np.any(pm_pred != obs, axis=1)
    pm_k = int(pm_wrong.sum())
    pm_ler, pm_lo, pm_hi = wilson_ci(pm_k, n_shots)

    # PFWL3S 3-seed ensemble (logit average → sign)
    print(f"\nRunning PFWL3S 3 seeds at d={D} r={R}, H=384 (rescue checkpoints)...")
    all_logits = np.zeros((len(PFWL3S_SEEDS), n_shots), dtype=np.float32)
    per_seed_errs = []
    for i, ckpt in enumerate(PFWL3S_SEEDS):
        pf_pred, logits = run_pf_seed(clean, det, obs, ckpt)
        all_logits[i] = logits
        wrong = np.any(pf_pred != obs, axis=1)
        k = int(wrong.sum())
        ler_s = k / n_shots
        per_seed_errs.append(k)
        print(f"  seed{i}: errors={k}/{n_shots} LER={ler_s*100:6.3f}%")

    # ensemble: average logits, threshold at 0
    mean_logits = all_logits.mean(axis=0)
    pf_ens_pred = (mean_logits > 0).astype(np.uint8).reshape(-1, 1)
    pf_ens_wrong = np.any(pf_ens_pred != obs, axis=1)
    pf_ens_k = int(pf_ens_wrong.sum())
    pf_ens_ler, pf_ens_lo, pf_ens_hi = wilson_ci(pf_ens_k, n_shots)

    print(f"\n=== IBM Heron r2 d={D} r={R} (n={n_shots}) — proper comparison ===")
    print(f"detector_flip_rate:   {det.mean():.4f}")
    print(f"observable_flip_rate: {obs.mean():.4f}")
    print()
    print(f"PM (p_dem={p_dem:.3f}):                  LER={pm_ler*100:6.3f}% CI=[{pm_lo*100:.3f}%, {pm_hi*100:.3f}%] ({pm_k} errors)")
    print(f"PFWL3S 3-seed ensemble (logit avg):  LER={pf_ens_ler*100:6.3f}% CI=[{pf_ens_lo*100:.3f}%, {pf_ens_hi*100:.3f}%] ({pf_ens_k} errors)")
    print(f"per-seed errors: {per_seed_errs}")

    delta = pm_ler - pf_ens_ler
    rel = delta / pm_ler * 100 if pm_ler > 0 else 0
    print(f"\nΔ(PM - PF) = {delta*100:+.3f}pp  ({rel:+.1f}% relative)")
    if pf_ens_hi < pm_lo:
        print(">> PFWL3S STRICTLY beats PM (CIs disjoint)")
    elif pf_ens_lo > pm_hi:
        print(">> PM STRICTLY beats PFWL3S (CIs disjoint)")
    else:
        print(">> PF and PM CIs overlap — tie within noise")

    out = {
        "distance": D, "rounds": R, "n_shots": n_shots,
        "detector_flip_rate": float(det.mean()),
        "observable_flip_rate": float(obs.mean()),
        "p_dem_used": p_dem,
        "pymatching": {"ler": pm_ler, "ci": [pm_lo, pm_hi], "errors": pm_k},
        "pfwl3s_3seed": {
            "ler": pf_ens_ler, "ci": [pf_ens_lo, pf_ens_hi],
            "errors": pf_ens_k,
            "per_seed_errors": per_seed_errs,
            "ckpts": PFWL3S_SEEDS,
            "ensemble_method": "logit_mean",
        },
        "delta_pp": delta * 100,
        "delta_relative_pct": rel,
        "strict_winner": "PFWL3S" if pf_ens_hi < pm_lo else ("PM" if pf_ens_lo > pm_hi else "tie"),
    }
    with open(f"ibm_d{D}r{R}_pfwl3s_proper.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: ibm_d{D}r{R}_pfwl3s_proper.json")


if __name__ == "__main__":
    main()
