"""Re-decode IBM d=5 r=5 data with proper PFWL3S 5-seed ensemble (H=384, all 5 seeds).

The earlier decode used finetune_d5 (H=256, single seed) which is NOT PFWL3S.
This uses the actual PFWL3S 5-seed checkpoints at the matching distribution.
"""
import sys, json, math
import numpy as np
import stim
import pymatching
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'train')

from decode_ibm_result import bitarray_packed_to_bools, wilson_ci
from redecode_ibm_d3r3_pfwl3s import PathfinderMapper, run_pf_seed
from model import NeuralDecoder

PFWL3S_SEEDS = [
    f'bench/results/h200_main/tierC1/pathfinder_wide_long_d5_seed{i}/best_model.pt'
    for i in range(5)
]
D, R = 5, 5


def main():
    obj = json.load(open(f'ibm_d{D}r{R}_result.json'))
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

    # PM with DEM swept
    pm_results = {}
    for p_dem in [0.005, 0.010, 0.015, 0.020, 0.025]:
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
        k = int(pm_wrong.sum())
        ler, lo, hi = wilson_ci(k, n_shots)
        pm_results[f"p{p_dem}"] = {"ler": ler, "ci": [lo, hi], "errors": k}
        print(f"  PM (p_dem={p_dem:.3f}): LER={ler*100:6.3f}% CI=[{lo*100:.3f}%, {hi*100:.3f}%]")

    # use best PM (lowest LER over sweep)
    best_p = min(pm_results.keys(), key=lambda k: pm_results[k]["ler"])
    pm_ler = pm_results[best_p]["ler"]
    pm_lo, pm_hi = pm_results[best_p]["ci"]
    pm_k = pm_results[best_p]["errors"]
    print(f"  >> best PM at {best_p}: LER={pm_ler*100:.3f}%")

    # PFWL3S 5-seed
    print(f"\nRunning PFWL3S 5 seeds at d={D} r={R}, H=384...")
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

    mean_logits = all_logits.mean(axis=0)
    pf_ens_pred = (mean_logits > 0).astype(np.uint8).reshape(-1, 1)
    pf_ens_wrong = np.any(pf_ens_pred != obs, axis=1)
    pf_ens_k = int(pf_ens_wrong.sum())
    pf_ens_ler, pf_ens_lo, pf_ens_hi = wilson_ci(pf_ens_k, n_shots)

    print(f"\n=== IBM Heron r2 d={D} r={R} (n={n_shots}) — proper comparison ===")
    print(f"detector_flip_rate:   {det.mean():.4f}")
    print(f"observable_flip_rate: {obs.mean():.4f}")
    print()
    print(f"PM (best p_dem={best_p}):              LER={pm_ler*100:6.3f}% CI=[{pm_lo*100:.3f}%, {pm_hi*100:.3f}%] ({pm_k} errors)")
    print(f"PFWL3S 5-seed (logit avg):           LER={pf_ens_ler*100:6.3f}% CI=[{pf_ens_lo*100:.3f}%, {pf_ens_hi*100:.3f}%] ({pf_ens_k} errors)")
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
        "pymatching_sweep": pm_results,
        "pymatching_best": {"p_dem": best_p, "ler": pm_ler, "ci": [pm_lo, pm_hi], "errors": pm_k},
        "pfwl3s_5seed": {
            "ler": pf_ens_ler, "ci": [pf_ens_lo, pf_ens_hi],
            "errors": pf_ens_k, "per_seed_errors": per_seed_errs,
            "ckpts": PFWL3S_SEEDS, "ensemble_method": "logit_mean",
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
