"""U1 (audit pre-submission gate): re-decode IBM d=5 r=5 at 100K shots with the
CALIBRATED 3-seed PFWL3S (the model the paper's Table 13a row 3 actually reports),
to test whether the 10K-shot tie with PyMatching survives at tight CIs.

Paper §5.15.1 reports d5r5 as a 10K-shot tie (calibrated 47.27% vs PM 45.68%).
The 100K IBM data is on disk; at 100K the ~1.6pp point gap may resolve a small PM
advantage. Either outcome is reported honestly.

Run from anywhere; paths resolve relative to the repo root (this file's grandparent).
"""
import os, sys, json
import numpy as np
import stim, pymatching, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "coda_experiments"))
sys.path.insert(0, os.path.join(REPO, "train"))

from decode_ibm_result import bitarray_packed_to_bools, wilson_ci
from redecode_ibm_d3r3_pfwl3s import PathfinderMapper
from model import NeuralDecoder

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_pf_seed_gpu(circ, det, obs, ckpt_path, chunk=4096):
    """GPU port of redecode_ibm_d3r3_pfwl3s.run_pf_seed (which hardcodes CPU).
    Architecture is read from the checkpoint's saved config, so H=384 loads correctly."""
    pfm = PathfinderMapper(circ)
    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    model = NeuralDecoder(ck["config"]).to(DEV).eval()
    model.load_state_dict(ck["model_state_dict"])
    n = det.shape[0]
    det_u8 = det.astype(np.uint8)
    logits = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, chunk):
            t = pfm.to_tensor(det_u8[i:i+chunk]).to(DEV)
            logits[i:i+chunk] = model(t).float().cpu().numpy().squeeze(-1)
    pred = (logits > 0).astype(np.uint8).reshape(-1, 1)
    return pred, logits

D, R = 5, 5
CALIB_SEEDS = [os.path.join(REPO, f"bench/results/h200_main/calibrated/seed{i}/best_model.pt")
               for i in range(3)]
DATA = os.path.join(REPO, "coda_experiments", "ibm_d5r5_100k_result.json")
OUT = os.path.join(REPO, "coda_experiments", "ibm_d5r5_calibrated_100k_eval.json")


def main():
    obj = json.load(open(DATA))
    r = obj["result"]
    packed = np.array(r["shots_array_packed"], dtype=np.uint8)
    meas = bitarray_packed_to_bools(packed, r["n_shots"], r["n_bits"])
    print(f"IBM d={D} r={R}: {meas.shape[0]} shots, {meas.shape[1]} bits")

    clean = stim.Circuit.generated("surface_code:rotated_memory_z", distance=D, rounds=R)
    det, obs = clean.compile_m2d_converter().convert(measurements=meas, separate_observables=True)
    n = det.shape[0]
    print(f"det {det.shape} obs {obs.shape}  det_flip={det.mean():.4f} obs_flip={obs.mean():.4f}")

    # PM with DEM swept (same protocol as the 10K decode)
    pm_results = {}
    for p_dem in [0.005, 0.010, 0.015, 0.020, 0.025]:
        dem = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=D, rounds=R,
            after_clifford_depolarization=p_dem, before_measure_flip_probability=p_dem,
            after_reset_flip_probability=p_dem, before_round_data_depolarization=p_dem,
        ).detector_error_model(decompose_errors=True)
        pm = pymatching.Matching.from_detector_error_model(dem)
        pred = pm.decode_batch(det).astype(np.uint8)
        k = int(np.any(pred != obs, axis=1).sum())
        ler, lo, hi = wilson_ci(k, n)
        pm_results[f"p{p_dem}"] = {"ler": ler, "ci": [lo, hi], "errors": k}
        print(f"  PM p_dem={p_dem:.3f}: LER={ler*100:6.3f}% [{lo*100:.3f},{hi*100:.3f}] ({k})")
    best = min(pm_results, key=lambda kk: pm_results[kk]["ler"])
    pm_ler, (pm_lo, pm_hi), pm_k = pm_results[best]["ler"], pm_results[best]["ci"], pm_results[best]["errors"]
    print(f"  >> best PM at {best}: {pm_ler*100:.3f}%")

    print(f"\nCalibrated PFWL3S 3-seed (H=384) @100K...")
    logits = np.zeros((len(CALIB_SEEDS), n), dtype=np.float32)
    per_seed = []
    for i, ckpt in enumerate(CALIB_SEEDS):
        pred, lg = run_pf_seed_gpu(clean, det, obs, ckpt)
        logits[i] = lg
        k = int(np.any(pred != obs, axis=1).sum()); per_seed.append(k)
        print(f"  seed{i}: {k}/{n} = {k/n*100:.3f}%")
    ens = (logits.mean(0) > 0).astype(np.uint8).reshape(-1, 1)
    pf_k = int(np.any(ens != obs, axis=1).sum())
    pf_ler, pf_lo, pf_hi = wilson_ci(pf_k, n)

    if pf_hi < pm_lo: verdict = "PFWL3S strict-wins"
    elif pf_lo > pm_hi: verdict = "PM strict-wins"
    else: verdict = "tie (CIs overlap)"

    print(f"\n=== d5r5 @100K (CALIBRATED 3-seed) ===")
    print(f"PM:               {pm_ler*100:6.3f}% [{pm_lo*100:.3f},{pm_hi*100:.3f}] ({pm_k})")
    print(f"PFWL3S calib 3s:  {pf_ler*100:6.3f}% [{pf_lo*100:.3f},{pf_hi*100:.3f}] ({pf_k})  per-seed {per_seed}")
    print(f"Δ(PM−PF) = {(pm_ler-pf_ler)*100:+.3f}pp ({(pm_ler-pf_ler)/pm_ler*100:+.1f}% rel)")
    print(f">> VERDICT @100K: {verdict}")

    json.dump({
        "distance": D, "rounds": R, "n_shots": n,
        "detector_flip_rate": float(det.mean()), "observable_flip_rate": float(obs.mean()),
        "pymatching_sweep": pm_results,
        "pymatching_best": {"p_dem": best, "ler": pm_ler, "ci": [pm_lo, pm_hi], "errors": pm_k},
        "pfwl3s_calibrated_3seed": {"ler": pf_ler, "ci": [pf_lo, pf_hi], "errors": pf_k,
                                    "per_seed_errors": per_seed, "ckpts": CALIB_SEEDS,
                                    "ensemble_method": "logit_mean"},
        "verdict": verdict,
    }, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
