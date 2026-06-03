"""IBM d=7 r=7 full eval: PM + PFWL3S 3-seed + Lange + Triad.

Uses the existing PFWL3S d=7 3-seed at H=384 (no calibrated d=7 trained yet)
and Lange's published d7_d_t_7 ckpt.
"""
import sys, json
import numpy as np
import stim
import pymatching
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'coda_experiments/GNN_decoder')
sys.path.insert(0, 'train')

from decode_ibm_result import bitarray_packed_to_bools, wilson_ci
from redecode_ibm_d3r3_pfwl3s import PathfinderMapper, run_pf_seed
from lange_decode_ibm import LangeMapper, build_lange_graph, load_lange_model
from eval_ibm_full import lange_predict_all

PFWL3S_D7 = [
    'bench/results/h200_main/tierC1/pathfinder_wide_long_d7/best_model.pt',
    'bench/results/h200_main/tierC1/pathfinder_wide_long_d7_seed1/best_model.pt',
    'bench/results/h200_main/tierC1/pathfinder_wide_long_d7_seed2/best_model.pt',
]
D, R = 7, 7

def main():
    print(f"=== IBM Heron r2 d={D} r={R} : PM + PFWL3S + Lange + Triad ===")
    obj = json.load(open(f'ibm_d{D}r{R}_result.json'))
    r = obj['result']
    packed = np.array(r['shots_array_packed'], dtype=np.uint8)
    measurements = bitarray_packed_to_bools(packed, r['n_shots'], r['n_bits'])
    print(f"shots={measurements.shape[0]}, bits={measurements.shape[1]}")

    clean = stim.Circuit.generated('surface_code:rotated_memory_z', distance=D, rounds=R)
    m2d = clean.compile_m2d_converter()
    det, obs = m2d.convert(measurements=measurements, separate_observables=True)
    n_shots = det.shape[0]
    print(f"detectors={det.shape}, det_flip={det.mean():.4f}, obs_flip={obs.mean():.4f}")

    # PM
    best_k, best_p, best_pred = n_shots, None, None
    for p_dem in [0.005, 0.010, 0.015, 0.020]:
        dc = stim.Circuit.generated('surface_code:rotated_memory_z', distance=D, rounds=R,
            after_clifford_depolarization=p_dem, before_measure_flip_probability=p_dem,
            after_reset_flip_probability=p_dem, before_round_data_depolarization=p_dem)
        pm = pymatching.Matching.from_detector_error_model(dc.detector_error_model(decompose_errors=True))
        pp = pm.decode_batch(det).astype(np.uint8)
        k = int(np.any(pp != obs, axis=1).sum())
        if k < best_k: best_k, best_p, best_pred = k, p_dem, pp
    pm_pred = best_pred
    pm_k = best_k
    pm_ler, pm_lo, pm_hi = wilson_ci(pm_k, n_shots)
    print(f"PM (best p_dem={best_p}): LER={pm_ler*100:6.3f}% CI=[{pm_lo*100:.3f}, {pm_hi*100:.3f}]")

    # PFWL3S 3-seed
    all_logits = np.zeros((len(PFWL3S_D7), n_shots), dtype=np.float32)
    per_seed_k = []
    for i, ck in enumerate(PFWL3S_D7):
        sp, lg = run_pf_seed(clean, det, obs, ck)
        all_logits[i] = lg
        per_seed_k.append(int(np.any(sp != obs, axis=1).sum()))
    pf_pred = ((all_logits.mean(axis=0)) > 0).astype(np.uint8).reshape(-1, 1)
    pf_k = int(np.any(pf_pred != obs, axis=1).sum())
    pf_ler, pf_lo, pf_hi = wilson_ci(pf_k, n_shots)
    print(f"PFWL3S (3-seed wide-long-d7, H=384): LER={pf_ler*100:6.3f}% CI=[{pf_lo*100:.3f}, {pf_hi*100:.3f}]")
    print(f"  per-seed errors: {per_seed_k}")

    # Lange published d_t=7
    mapper = LangeMapper(clean, D)
    la_model, la_ckpt = load_lange_model(D, R)
    la_pred = lange_predict_all(la_model, det.astype(np.uint8), mapper)
    la_k = int(np.any(la_pred != obs, axis=1).sum())
    la_ler, la_lo, la_hi = wilson_ci(la_k, n_shots)
    print(f"Lange (published d_t=7): LER={la_ler*100:6.3f}% CI=[{la_lo*100:.3f}, {la_hi*100:.3f}]")

    # Triad
    triad = ((pf_pred.astype(int) + la_pred.astype(int) + pm_pred.astype(int)) >= 2).astype(np.uint8)
    tr_k = int(np.any(triad != obs, axis=1).sum())
    tr_ler, tr_lo, tr_hi = wilson_ci(tr_k, n_shots)
    print(f"Pathfinder-Triad: LER={tr_ler*100:6.3f}% CI=[{tr_lo*100:.3f}, {tr_hi*100:.3f}]")

    def vd(a_ler, a_ci, b_ler, b_ci, A, B):
        if a_ci[1] < b_ci[0]: return f"{A} strict-wins"
        if a_ci[0] > b_ci[1]: return f"{B} strict-wins"
        return "tie"

    print(f"\nVerdicts:")
    print(f"  PFWL3S vs PM:    {vd(pf_ler, [pf_lo, pf_hi], pm_ler, [pm_lo, pm_hi], 'PFWL3S', 'PM')}")
    print(f"  PFWL3S vs Lange: {vd(pf_ler, [pf_lo, pf_hi], la_ler, [la_lo, la_hi], 'PFWL3S', 'Lange')}")
    print(f"  Triad vs PM:     {vd(tr_ler, [tr_lo, tr_hi], pm_ler, [pm_lo, pm_hi], 'Triad', 'PM')}")
    print(f"  Triad vs Lange:  {vd(tr_ler, [tr_lo, tr_hi], la_ler, [la_lo, la_hi], 'Triad', 'Lange')}")

    out = {
        'distance': D, 'rounds': R, 'n_shots': n_shots,
        'det_flip_rate': float(det.mean()),
        'obs_flip_rate': float(obs.mean()),
        'pm': {'ler': pm_ler, 'ci': [pm_lo, pm_hi], 'errors': pm_k, 'p_dem_best': best_p},
        'pfwl3s': {'ler': pf_ler, 'ci': [pf_lo, pf_hi], 'errors': pf_k,
                   'per_seed_errors': per_seed_k, 'ckpts': PFWL3S_D7,
                   'label': 'wide-long-3seed-H384 (not calibrated)'},
        'lange_published': {'ler': la_ler, 'ci': [la_lo, la_hi], 'errors': la_k,
                            'ckpt': la_ckpt, 'd_t': R},
        'pathfinder_triad': {'ler': tr_ler, 'ci': [tr_lo, tr_hi], 'errors': tr_k},
    }
    with open(f'ibm_d{D}r{R}_full_eval.json', 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
