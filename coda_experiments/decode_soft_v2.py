"""Correct soft decode (parameterized by distance/rounds): feed FLOAT soft detectors to the
soft-trained model (no uint8 cast) via the SAME grid mapping the model trained with. The old
decode_soft_real.py routed soft through run_pf_seed which casts detectors->uint8, flooring all
soft values to 0. Compares PM(hard) vs PFWL3S(hard) vs PFWL3S(soft) with v1 + v2 calibration.

Usage: python decode_soft_v2.py [D] [R]   (default 3 3). Uses SOFT_d{D}r{R}_seed* checkpoints.
"""
import sys, json, os, glob, numpy as np, stim, pymatching, torch
sys.path.insert(0,'coda_experiments'); sys.path.insert(0,'train')
from decode_ibm_result import wilson_ci
from soft_info import calibrate_iq, soft_detectors_from_p1
from soft_info_v2 import calibrate_iq_v2
from soft_detmap import detector_to_measurements
from model import NeuralDecoder
from data_soft import SoftSyndromeDataset, SoftDataConfig

D = int(sys.argv[1]) if len(sys.argv) > 1 else 3
R = int(sys.argv[2]) if len(sys.argv) > 2 else 3
dr = f'd{D}r{R}'
SOFT = sorted(glob.glob(f'bench/results/h200_main/soft/SOFT_{dr}_seed*/best_model.pt'))

def load(ck):
    d = torch.load(ck, map_location='cpu', weights_only=False)
    m = NeuralDecoder(d['config']).eval(); m.load_state_dict(d['model_state_dict']); return m

def decode_tensor(models, tens, obs):
    n = tens.shape[0]; lg = np.zeros((len(models), n), np.float32)
    with torch.no_grad():
        for i, m in enumerate(models):
            out = []
            for s in range(0, n, 2000): out.append(m(tens[s:s+2000]).cpu().numpy().squeeze(-1))
            lg[i] = np.concatenate(out)
    pred = (lg.mean(0) > 0).astype(np.uint8).reshape(-1, 1)
    return wilson_ci(int(np.any(pred != obs, axis=1).sum()), n)

def best_pdem(detflip):
    cand = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    errs = [abs(stim.Circuit.generated('surface_code:rotated_memory_z', distance=D, rounds=R,
                after_clifford_depolarization=p, before_measure_flip_probability=p,
                after_reset_flip_probability=p, before_round_data_depolarization=p
            ).compile_detector_sampler(seed=2).sample(shots=3000, separate_observables=True)[0].mean() - detflip)
            for p in cand]
    return cand[int(np.argmin(errs))]

def main():
    assert SOFT, f"no SOFT checkpoints for {dr}"
    print(f"{dr}: {len(SOFT)} soft checkpoint(s)", flush=True)
    models = [load(c) for c in SOFT]
    rec = np.load(f'coda_experiments/ibm_{dr}_kerneled.npz')['rec']
    dtm, nm, nd, _ = detector_to_measurements(D, R)
    ds = SoftSyndromeDataset(SoftDataConfig(distance=D, rounds=R, n_alphas=3), rng=np.random.default_rng(0))
    clean = stim.Circuit.generated('surface_code:rotated_memory_z', distance=D, rounds=R)
    out = {}
    for caltag, calfn in [('v1', calibrate_iq), ('v2', calibrate_iq_v2)]:
        p1, _ = calfn(rec)
        hard_meas = (p1 > 0.5).astype(bool)
        det_hard, obs = clean.compile_m2d_converter().convert(measurements=hard_meas, separate_observables=True)
        det_hard = np.asarray(det_hard); obs = np.asarray(obs); n = det_hard.shape[0]
        soft_det = soft_detectors_from_p1(p1, dtm)
        detflip = float(det_hard.mean()); pdem = best_pdem(detflip)
        dem = stim.Circuit.generated('surface_code:rotated_memory_z', distance=D, rounds=R,
                  after_clifford_depolarization=pdem, before_measure_flip_probability=pdem,
                  after_reset_flip_probability=pdem, before_round_data_depolarization=pdem).detector_error_model(decompose_errors=True)
        pm = wilson_ci(int(np.any(pymatching.Matching.from_detector_error_model(dem).decode_batch(det_hard).astype(np.uint8) != obs, axis=1).sum()), n)
        t_hard = ds.soft_detectors_to_tensor(det_hard.astype(np.float64))
        t_soft = ds.soft_detectors_to_tensor(soft_det.astype(np.float64))
        pf_hard = decode_tensor(models, t_hard, obs)
        pf_soft = decode_tensor(models, t_soft, obs)
        out[caltag] = {'n': n, 'det_flip': detflip, 'p_dem': pdem,
            'gate_soft_eq_hard': bool((det_hard == (soft_det > 0.5)).all()),
            'uncertain_frac': float(((soft_det > 0.1) & (soft_det < 0.9)).mean()),
            'PM_hard': list(pm), 'PF_hard': list(pf_hard), 'PF_soft': list(pf_soft),
            'soft_vs_pf_hard': ('soft_strict' if pf_soft[2] < pf_hard[1] else ('hard_strict' if pf_hard[2] < pf_soft[1] else 'tie')),
            'soft_vs_pm': ('soft_strict' if pf_soft[2] < pm[1] else ('pm_strict' if pf_soft[1] > pm[2] else 'overlap'))}
        print(f"[{caltag}] {dr}: PM_hard {pm[0]*100:.2f}%  PF_hard {pf_hard[0]*100:.2f}%  PF_soft {pf_soft[0]*100:.2f}%  "
              f"(uncertain {out[caltag]['uncertain_frac']*100:.2f}%, gate {out[caltag]['gate_soft_eq_hard']})  "
              f"soft_vs_hard={out[caltag]['soft_vs_pf_hard']}  soft_vs_pm={out[caltag]['soft_vs_pm']}", flush=True)
    json.dump(out, open(f'coda_experiments/ibm_soft_v2_{dr}_eval.json', 'w'), indent=2, default=float)
    print(f"saved ibm_soft_v2_{dr}_eval.json", flush=True)

if __name__ == '__main__': main()
