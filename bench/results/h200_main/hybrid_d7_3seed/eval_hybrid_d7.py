'''Eval Hybrid CNN+GNN 3-seed-avg vs PFWL3S vs Lange vs PM at d=7.'''
import sys, os, json
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/pathfinder/train')
sys.path.insert(0, '/workspace/GNN_decoder')
import numpy as np, torch, pymatching
from ensemble_pf_lange import LangeWrapper, PathfinderMapper, wilson, make_circuit
from model import NeuralDecoder
from hybrid_model import HybridDecoder

device = torch.device('cuda')

HYBRID_CKPTS = [f'/workspace/persist/checkpoints/hybrid_d7_seed{s}/best_model.pt' for s in (0, 1, 2)]
PFW_CKPTS = [
    '/workspace/persist/checkpoints/pathfinder_wide_long_d7/best_model.pt',
    '/workspace/persist/checkpoints/pathfinder_wide_long_d7_seed1/best_model.pt',
    '/workspace/persist/checkpoints/pathfinder_wide_long_d7_seed2/best_model.pt',
]

def load_models(paths, cls):
    ms = []
    for p in paths:
        ck = torch.load(p, weights_only=False, map_location=device)
        m = cls(ck['config']).to(device); m.load_state_dict(ck['model_state_dict']); m.eval()
        ms.append(m)
    return ms

def avg_predict(models, syn):
    with torch.no_grad():
        avg = None
        for m in models:
            lg = m(syn).cpu().numpy()
            avg = lg if avg is None else avg + lg
        return ((avg / len(models)) > 0).astype(np.uint8)

hybrids = load_models([p for p in HYBRID_CKPTS if os.path.exists(p)], HybridDecoder)
pfwl3s = load_models([p for p in PFW_CKPTS if os.path.exists(p)], NeuralDecoder)
print(f'Loaded {len(hybrids)} Hybrid, {len(pfwl3s)} PFWL3S', flush=True)

NOISE_RATES = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015]
N_PER_SEED = 20000
SAMPLE_SEEDS = [3000, 3001, 3002, 3003, 3004]
d = 7
results = {'arch': 'Hybrid CNN+GNN', 'n_per_p': 100000, 'ckpts': HYBRID_CKPTS, 'rates': {}}

for p in NOISE_RATES:
    c = make_circuit(d, p); pfm = PathfinderMapper(c)
    hy_e = pf_e = la_e = pm_e = maj_hy_e = maj_pf_e = tot = 0
    for sseed in SAMPLE_SEEDS:
        sampler = c.compile_detector_sampler(seed=sseed)
        det, obs = sampler.sample(shots=N_PER_SEED, separate_observables=True)
        det = det.astype(np.uint8); obs = obs.astype(np.uint8)
        dem = c.detector_error_model(decompose_errors=True)
        pm = pymatching.Matching.from_detector_error_model(dem)
        pm_pr = pm.decode_batch(det).astype(np.uint8)
        hy_pr = np.zeros_like(obs); pf_pr = np.zeros_like(obs)
        for i in range(0, N_PER_SEED, 500):
            syn = pfm.to_tensor(det[i:i+500]).to(device)
            hy_pr[i:i+500] = avg_predict(hybrids, syn)
            pf_pr[i:i+500] = avg_predict(pfwl3s, syn)
        lw = LangeWrapper(d, d); lw.init_from_circuit(c)
        la_pr = np.zeros_like(obs)
        for i in range(0, N_PER_SEED, 500):
            la_pr[i:i+500] = lw.predict_batch(det[i:i+500])
        hyw = np.any(hy_pr != obs, axis=1); pfw = np.any(pf_pr != obs, axis=1)
        law = np.any(la_pr != obs, axis=1); pmw = np.any(pm_pr != obs, axis=1)
        maj_hy = ((hy_pr.astype(int) + la_pr.astype(int) + pm_pr.astype(int)) >= 2).astype(np.uint8)
        maj_pf = ((pf_pr.astype(int) + la_pr.astype(int) + pm_pr.astype(int)) >= 2).astype(np.uint8)
        maj_hy_w = np.any(maj_hy != obs, axis=1)
        maj_pf_w = np.any(maj_pf != obs, axis=1)
        hy_e += int(hyw.sum()); pf_e += int(pfw.sum()); la_e += int(law.sum()); pm_e += int(pmw.sum())
        maj_hy_e += int(maj_hy_w.sum()); maj_pf_e += int(maj_pf_w.sum())
        tot += N_PER_SEED
    hyl, hylo, hyhi = wilson(hy_e, tot); pfl, pflo, pfhi = wilson(pf_e, tot)
    lal, lalo, lahi = wilson(la_e, tot); pml, pmlo, pmhi = wilson(pm_e, tot)
    mhl, mhlo, mhhi = wilson(maj_hy_e, tot); mpl, mplo, mphi = wilson(maj_pf_e, tot)
    hy_vs_pf = 'Hyb<<PF' if hyhi < pflo else ('PF<<Hyb' if pfhi < hylo else 'overlap')
    hy_vs_la = 'Hyb<<Lange' if hyhi < lalo else ('Lange<<Hyb' if lahi < hylo else 'overlap')
    print(f'  d=7 p={p}: Hyb={hyl*100:.4f}% [{hylo*100:.4f},{hyhi*100:.4f}]  PF={pfl*100:.4f}%  Lange={lal*100:.4f}%  PM={pml*100:.4f}%  MajHyb={mhl*100:.4f}%  MajPF={mpl*100:.4f}%  HybvsPF={hy_vs_pf}  HybvsLange={hy_vs_la}', flush=True)
    results['rates'][f'p{p}'] = {
        'p': p, 'n': tot,
        'hybrid_ler': hyl, 'hybrid_ci': [hylo, hyhi],
        'pfwl3s_ler': pfl, 'pfwl3s_ci': [pflo, pfhi],
        'lange_ler': lal, 'lange_ci': [lalo, lahi],
        'pm_ler': pml, 'pm_ci': [pmlo, pmhi],
        'maj_with_hybrid_ler': mhl, 'maj_with_hybrid_ci': [mhlo, mhhi],
        'maj_with_pfwl3s_ler': mpl, 'maj_with_pfwl3s_ci': [mplo, mphi],
    }

os.makedirs('/workspace/persist/results', exist_ok=True)
with open('/workspace/persist/results/hybrid_eval_d7.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved')
