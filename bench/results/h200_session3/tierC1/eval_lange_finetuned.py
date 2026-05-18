"""C2 audit: head-to-head with FINE-TUNED Lange (resumed from published d=7
weights, fine-tuned 30 epochs at p=0.007). Tests whether PFWL3S's strict-CI
win at d=7 p=0.007 survives when Lange is also fine-tuned at the operational
noise rate. 100K shots per p, 4-parameter noise."""
import sys, os, json, glob, copy
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/pathfinder/train')
sys.path.insert(0, '/workspace/GNN_decoder')
import numpy as np, torch, pymatching
from ensemble_pf_lange import LangeWrapper, PathfinderMapper, wilson, make_circuit
from model import NeuralDecoder

device = torch.device('cuda')

# === Build a fine-tuned LangeWrapper that loads the new ckpt ===
class LangeWrapperFT(LangeWrapper):
    def __init__(self, d, d_t, ft_ckpt_path, m_nearest_nodes=10, power=2):
        super().__init__(d, d_t, m_nearest_nodes, power)
        # overwrite weights with the fine-tuned version
        ck = torch.load(ft_ckpt_path, weights_only=False, map_location=device)
        self.model.load_state_dict(ck['model'])
        self.model.eval()

FT_CKPT = '/workspace/persist/lange_finetune_d7_p007/saved_models/d_7/d7_d_t_7_260518_040316_my_note_my_name_model.pt'
PUB_CKPT = '/workspace/GNN_decoder/models/circuit_level_noise/d7/d7_d_t_7.pt'

# Verify fine-tuned ckpt's final state
ft = torch.load(FT_CKPT, weights_only=False, map_location='cpu')
print(f"Fine-tuned ckpt epoch: {ft.get('training_history', {}).get('epoch')}")
print(f"Test_acc trajectory (last 10): {[round(float(a), 5) for a in ft.get('training_history', {}).get('test_accuracy', [])[-10:]]}")

# === Load PFWL3S 3-seed ===
PF_CKPTS = [
    '/workspace/persist/checkpoints/pathfinder_wide_long_d7/best_model.pt',
    '/workspace/persist/checkpoints/pathfinder_wide_long_d7_seed1/best_model.pt',
    '/workspace/persist/checkpoints/pathfinder_wide_long_d7_seed2/best_model.pt',
]
def load_pf(paths):
    models = []
    for p in paths:
        ck = torch.load(p, weights_only=False, map_location=device)
        m = NeuralDecoder(ck['config']).to(device); m.load_state_dict(ck['model_state_dict']); m.eval()
        models.append(m)
    return models
def pf_predict_avg(models, syn):
    with torch.no_grad():
        avg = None
        for m in models:
            lg = m(syn).cpu().numpy()
            avg = lg if avg is None else avg + lg
        return ((avg / len(models)) > 0).astype(np.uint8)

pf_models = load_pf(PF_CKPTS)
print(f'Loaded {len(pf_models)} PFWL3S ckpts')

# === Eval at d=7 across operational noise rates ===
NOISE_RATES = [0.005, 0.007, 0.010, 0.015]
N_PER_SEED = 20000
SAMPLE_SEEDS = [3000, 3001, 3002, 3003, 3004]  # 100K shots total
d = 7
results = {'note': 'C2 audit: head-to-head with fine-tuned Lange (30 epochs at p=0.007)',
           'pub_ckpt': PUB_CKPT, 'ft_ckpt': FT_CKPT,
           'pf_ckpts': PF_CKPTS, 'rates': {}}

for p in NOISE_RATES:
    c = make_circuit(d, p); pfm = PathfinderMapper(c)
    pf_e = la_pub_e = la_ft_e = pm_e = maj_pub_e = maj_ft_e = tot = 0
    for sseed in SAMPLE_SEEDS:
        sampler = c.compile_detector_sampler(seed=sseed)
        det, obs = sampler.sample(shots=N_PER_SEED, separate_observables=True)
        det = det.astype(np.uint8); obs = obs.astype(np.uint8)
        dem = c.detector_error_model(decompose_errors=True)
        pm = pymatching.Matching.from_detector_error_model(dem)
        pm_pr = pm.decode_batch(det).astype(np.uint8)
        pf_pr = np.zeros_like(obs)
        for i in range(0, N_PER_SEED, 500):
            syn = pfm.to_tensor(det[i:i+500]).to(device)
            pf_pr[i:i+500] = pf_predict_avg(pf_models, syn)
        # Run BOTH Lange variants
        lw_pub = LangeWrapper(d, d); lw_pub.init_from_circuit(c)
        lw_ft = LangeWrapperFT(d, d, FT_CKPT); lw_ft.init_from_circuit(c)
        la_pub_pr = np.zeros_like(obs)
        la_ft_pr = np.zeros_like(obs)
        for i in range(0, N_PER_SEED, 500):
            la_pub_pr[i:i+500] = lw_pub.predict_batch(det[i:i+500])
            la_ft_pr[i:i+500] = lw_ft.predict_batch(det[i:i+500])
        pfw = np.any(pf_pr != obs, axis=1)
        la_pub_w = np.any(la_pub_pr != obs, axis=1)
        la_ft_w = np.any(la_ft_pr != obs, axis=1)
        pmw = np.any(pm_pr != obs, axis=1)
        maj_pub = ((pf_pr.astype(int) + la_pub_pr.astype(int) + pm_pr.astype(int)) >= 2).astype(np.uint8)
        maj_ft = ((pf_pr.astype(int) + la_ft_pr.astype(int) + pm_pr.astype(int)) >= 2).astype(np.uint8)
        maj_pub_w = np.any(maj_pub != obs, axis=1)
        maj_ft_w = np.any(maj_ft != obs, axis=1)
        pf_e += int(pfw.sum()); la_pub_e += int(la_pub_w.sum()); la_ft_e += int(la_ft_w.sum())
        pm_e += int(pmw.sum()); maj_pub_e += int(maj_pub_w.sum()); maj_ft_e += int(maj_ft_w.sum())
        tot += N_PER_SEED
    pfl, pflo, pfhi = wilson(pf_e, tot)
    lpub_l, lpub_lo, lpub_hi = wilson(la_pub_e, tot)
    lft_l, lft_lo, lft_hi = wilson(la_ft_e, tot)
    pml, pmlo, pmhi = wilson(pm_e, tot)
    mpub_l, mpub_lo, mpub_hi = wilson(maj_pub_e, tot)
    mft_l, mft_lo, mft_hi = wilson(maj_ft_e, tot)
    pf_v_pub = 'PF<<LpubFT' if pfhi < lpub_lo else ('LpubFT<<PF' if lpub_hi < pflo else 'overlap')
    pf_v_ft = 'PF<<Lft' if pfhi < lft_lo else ('Lft<<PF' if lft_hi < pflo else 'overlap')
    print(f'  d={d} p={p}: PF={pfl*100:.4f}% [{pflo*100:.4f},{pfhi*100:.4f}]  Lpub={lpub_l*100:.4f}% [{lpub_lo*100:.4f},{lpub_hi*100:.4f}]  Lft={lft_l*100:.4f}% [{lft_lo*100:.4f},{lft_hi*100:.4f}]  PM={pml*100:.4f}%  MajPub={mpub_l*100:.4f}%  MajFT={mft_l*100:.4f}%  PFvsLpub={pf_v_pub}  PFvsLft={pf_v_ft}', flush=True)
    results['rates'][f'p{p}'] = {
        'p': p, 'n': tot,
        'pf_ler': pfl, 'pf_ci': [pflo, pfhi],
        'lange_pub_ler': lpub_l, 'lange_pub_ci': [lpub_lo, lpub_hi],
        'lange_ft_ler': lft_l, 'lange_ft_ci': [lft_lo, lft_hi],
        'pm_ler': pml, 'pm_ci': [pmlo, pmhi],
        'maj_pub_ler': mpub_l, 'maj_pub_ci': [mpub_lo, mpub_hi],
        'maj_ft_ler': mft_l, 'maj_ft_ci': [mft_lo, mft_hi],
    }

os.makedirs('/workspace/persist/results', exist_ok=True)
with open('/workspace/persist/results/lange_finetuned_eval_d7.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved')
