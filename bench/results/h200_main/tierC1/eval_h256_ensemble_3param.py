"""M7 (3-param noise, matches original Table 5): §5.6 ensemble redone with
canonical H=256 PF. Uses the Table-1 ckpt for in-distribution accuracy."""
import sys, os, json
sys.path.insert(0, '/workspace/pathfinder/train')
import numpy as np, torch, stim, pymatching
from model import NeuralDecoder

device = torch.device('cuda')

# 3-parameter noise circuit (matches Table 1)
def make_circuit_3param(d, p):
    return stim.Circuit.generated('surface_code:rotated_memory_z', distance=d, rounds=d,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p)

class PFMapper:
    def __init__(self, circuit):
        nd = circuit.num_detectors
        coords = circuit.get_detector_coordinates()
        ac = np.array([coords[i] for i in range(nd)])
        sp, tm = ac[:, :-1], ac[:, -1]
        tu = np.sort(np.unique(tm)); xu = np.sort(np.unique(sp[:, 0]))
        yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
        self.grid = (len(tu), len(yu), len(xu))
        tm_m = {v: i for i, v in enumerate(tu)}; xm = {v: i for i, v in enumerate(xu)}; ym = {v: i for i, v in enumerate(yu)}
        self.det_idx = np.zeros((nd, 3), dtype=np.int64)
        for did in range(nd):
            c = coords[did]
            self.det_idx[did] = [tm_m[c[-1]], ym.get(c[1], 0) if len(c) > 2 else 0, xm[c[0]]]
        self.nd = nd
    def to_tensor(self, det):
        B = det.shape[0]; T, H, W = self.grid
        t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        d = torch.from_numpy(det.astype(np.float32))
        for i in range(self.nd):
            t[:, 0, self.det_idx[i, 0], self.det_idx[i, 1], self.det_idx[i, 2]] = d[:, i]
        return t

# Canonical H=256 Table-1 ckpt (trained on 3-parameter noise)
CKPT = '/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt'
ck = torch.load(CKPT, weights_only=False, map_location=device)
print(f"Loading canonical H={ck['config'].hidden_dim} Table-1 ckpt: {CKPT}", flush=True)
m = NeuralDecoder(ck['config']).to(device); m.load_state_dict(ck['model_state_dict']); m.eval()

NOISE_RATES = [0.003, 0.005, 0.007, 0.010]
SHOTS_PER_SEED = 10000
SEEDS = [3000, 3001]
results = {'arch': f'H={ck["config"].hidden_dim} (canonical Table-1)', 'noise_model': '3-parameter', 'ckpt': CKPT, 'n_per_p': 20000, 'rates': {}}

for p in NOISE_RATES:
    c = make_circuit_3param(7, p); pfm = PFMapper(c)
    pf_logits_all = []; pf_pred_all = []; pm_pred_all = []; obs_all = []; tot = 0
    for s in SEEDS:
        sampler = c.compile_detector_sampler(seed=s)
        det, obs = sampler.sample(shots=SHOTS_PER_SEED, separate_observables=True)
        det = det.astype(np.uint8); obs = obs.astype(np.uint8)
        dem = c.detector_error_model(decompose_errors=True)
        pm = pymatching.Matching.from_detector_error_model(dem)
        pm_pr = pm.decode_batch(det).astype(np.uint8)
        pf_pr = np.zeros_like(obs); pf_logits = np.zeros((SHOTS_PER_SEED, obs.shape[1]))
        for i in range(0, SHOTS_PER_SEED, 500):
            syn = pfm.to_tensor(det[i:i+500]).to(device)
            with torch.no_grad():
                lg = m(syn).cpu().numpy()
            pf_logits[i:i+500] = lg
            pf_pr[i:i+500] = (lg > 0).astype(np.uint8)
        pf_logits_all.append(pf_logits); pf_pred_all.append(pf_pr); pm_pred_all.append(pm_pr); obs_all.append(obs)
        tot += SHOTS_PER_SEED
    pf_logits = np.concatenate(pf_logits_all); pf_pred = np.concatenate(pf_pred_all); pm_pred = np.concatenate(pm_pred_all); obs = np.concatenate(obs_all)
    pf_wrong = np.any(pf_pred != obs, axis=1); pm_wrong = np.any(pm_pred != obs, axis=1)
    or_wrong = pf_wrong & pm_wrong
    abs_logit = np.abs(pf_logits).max(axis=1)
    use_pf = abs_logit > 2.0
    ens_pred = np.where(use_pf[:, None], pf_pred, pm_pred)
    ens_wrong = np.any(ens_pred != obs, axis=1)
    both_correct = int(((~pf_wrong) & (~pm_wrong)).sum())
    both_wrong = int((pf_wrong & pm_wrong).sum())
    pf_wrong_pm_right = int((pf_wrong & ~pm_wrong).sum())
    pf_right_pm_wrong = int((~pf_wrong & pm_wrong).sum())
    pf_ler = pf_wrong.sum() / tot; pm_ler = pm_wrong.sum() / tot
    ens_ler = ens_wrong.sum() / tot; or_ler = or_wrong.sum() / tot
    print(f'  d=7 p={p}: PF={pf_ler*100:.4f}%  PM={pm_ler*100:.4f}%  ENS={ens_ler*100:.4f}%  OR={or_ler*100:.4f}%  ||  PFwrong&PMright={pf_wrong_pm_right}  PFright&PMwrong={pf_right_pm_wrong}  both_wrong={both_wrong}  pf_use_frac={use_pf.mean():.3f}', flush=True)
    results['rates'][f'p{p}'] = {'p': p, 'pf_ler': pf_ler, 'pm_ler': pm_ler, 'ens_ler': ens_ler, 'or_ler': or_ler,
        'pf_use_fraction': float(use_pf.mean()),
        'both_correct': both_correct, 'pf_wrong_pm_right': pf_wrong_pm_right,
        'pf_right_pm_wrong': pf_right_pm_wrong, 'both_wrong': both_wrong, 'n': tot}

os.makedirs('/workspace/persist/results', exist_ok=True)
with open('/workspace/persist/results/ensemble_h256_d7_3param.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved', flush=True)
