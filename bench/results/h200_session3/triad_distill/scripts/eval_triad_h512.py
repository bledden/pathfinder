"""H=512 Triad-distill 3-seed-avg eval at d=7, 100K shots."""
import sys, os, json
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace/GNN_decoder")
import numpy as np, torch, pymatching
from ensemble_pf_lange import LangeWrapper, PathfinderMapper, wilson, make_circuit
from model import NeuralDecoder
device = torch.device("cuda")

PF_CKPTS = [f"/workspace/persist/checkpoints/pathfinder_triad_h512_d7_seed{s}/best_model.pt" for s in (0, 1, 2)]

def load(paths):
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

paths_present = [p for p in PF_CKPTS if os.path.exists(p)]
print(f"d=7 H=512: {len(paths_present)} Triad-distill PF model(s)", flush=True)
for p in paths_present: print(f"  {p}", flush=True)
pf_models = load(paths_present)

NOISE_RATES = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015]
N_PER_SEED = 20000
SAMPLE_SEEDS = [3000, 3001, 3002, 3003, 3004]
d = 7
results = {}

for p in NOISE_RATES:
    c = make_circuit(d, p); pfm = PathfinderMapper(c)
    pf_e = la_e = pm_e = maj_e = all3 = tot = 0
    for sseed in SAMPLE_SEEDS:
        sampler = c.compile_detector_sampler(seed=sseed)
        det, obs = sampler.sample(shots=N_PER_SEED, separate_observables=True)
        det = det.astype(np.uint8); obs = obs.astype(np.uint8)
        dem = c.detector_error_model(decompose_errors=True)
        pm = pymatching.Matching.from_detector_error_model(dem)
        pm_pr = pm.decode_batch(det).astype(np.uint8)
        pf_pr = np.zeros_like(obs)
        for i in range(0, N_PER_SEED, 250):
            syn = pfm.to_tensor(det[i:i+250]).to(device)
            pf_pr[i:i+250] = pf_predict_avg(pf_models, syn)
        lw = LangeWrapper(d, d); lw.init_from_circuit(c)
        la_pr = np.zeros_like(obs)
        for i in range(0, N_PER_SEED, 500):
            la_pr[i:i+500] = lw.predict_batch(det[i:i+500])
        pfw = np.any(pf_pr != obs, axis=1); law = np.any(la_pr != obs, axis=1); pmw = np.any(pm_pr != obs, axis=1)
        maj = ((pf_pr.astype(int) + la_pr.astype(int) + pm_pr.astype(int)) >= 2).astype(np.uint8)
        majw = np.any(maj != obs, axis=1)
        pf_e += int(pfw.sum()); la_e += int(law.sum()); pm_e += int(pmw.sum())
        maj_e += int(majw.sum()); all3 += int((pfw & law & pmw).sum()); tot += N_PER_SEED
    pfl, pflo, pfhi = wilson(pf_e, tot); lal, lalo, lahi = wilson(la_e, tot)
    pml, pmlo, pmhi = wilson(pm_e, tot); mal, malo, mahi = wilson(maj_e, tot)
    non_pf = "PF<<Lange" if pfhi < lalo else ("Lange<<PF" if lahi < pflo else "overlap")
    non_pf_maj = "PF<<Maj" if pfhi < malo else ("Maj<<PF" if mahi < pflo else "overlap")
    print(f"  d={d} p={p}: PF={pfl*100:.4f}% [{pflo*100:.4f},{pfhi*100:.4f}]  Lange={lal*100:.4f}%  PM={pml*100:.4f}%  Maj={mal*100:.4f}% [{malo*100:.4f},{mahi*100:.4f}]  PFvsLange={non_pf}  PFvsMaj={non_pf_maj}", flush=True)
    results[f"d{d}_p{p}"] = {"d": d, "p": p, "n": tot, "n_pf_seeds": len(pf_models),
        "pf_ler": pfl, "pf_ci": [pflo, pfhi], "lange_ler": lal, "lange_ci": [lalo, lahi],
        "pm_ler": pml, "pm_ci": [pmlo, pmhi], "majority_ler": mal, "majority_ci": [malo, mahi],
        "oracle_lb": all3 / tot}

os.makedirs("/workspace/persist/results", exist_ok=True)
with open("/workspace/persist/results/ensemble_triad_h512_d7.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved", flush=True)
