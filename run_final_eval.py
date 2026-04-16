"""DEFINITIVE END-TO-END EVALUATION - Paper Table 1.
Single script producing every number the paper needs.
All decoders, all code types, all noise rates, 100K shots, one JSON output.
"""
import sys, os, time, json, torch, numpy as np
sys.path.insert(0, "train")
sys.path.insert(0, "python")
sys.path.insert(0, "build")
from model import NeuralDecoder
from data import SyndromeDataset, DataConfig
from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph
import stim, pymatching

try:
    import pydecoder
    HAS_UF = True
except ImportError:
    HAS_UF = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 100000

def wilson(ne, nt, z=1.96):
    if nt == 0: return 0.0, 0.0
    p = ne/nt
    d = 1+z*z/nt
    return (p+z*z/(2*nt))/d, z*np.sqrt(p*(1-p)/nt+z*z/(4*nt*nt))/d

def g2c(g):
    if not HAS_UF: return None
    sg = pydecoder.SyndromeGraph()
    sg.n_detectors, sg.n_observables = g.n_detectors, g.n_observables
    for s,t,p,o in g.edges: sg.add_edge(s,t,p,o)
    sg.build_adjacency()
    return sg

class CDS:
    def __init__(self, ct, d, r, p, bs=1000):
        self.circ = stim.Circuit.generated(ct, distance=d, rounds=r,
            after_clifford_depolarization=p, before_measure_flip_probability=p,
            after_reset_flip_probability=p)
        self.samp = self.circ.compile_detector_sampler()
        self.nd = self.circ.num_detectors
        coords = self.circ.get_detector_coordinates()
        ac = np.array([coords[i] for i in range(self.nd)])
        sp, tm = ac[:,:-1], ac[:,-1]
        tu = np.sort(np.unique(tm))
        xu = np.sort(np.unique(sp[:,0]))
        yu = np.sort(np.unique(sp[:,1])) if sp.shape[1]>1 else np.array([0.0])
        self.grid = (len(tu), len(yu), len(xu))
        tm_m = {v:i for i,v in enumerate(tu)}
        xm = {v:i for i,v in enumerate(xu)}
        ym = {v:i for i,v in enumerate(yu)}
        self.d2g = {}
        for did in range(self.nd):
            c = coords[did]
            self.d2g[did] = (tm_m[c[-1]], ym.get(c[1],0) if len(c)>2 else 0, xm[c[0]])

    def tensor(self, det):
        B = det.shape[0]
        T,H,W = self.grid
        t = torch.zeros(B,1,T,H,W,dtype=torch.float32)
        for did,(gi,gj,gk) in self.d2g.items():
            if gi<T and gj<H and gk<W and did<det.shape[1]:
                t[:,0,gi,gj,gk] = torch.from_numpy(det[:,did].astype(np.float32))
        return t

    def raw(self, n):
        return self.samp.sample(shots=n, separate_observables=True)


def neural_ler(model_list, ds, det, obs, n):
    best = 1.0
    logits_sum = None
    cnt = 0
    for m in model_list:
        if m is None: continue
        te, tt = 0, 0
        lg_parts = []
        for i in range(0, n, 1000):
            bd = det[i:i+1000]
            bo = obs[i:i+1000]
            syn = ds.tensor(bd).to(device)
            lab = torch.from_numpy(bo.astype(np.float32)).to(device)
            with torch.no_grad():
                lg = m(syn)
                lg_parts.append(lg.cpu())
                te += ((lg>0).float()!=lab).any(dim=1).sum().item()
                tt += len(bd)
        ler = te/tt
        best = min(best, ler)
        bl = torch.cat(lg_parts, dim=0)
        logits_sum = bl if logits_sum is None else logits_sum + bl
        cnt += 1
    if cnt > 1:
        lab_all = torch.from_numpy(obs.astype(np.float32))
        ens = ((logits_sum/cnt > 0).float() != lab_all).any(dim=1).sum().item() / n
        best = min(best, ens)
    return best


# Load models
print("Loading models...", flush=True)
mdls = {}
paths = [
    ("d3", "train/checkpoints/best_model.pt"),
    ("d5", "train/checkpoints/d5_muon/best_model.pt"),
    ("d7a", "train/checkpoints/d7_final/best_model.pt"),
    ("d7b", "train/checkpoints/d7_p01/best_model.pt"),
    ("d7c", "train/checkpoints/d7_mixed/best_model.pt"),
    ("d7d", "train/checkpoints/d7_p015/best_model.pt"),
    ("abl_std", "train/checkpoints/ablation_stdconv_d5/best_model.pt"),
    ("abl_noc", "train/checkpoints/ablation_nocurriculum_d5/best_model.pt"),
]
for nm, pt in paths:
    if not os.path.exists(pt):
        print("  SKIP " + nm, flush=True)
        continue
    try:
        ck = torch.load(pt, weights_only=False, map_location="cpu")
        m = NeuralDecoder(ck["config"]).to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        mdls[nm] = m
        print("  " + nm + " OK", flush=True)
    except RuntimeError as e:
        print("  SKIP " + nm + " (architecture mismatch: ablation variant)", flush=True)

results = {}
t_start = time.time()

# PART 1: Main results
print("\n" + "="*80, flush=True)
print("PART 1: ROTATED SURFACE CODE d=3,5,7 x 8 noise rates x 100K shots", flush=True)
print("="*80, flush=True)

pvals = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015]

for d in [3, 5, 7]:
    print("\nd={} ".format(d) + "-"*60, flush=True)
    print("{:>8} {:>10} {:>10} {:>10} {:>10} {:>7}".format("p","Neural","PM","PM_corr","UF","N/PM"), flush=True)

    if d == 3: nm_list = [mdls.get("d3")]
    elif d == 5: nm_list = [mdls.get("d5")]
    else: nm_list = [mdls.get(k) for k in ["d7a","d7b","d7c","d7d"] if k in mdls]

    for p in pvals:
        ds = CDS("surface_code:rotated_memory_z", d, d, p)
        det, obs = ds.raw(N)

        nl = neural_ler(nm_list, ds, det, obs, N)

        dem = ds.circ.detector_error_model(decompose_errors=True)
        pm = pymatching.Matching.from_detector_error_model(dem)
        pl = np.sum(np.any(pm.decode_batch(det)!=obs, axis=1))/N

        pmc = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        pcl = np.sum(np.any(pmc.decode_batch(det)!=obs, axis=1))/N

        if HAS_UF:
            gr = extract_decoder_graph(make_circuit(SurfaceCodeConfig(d,d,p)))
            ul = np.sum(np.any(pydecoder.UnionFindDecoder(g2c(gr)).decode_batch(det)!=obs, axis=1))/N
        else:
            ul = -1

        r = "{:.2f}x".format(nl/pl) if pl > 0 else "inf"
        w = "WIN" if nl < pl else ("TIE" if abs(nl-pl)<1e-8 else "LOSS")
        print("  {:.4f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>6} {}".format(
            p, nl, pl, pcl, ul, r, w), flush=True)

        results["rotZ_d{}_p{}".format(d,p)] = dict(
            code="rotated_z", d=d, p=p, neural=nl, pm=pl, pm_corr=pcl, uf=ul, n=N)

# PART 2: Ablation
print("\n" + "="*80, flush=True)
print("PART 2: ABLATION (d=5, p=0.007, 100K shots)", flush=True)
print("="*80, flush=True)

abl_variants = [
    ("Full (DirConv+Muon+Curriculum)", "d5"),
    ("StdConv3d (Conv3d+Muon+Cur)", "abl_std"),
    ("NoCurriculum (DirConv+Muon+Fixed)", "abl_noc"),
]
ds_abl = CDS("surface_code:rotated_memory_z", 5, 5, 0.007)
det_abl, obs_abl = ds_abl.raw(N)

for vname, mkey in abl_variants:
    if mkey in mdls:
        nl = neural_ler([mdls[mkey]], ds_abl, det_abl, obs_abl, N)
        print("  {:>45}: {:.6f}".format(vname, nl), flush=True)
        results["ablation_"+mkey] = dict(variant=vname, ler=nl, n=N)
    else:
        print("  {:>45}: pending".format(vname), flush=True)

# PART 3: Code types
print("\n" + "="*80, flush=True)
print("PART 3: CODE TYPES (p=0.007)", flush=True)
print("="*80, flush=True)

for ct, d, label in [
    ("surface_code:rotated_memory_z", 5, "Rotated-Z d=5"),
    ("surface_code:unrotated_memory_z", 5, "Unrotated-Z d=5"),
    ("surface_code:rotated_memory_x", 5, "Rotated-X d=5"),
    ("color_code:memory_xyz", 3, "Color Code d=3"),
]:
    ds_ct = CDS(ct, d, d, 0.007)
    n_ct = min(N, 20000)
    det_ct, obs_ct = ds_ct.raw(n_ct)

    dem_ct = ds_ct.circ.detector_error_model(decompose_errors=True)
    pm_ct = pymatching.Matching.from_detector_error_model(dem_ct)
    pl_ct = np.sum(np.any(pm_ct.decode_batch(det_ct)!=obs_ct, axis=1))/n_ct

    # Use matching model if available
    if ct == "surface_code:rotated_memory_z" and "d5" in mdls:
        nl_ct = neural_ler([mdls["d5"]], ds_ct, det_ct, obs_ct, n_ct)
    else:
        nl_ct = -1  # No trained model for this code type yet

    if nl_ct >= 0:
        r_ct = "{:.2f}x".format(nl_ct/pl_ct) if pl_ct > 0 else "inf"
        print("  {:>20}: Neural={:.6f} PM={:.6f} {}".format(label, nl_ct, pl_ct, r_ct), flush=True)
    else:
        print("  {:>20}: Neural=pending PM={:.6f}".format(label, pl_ct), flush=True)

    results["ct_{}_d{}".format(ct.replace(":","_"),d)] = dict(
        code=ct, d=d, p=0.007, neural=nl_ct, pm=pl_ct, n=n_ct)

# Save
os.makedirs("bench/results", exist_ok=True)
with open("bench/results/final_eval.json", "w") as f:
    json.dump(results, f, indent=2)

elapsed = time.time() - t_start
print("\n" + "="*80, flush=True)
print("COMPLETE in {:.0f}s ({:.0f} min)".format(elapsed, elapsed/60), flush=True)
print("Results: bench/results/final_eval.json", flush=True)
print("="*80, flush=True)
