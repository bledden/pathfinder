"""
Final comparison: all Pathfinder variants vs Lange vs PM, under proper noise.
Tests (if available): d5_muon (original, weak-noise trained), fixed_d5 (proper noise retrained),
distill_d5 (distilled from Lange), and same for d=7.
"""
import sys, os, json
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace/GNN_decoder")
import numpy as np, torch, stim, pymatching
from torch_geometric.nn import knn_graph
from src.gnn_models import GNN_7
from model import NeuralDecoder, DecoderConfig

device = torch.device("cuda")

def wilson(k, n, z=1.96):
    if n == 0: return 0, 0, 0
    p = k/n; denom = 1 + z*z/n
    ctr = (p + z*z/(2*n)) / denom
    half = (z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return p, max(0, ctr-half), min(1, ctr+half)

def make_circuit(d, p):
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)

class PFMap:
    def __init__(self, c):
        nd = c.num_detectors
        coords = c.get_detector_coordinates()
        ac = np.array([coords[i] for i in range(nd)])
        sp, tm = ac[:, :-1], ac[:, -1]
        tu = np.sort(np.unique(tm))
        xu = np.sort(np.unique(sp[:, 0]))
        yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
        self.grid = (len(tu), len(yu), len(xu))
        tm_m = {v: i for i, v in enumerate(tu)}; xm = {v: i for i, v in enumerate(xu)}; ym = {v: i for i, v in enumerate(yu)}
        di = np.zeros((nd, 3), dtype=np.int64)
        for did in range(nd):
            cc = coords[did]
            di[did] = [tm_m[cc[-1]], ym.get(cc[1], 0) if len(cc) > 2 else 0, xm[cc[0]]]
        self.di = di; self.nd = nd
    def tensor(self, det):
        B = det.shape[0]; T, H, W = self.grid
        t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        d = torch.from_numpy(det.astype(np.float32))
        for i in range(self.nd):
            t[:, 0, self.di[i, 0], self.di[i, 1], self.di[i, 2]] = d[:, i]
        return t

def load_pf(path):
    if not os.path.exists(path): return None
    ck = torch.load(path, weights_only=False, map_location=device)
    m = NeuralDecoder(ck["config"]).to(device)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m

class LangeW:
    def __init__(self, d, d_t):
        self.d = d; self.d_t = d_t
        self.model = GNN_7(
            hidden_channels_GCN=[32, 128, 256, 512, 512, 256, 256],
            hidden_channels_MLP=[256, 128, 64], num_classes=1).to(device)
        ck = torch.load(f"/workspace/GNN_decoder/models/circuit_level_noise/d{d}/d{d}_d_t_{d_t}.pt",
                        weights_only=False, map_location=device)
        self.model.load_state_dict(ck["model"]); self.model.eval()
    def init_circ(self, c):
        coords = c.get_detector_coordinates()
        dc = np.array(list(coords.values())); dc[:, :2] = dc[:, :2] / 2
        self.dc = dc.astype(np.uint8)
        sz = self.d + 1
        sx = np.zeros((sz, sz), dtype=np.uint8); sx[::2, 1:sz-1:2] = 1; sx[1::2, 2::2] = 1
        smz = np.rot90(sx) * 3
        self.mask = np.dstack([sx + smz] * (self.d_t + 1))
    def predict(self, det):
        if det.shape[0] == 0: return np.zeros((0, 1), dtype=np.uint8)
        m = np.repeat(self.mask[None, ...], det.shape[0], 0)
        s = np.zeros_like(m)
        s[:, self.dc[:, 1], self.dc[:, 0], self.dc[:, 2]] = det.astype(np.uint8)
        s[np.nonzero(s)] = m[np.nonzero(s)]
        s = s.astype(np.float32)
        inds = np.nonzero(s); defs = s[inds]; inds = np.transpose(np.array(inds))
        xd = defs == 1; zd = defs == 3
        nf = np.zeros((defs.shape[0], 6), dtype=np.float32)
        nf[xd, 0] = 1; nf[xd, 2:] = inds[xd, ...]; nf[zd, 1] = 1; nf[zd, 2:] = inds[zd, ...]
        x = torch.tensor(nf[:, [0, 1, 3, 4, 5]]).to(device)
        batch = torch.tensor(nf[:, 2]).long().to(device)
        pos = x[:, 2:]
        ei = knn_graph(pos, 10, batch=batch)
        dist = torch.sqrt(((pos[ei[0], :] - pos[ei[1], :])**2).sum(dim=1, keepdim=True))
        ea = 1.0 / (dist ** 2 + 1e-8)
        with torch.no_grad():
            out = self.model(x, ei, batch, ea)
        return (torch.sigmoid(out) > 0.5).cpu().numpy().astype(np.uint8)


def eval_point(d, p, seed, n, models, mapper, lange):
    c = make_circuit(d, p)
    s = c.compile_detector_sampler(seed=seed)
    det, obs = s.sample(n, separate_observables=True)
    det = det.astype(np.uint8); obs = obs.astype(np.uint8)
    dem = c.detector_error_model(decompose_errors=True)
    pm = pymatching.Matching.from_detector_error_model(dem)
    pm_errs = int(np.sum(np.any(pm.decode_batch(det) != obs, axis=1)))
    result = {"pm": pm_errs, "n": n}
    for name, m in models.items():
        if m is None:
            result[name] = None
            continue
        errs = 0
        for i in range(0, n, 1000):
            bd = det[i:i+1000]; bo = obs[i:i+1000]
            syn = mapper.tensor(bd).to(device)
            with torch.no_grad():
                lg = m(syn)
            preds = (lg > 0).cpu().numpy().astype(np.uint8)
            errs += int(np.sum(np.any(preds != bo, axis=1)))
        result[name] = errs
    # Lange
    lange.init_circ(c)
    shots_nt = np.sum(det, axis=1) != 0
    det_nt = det[shots_nt]; obs_nt = obs[shots_nt]
    l_errs = 0
    for i in range(0, len(det_nt), 500):
        bd = det_nt[i:i+500]; bo = obs_nt[i:i+500]
        preds = lange.predict(bd)
        l_errs += int(np.sum(np.any(preds != bo, axis=1)))
    result["lange"] = l_errs
    return result


def main():
    configs = [(d, p) for d in [5, 7] for p in [0.001, 0.003, 0.005, 0.007, 0.010]]
    N_SEEDS = 3
    N_PER = 20000
    out = {}
    for d, p in configs:
        print(f"\n=== d={d} p={p} ===", flush=True)
        c = make_circuit(d, p)
        mapper = PFMap(c)
        # Load available models
        mods = {}
        mods["original"] = load_pf(f"/workspace/pathfinder/train/checkpoints/{'best_model.pt' if d == 3 else ('d5_muon/best_model.pt' if d == 5 else 'd7_final/best_model.pt')}")
        mods["fixed"] = load_pf(f"/workspace/pathfinder/train/checkpoints/fixed_d{d}/best_model.pt")
        mods["distill"] = load_pf(f"/workspace/pathfinder/train/checkpoints/distill_d{d}/best_model.pt")
        lg = LangeW(d, d)
        tot = {k: 0 for k in ["pm", "lange", "original", "fixed", "distill", "n"]}
        avail = {k: mods[k] is not None for k in ["original", "fixed", "distill"]}
        for seed in range(5000, 5000 + N_SEEDS):
            r = eval_point(d, p, seed, N_PER, mods, mapper, lg)
            print(f"  seed={seed}: pm={r['pm']} original={r['original']} fixed={r['fixed']} distill={r['distill']} lange={r['lange']}", flush=True)
            for k in ["pm", "lange", "n"]: tot[k] += r[k]
            for k in ["original", "fixed", "distill"]:
                if r[k] is not None: tot[k] += r[k]
        n = tot["n"]
        parts = []
        for k in ["original", "fixed", "distill", "lange", "pm"]:
            if k in ["original", "fixed", "distill"] and not avail[k]:
                parts.append(f"{k}=n/a")
            else:
                kp, klo, khi = wilson(tot[k], n)
                parts.append(f"{k}={kp*100:.4f}%[{klo*100:.4f},{khi*100:.4f}]")
        print(f"  SUMMARY: {' | '.join(parts)}", flush=True)
        out[f"d{d}_p{p}"] = {"d": d, "p": p, "n": n, **{f"{k}_ler": (tot[k]/n if (k not in ['original','fixed','distill'] or avail[k]) else None) for k in ["pm", "lange", "original", "fixed", "distill"]}}
    with open("/workspace/final_comparison_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
