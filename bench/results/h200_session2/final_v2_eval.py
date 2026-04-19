"""
Final head-to-head: Pathfinder-v2 vs Pathfinder-v1 vs Lange vs PM
at d=3,5,7 × full noise range, under correct noise model, multi-seed.

Uses v2 checkpoint if available, falls back to v1.
"""
import sys, os, json, math, time
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace/GNN_decoder")
sys.path.insert(0, "/workspace")
import numpy as np, torch, stim, pymatching
from torch_geometric.nn import knn_graph
from src.gnn_models import GNN_7
from model import NeuralDecoder as NeuralDecoderV1
from train_v2 import NeuralDecoderV2, MixedNoiseDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class PFMapper:
    def __init__(self, circ):
        nd = circ.num_detectors
        coords = circ.get_detector_coordinates()
        ac = np.array([coords[i] for i in range(nd)])
        sp, tm = ac[:, :-1], ac[:, -1]
        tu = np.sort(np.unique(tm))
        xu = np.sort(np.unique(sp[:, 0]))
        yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
        self.grid = (len(tu), len(yu), len(xu))
        tm_m = {v: i for i, v in enumerate(tu)}; xm = {v: i for i, v in enumerate(xu)}; ym = {v: i for i, v in enumerate(yu)}
        di = np.zeros((nd, 3), dtype=np.int64)
        for did in range(nd):
            c = coords[did]; di[did] = [tm_m[c[-1]], ym.get(c[1], 0) if len(c) > 2 else 0, xm[c[0]]]
        self.di = di; self.nd = nd

    def to_tensor(self, det):
        B = det.shape[0]; T, H, W = self.grid
        t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        d = torch.from_numpy(det.astype(np.float32))
        for i in range(self.nd):
            t[:, 0, self.di[i, 0], self.di[i, 1], self.di[i, 2]] = d[:, i]
        return t


def load_v1(d):
    paths = {3: "/workspace/pathfinder/train/checkpoints/best_model.pt",
             5: "/workspace/pathfinder/train/checkpoints/d5_muon/best_model.pt",
             7: "/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt"}
    ck = torch.load(paths[d], weights_only=False, map_location=device)
    m = NeuralDecoderV1(ck["config"]).to(device); m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


def load_v2(d):
    p = f"/workspace/pathfinder/train/checkpoints/v2_d{d}/best_model.pt"
    if not os.path.exists(p):
        return None
    ck = torch.load(p, weights_only=False, map_location=device)
    m = NeuralDecoderV2(distance=ck["distance"], rounds=ck["rounds"],
                       hidden_dim=ck["hidden_dim"], n_blocks=ck["n_blocks"]).to(device)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


class LangeWrapper:
    def __init__(self, d, d_t):
        self.code_size = d; self.d_t = d_t
        self.m_nearest_nodes = 10; self.power = 2
        self.model = GNN_7(
            hidden_channels_GCN=[32, 128, 256, 512, 512, 256, 256],
            hidden_channels_MLP=[256, 128, 64], num_classes=1).to(device)
        ck = torch.load(f"/workspace/GNN_decoder/models/circuit_level_noise/d{d}/d{d}_d_t_{d_t}.pt",
                        weights_only=False, map_location=device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    def init_from_circuit(self, circ):
        coords = circ.get_detector_coordinates()
        dc = np.array(list(coords.values()))
        dc[:, :2] = dc[:, :2] / 2
        self.det_coords = dc.astype(np.uint8)
        sz = self.code_size + 1
        sx = np.zeros((sz, sz), dtype=np.uint8)
        sx[::2, 1:sz-1:2] = 1; sx[1::2, 2::2] = 1
        smz = np.rot90(sx) * 3
        self.syn_mask = np.dstack([sx + smz] * (self.d_t + 1))

    def stim_to_syn3d(self, det):
        mask = np.repeat(self.syn_mask[None, ...], det.shape[0], 0)
        s = np.zeros_like(mask)
        s[:, self.det_coords[:, 1], self.det_coords[:, 0], self.det_coords[:, 2]] = det
        s[np.nonzero(s)] = mask[np.nonzero(s)]
        return s

    def predict(self, det):
        if det.shape[0] == 0: return np.zeros((0, 1), dtype=np.uint8)
        s3d = self.stim_to_syn3d(det.astype(np.uint8)).astype(np.float32)
        inds = np.nonzero(s3d); defs = s3d[inds]; inds = np.transpose(np.array(inds))
        xd = defs == 1; zd = defs == 3
        nf = np.zeros((defs.shape[0], 6), dtype=np.float32)
        nf[xd, 0] = 1; nf[xd, 2:] = inds[xd, ...]; nf[zd, 1] = 1; nf[zd, 2:] = inds[zd, ...]
        x_cols = [0, 1, 3, 4, 5]
        x = torch.tensor(nf[:, x_cols]).to(device); batch = torch.tensor(nf[:, 2]).long().to(device)
        pos = x[:, 2:]
        ei = knn_graph(pos, self.m_nearest_nodes, batch=batch)
        dist = torch.sqrt(((pos[ei[0],:] - pos[ei[1],:])**2).sum(dim=1, keepdim=True))
        ea = 1.0 / (dist ** self.power + 1e-8)
        with torch.no_grad():
            out = self.model(x, ei, batch, ea)
        return (torch.sigmoid(out) > 0.5).cpu().numpy().astype(np.uint8)


def eval_all(d, p, seed, n_shots, v1, v2, lange, pfm):
    c = make_circuit(d, p)
    s = c.compile_detector_sampler(seed=seed)
    det, obs = s.sample(shots=n_shots, separate_observables=True)
    det = det.astype(np.uint8); obs = obs.astype(np.uint8)

    # PM
    dem = c.detector_error_model(decompose_errors=True)
    pm = pymatching.Matching.from_detector_error_model(dem)
    pm_preds = pm.decode_batch(det)
    pm_errs = int(np.sum(np.any(pm_preds != obs, axis=1)))

    # v1
    v1_errs = 0
    for i in range(0, n_shots, 1000):
        bd = det[i:i+1000]; bo = obs[i:i+1000]
        syn = pfm.to_tensor(bd).to(device)
        with torch.no_grad(): lg = v1(syn)
        preds = (lg > 0).cpu().numpy().astype(np.uint8)
        v1_errs += int(np.sum(np.any(preds != bo, axis=1)))

    # v2 (if exists)
    if v2 is not None:
        v2_errs = 0
        log_p_t = torch.full((1000,), math.log(p), device=device)
        for i in range(0, n_shots, 1000):
            bd = det[i:i+1000]; bo = obs[i:i+1000]
            syn = pfm.to_tensor(bd).to(device)
            lp = log_p_t[:syn.shape[0]]
            with torch.no_grad(): lg = v2(syn, lp)
            preds = (lg > 0).cpu().numpy().astype(np.uint8)
            v2_errs += int(np.sum(np.any(preds != bo, axis=1)))
    else:
        v2_errs = None

    # Lange
    lange.init_from_circuit(c)
    shots_nt = np.sum(det, axis=1) != 0
    det_nt = det[shots_nt]; obs_nt = obs[shots_nt]
    l_errs = 0
    bs = 500
    for i in range(0, len(det_nt), bs):
        bd = det_nt[i:i+bs]; bo = obs_nt[i:i+bs]
        preds = lange.predict(bd)
        l_errs += int(np.sum(np.any(preds != bo, axis=1)))

    return {"n": n_shots, "pm": pm_errs, "v1": v1_errs, "v2": v2_errs, "lange": l_errs}


def main():
    configs = [(d, p) for d in [3, 5, 7] for p in [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015]]
    N_SEEDS = 3
    N_PER_SEED = 10000
    results = {}
    for d, p in configs:
        print(f"\n=== d={d} p={p} ===", flush=True)
        c = make_circuit(d, p)
        pfm = PFMapper(c)
        v1 = load_v1(d); v2 = load_v2(d)
        lg = LangeWrapper(d, d)
        tot = {"n": 0, "pm": 0, "v1": 0, "v2": 0, "lange": 0, "v2_avail": v2 is not None}
        for seed in range(4000, 4000 + N_SEEDS):
            r = eval_all(d, p, seed, N_PER_SEED, v1, v2, lg, pfm)
            print(f"  seed={seed}: pm={r['pm']}  v1={r['v1']}  v2={r['v2']}  lange={r['lange']}", flush=True)
            for k in ["pm", "v1", "lange"]:
                tot[k] += r[k]
            tot["n"] += r["n"]
            if r["v2"] is not None: tot["v2"] += r["v2"]
        n = tot["n"]
        parts = []
        for k in ["v1", "v2", "lange", "pm"]:
            if k == "v2" and not tot["v2_avail"]:
                parts.append(f"{k}=n/a")
            else:
                kp, klo, khi = wilson(tot[k], n)
                parts.append(f"{k}={kp*100:.4f}%[{klo*100:.4f},{khi*100:.4f}]")
        print(f"  SUMMARY: {' | '.join(parts)}", flush=True)
        results[f"d{d}_p{p}"] = {"d": d, "p": p, "n": n,
                                "v1_ler": tot["v1"]/n, "lange_ler": tot["lange"]/n, "pm_ler": tot["pm"]/n,
                                "v2_ler": tot["v2"]/n if tot["v2_avail"] else None}

    with open("/workspace/final_v2_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
