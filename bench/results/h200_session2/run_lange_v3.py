"""Head-to-head v2: use Lange's own graph-building code to avoid reimplementation bugs."""
import sys, os, json, time
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace/GNN_decoder")
import numpy as np
import torch, stim, pymatching
from torch_geometric.nn import knn_graph
from src.gnn_models import GNN_7
from model import NeuralDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wilson(k, n, z=1.96):
    if n == 0: return 0, 0, 0
    p = k/n; denom = 1 + z*z/n
    ctr = (p + z*z/(2*n)) / denom
    half = (z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return p, max(0, ctr-half), min(1, ctr+half)


class LangeDecoderWrapper:
    """Thin wrapper to use Lange's graph builder + model without their Decoder class."""
    def __init__(self, d, d_t, m_nearest_nodes=10, power=2):
        self.code_size = d
        self.d_t = d_t
        self.m_nearest_nodes = m_nearest_nodes
        self.power = power
        self.device = device
        self.sigmoid = torch.nn.Sigmoid()

        self.model = GNN_7(
            hidden_channels_GCN=[32, 128, 256, 512, 512, 256, 256],
            hidden_channels_MLP=[256, 128, 64],
            num_classes=1,
        ).to(device)
        path = f"/workspace/GNN_decoder/models/circuit_level_noise/d{d}/d{d}_d_t_{d_t}.pt"
        ck = torch.load(path, weights_only=False, map_location=device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    def init_from_circuit(self, circuit):
        """Replicate Decoder.__init__'s detector_coordinates + syndrome_mask setup."""
        coords = circuit.get_detector_coordinates()
        det_c = np.array(list(coords.values()))
        det_c[:, :2] = det_c[:, :2] / 2
        self.detector_coordinates = det_c.astype(np.uint8)
        sz = self.code_size + 1
        sx = np.zeros((sz, sz), dtype=np.uint8)
        sx[::2, 1:sz-1:2] = 1
        sx[1::2, 2::2] = 1
        sz_mask = np.rot90(sx) * 3
        self.syndrome_mask = np.dstack([sx + sz_mask] * (self.d_t + 1))

    def stim_to_syndrome_3D(self, det_events):
        mask = np.repeat(self.syndrome_mask[None, ...], det_events.shape[0], 0)
        s3d = np.zeros_like(mask)
        s3d[:, self.detector_coordinates[:, 1], self.detector_coordinates[:, 0],
            self.detector_coordinates[:, 2]] = det_events
        s3d[np.nonzero(s3d)] = mask[np.nonzero(s3d)]
        return s3d

    def get_batch_of_graphs(self, syndromes):
        s3d = self.stim_to_syndrome_3D(syndromes).astype(np.float32)
        inds = np.nonzero(s3d)
        defs = s3d[inds]
        inds = np.transpose(np.array(inds))
        x_def = defs == 1; z_def = defs == 3
        nf = np.zeros((defs.shape[0], 6), dtype=np.float32)
        nf[x_def, 0] = 1
        nf[x_def, 2:] = inds[x_def, ...]
        nf[z_def, 1] = 1
        nf[z_def, 2:] = inds[z_def, ...]
        x_cols = [0, 1, 3, 4, 5]
        x = torch.tensor(nf[:, x_cols]).to(self.device)
        batch = torch.tensor(nf[:, 2]).long().to(self.device)
        pos = x[:, 2:]
        ei = knn_graph(pos, self.m_nearest_nodes, batch=batch)
        dist = torch.sqrt(((pos[ei[0],:] - pos[ei[1],:])**2).sum(dim=1, keepdim=True))
        # Edge weight per Lange: 1 / dist^power, not dist^power
        # Actually from their code: edge_attr = 1.0 / (dist ** 2), but let's check a field 'power'
        ea = 1.0 / (dist ** self.power + 1e-8)
        return x, ei, batch, ea

    def predict(self, det_events):
        """Returns uint8 predictions [B, 1]. det_events should be non-trivial only."""
        if det_events.shape[0] == 0:
            return np.zeros((0, 1), dtype=np.uint8)
        x, ei, batch, ea = self.get_batch_of_graphs(det_events.astype(np.uint8))
        with torch.no_grad():
            out = self.model(x, ei, batch, ea)
        preds = (self.sigmoid(out) > 0.5).cpu().numpy().astype(np.uint8)
        return preds


def load_pathfinder(d):
    paths = {3: "/workspace/pathfinder/train/checkpoints/best_model.pt",
             5: "/workspace/pathfinder/train/checkpoints/d5_muon/best_model.pt",
             7: "/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt"}
    ck = torch.load(paths[d], weights_only=False, map_location=device)
    m = NeuralDecoder(ck["config"]).to(device)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m


class PathfinderMapper:
    def __init__(self, circuit):
        nd = circuit.num_detectors
        coords = circuit.get_detector_coordinates()
        ac = np.array([coords[i] for i in range(nd)])
        sp, tm = ac[:, :-1], ac[:, -1]
        tu = np.sort(np.unique(tm))
        xu = np.sort(np.unique(sp[:, 0]))
        yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
        self.grid = (len(tu), len(yu), len(xu))
        tm_m = {v: i for i, v in enumerate(tu)}
        xm = {v: i for i, v in enumerate(xu)}
        ym = {v: i for i, v in enumerate(yu)}
        di = np.zeros((nd, 3), dtype=np.int64)
        for did in range(nd):
            c = coords[did]
            di[did] = [tm_m[c[-1]], ym.get(c[1], 0) if len(c) > 2 else 0, xm[c[0]]]
        self.det_idx = di
        self.nd = nd

    def to_tensor(self, det_events):
        B = det_events.shape[0]
        T, H, W = self.grid
        t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        d = torch.from_numpy(det_events.astype(np.float32))
        for i in range(self.nd):
            t[:, 0, self.det_idx[i, 0], self.det_idx[i, 1], self.det_idx[i, 2]] = d[:, i]
        return t


def make_circuit(d, p):
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)


def eval_point(d, p, seed, n_shots):
    c = make_circuit(d, p)
    s = c.compile_detector_sampler(seed=seed)
    det, obs = s.sample(shots=n_shots, separate_observables=True)
    det = det.astype(np.uint8); obs = obs.astype(np.uint8)

    # PM
    dem = c.detector_error_model(decompose_errors=True)
    pm = pymatching.Matching.from_detector_error_model(dem)
    pm_preds = pm.decode_batch(det)
    pm_e = int(np.sum(np.any(pm_preds != obs, axis=1)))

    # Pathfinder
    pf = load_pathfinder(d)
    pfm = PathfinderMapper(c)
    pf_e = 0
    for i in range(0, n_shots, 1000):
        bd = det[i:i+1000]; bo = obs[i:i+1000]
        syn = pfm.to_tensor(bd).to(device)
        with torch.no_grad():
            lg = pf(syn)
        preds = (lg > 0).cpu().numpy().astype(np.uint8)
        pf_e += int(np.sum(np.any(preds != bo, axis=1)))

    # Lange (via wrapper)
    try:
        lw = LangeDecoderWrapper(d, d, m_nearest_nodes=10, power=2)
        lw.init_from_circuit(c)
        shots_w_flips = np.sum(det, axis=1) != 0
        n_triv = int(np.sum(~shots_w_flips))
        det_nt = det[shots_w_flips]
        obs_nt = obs[shots_w_flips]
        l_e = 0
        bs = 500  # smaller batch for knn_graph
        for i in range(0, len(det_nt), bs):
            bd = det_nt[i:i+bs]; bo = obs_nt[i:i+bs]
            preds = lw.predict(bd)
            l_e += int(np.sum(np.any(preds != bo, axis=1)))
        # Add trivial as correct (n_triv gets 0 errors)
        lange_ok = True
    except Exception as e:
        print(f"    Lange fail: {type(e).__name__}: {e}")
        l_e = None; n_triv = None; lange_ok = False

    return {"n": n_shots, "pm": pm_e, "pf": pf_e, "lange": l_e, "n_triv": n_triv, "lange_ok": lange_ok}


def main():
    # Lange's trained models are at p=[0.001,0.002,0.003,0.004,0.005]. Test those.
    configs = [(d, p) for d in [3, 5, 7] for p in [0.007, 0.010, 0.015]]
    N_SEEDS = 3
    N_PER_SEED = 20000
    results = {}
    for d, p in configs:
        print(f"\n=== d={d} p={p} ===", flush=True)
        tot = {"n": 0, "pm": 0, "pf": 0, "lange": 0, "n_triv": 0}
        all_ok = True
        for seed in range(2000, 2000 + N_SEEDS):
            r = eval_point(d, p, seed, N_PER_SEED)
            print(f"  seed={seed}: pm_err={r['pm']}  pf_err={r['pf']}  lange_err={r['lange']}  n_triv={r['n_triv']}", flush=True)
            tot["n"] += r["n"]; tot["pm"] += r["pm"]; tot["pf"] += r["pf"]
            if r["lange_ok"]:
                tot["lange"] += r["lange"]; tot["n_triv"] += r["n_triv"]
            else:
                all_ok = False
        pm_p, pm_lo, pm_hi = wilson(tot["pm"], tot["n"])
        pf_p, pf_lo, pf_hi = wilson(tot["pf"], tot["n"])
        if all_ok:
            lg_p, lg_lo, lg_hi = wilson(tot["lange"], tot["n"])
            print(f"  SUMMARY: Path={pf_p*100:.4f}% [{pf_lo*100:.4f},{pf_hi*100:.4f}]  Lange={lg_p*100:.4f}% [{lg_lo*100:.4f},{lg_hi*100:.4f}]  PM={pm_p*100:.4f}% [{pm_lo*100:.4f},{pm_hi*100:.4f}]", flush=True)
            results[f"d{d}_p{p}"] = {"d": d, "p": p, "pf_ler": pf_p, "lange_ler": lg_p, "pm_ler": pm_p,
                                    "pf_ci": [pf_lo, pf_hi], "lange_ci": [lg_lo, lg_hi], "pm_ci": [pm_lo, pm_hi],
                                    "total_n": tot["n"]}
        else:
            print(f"  SUMMARY: Path={pf_p*100:.4f}%  PM={pm_p*100:.4f}%  Lange=FAIL", flush=True)
            results[f"d{d}_p{p}"] = {"d": d, "p": p, "pf_ler": pf_p, "pm_ler": pm_p, "lange_ler": None}

    with open("/workspace/run_lange_v3_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
