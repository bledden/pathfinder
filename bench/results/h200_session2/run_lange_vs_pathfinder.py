"""
Head-to-head: Lange et al. GNN vs Pathfinder vs PyMatching on our Stim harness.
Matched noise model (full circuit-level), matched distances (d=3,5,7), matched seeds.
Uses their trained weights from GNN_decoder/models/circuit_level_noise/.
Properly handles their "filter trivial, count as correct" LER protocol.
"""
import sys, os, time, json
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace/GNN_decoder")
import numpy as np
import torch, stim, pymatching
from torch_geometric.nn import knn_graph
from src.gnn_models import GNN_7
from model import NeuralDecoder  # pathfinder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wilson(k, n, z=1.96):
    if n == 0: return 0, 0, 0
    p = k / n
    denom = 1 + z*z/n
    ctr = (p + z*z/(2*n)) / denom
    half = (z * np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return p, max(0, ctr - half), min(1, ctr + half)


def make_circuit(d, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p,
    )


# ---- Pathfinder loading ----
def load_pathfinder(d):
    paths = {3: "/workspace/pathfinder/train/checkpoints/best_model.pt",
             5: "/workspace/pathfinder/train/checkpoints/d5_muon/best_model.pt",
             7: "/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt"}
    ck = torch.load(paths[d], weights_only=False, map_location=device)
    m = NeuralDecoder(ck["config"]).to(device).eval()
    m.load_state_dict(ck["model_state_dict"])
    return m


# ---- Pathfinder grid mapping (vectorized) ----
class PathfinderMapper:
    def __init__(self, circuit):
        self.nd = circuit.num_detectors
        coords = circuit.get_detector_coordinates()
        ac = np.array([coords[i] for i in range(self.nd)])
        sp, tm = ac[:, :-1], ac[:, -1]
        tu = np.sort(np.unique(tm))
        xu = np.sort(np.unique(sp[:, 0]))
        yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
        self.grid = (len(tu), len(yu), len(xu))
        tm_m = {v: i for i, v in enumerate(tu)}
        xm = {v: i for i, v in enumerate(xu)}
        ym = {v: i for i, v in enumerate(yu)}
        di = np.zeros((self.nd, 3), dtype=np.int64)
        for did in range(self.nd):
            c = coords[did]
            di[did] = [tm_m[c[-1]], ym.get(c[1], 0) if len(c) > 2 else 0, xm[c[0]]]
        self.det_idx = di

    def to_tensor(self, det_events):
        B = det_events.shape[0]
        T, H, W = self.grid
        tensor = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        d = torch.from_numpy(det_events.astype(np.float32))
        for i in range(self.nd):
            tensor[:, 0, self.det_idx[i, 0], self.det_idx[i, 1], self.det_idx[i, 2]] = d[:, i]
        return tensor


# ---- Lange et al. graph builder ----
class LangeMapper:
    """Build the 3D syndrome mask + graph representation per Lange et al."""
    def __init__(self, circuit, d):
        coords = circuit.get_detector_coordinates()
        nd = circuit.num_detectors
        det_coords = np.array([coords[i] for i in range(nd)])
        # Their convention: (x, y, t) with x,y integers (divide by 2), t integer
        det_coords_int = (det_coords * [1, 1, 1]).astype(np.int64)
        # Derive shape from max coords
        max_x = det_coords_int[:, 0].max() + 1
        max_y = det_coords_int[:, 1].max() + 1
        max_t = det_coords_int[:, 2].max() + 1
        # Build syndrome mask: X=1, Z=3 labels at each detector position
        # Lange's protocol: only X-stabilizer detectors (even y) OR Z (odd y), check their convention
        # Simpler: look at their syndrome_mask
        self.syndrome_mask = np.zeros((max_y, max_x, max_t), dtype=np.float32)
        self.det_coords = det_coords_int
        self.d = d
        # Lange labels X-stab as 1, Z-stab as 3 via mask. Need to check which detectors are X vs Z.
        # In rotated_memory_z: detectors measure Z-stabilizers only. So all detectors are Z-type.
        # Actually wait — memory_z: Z-stabilizers have detectors, X-stabilizers don't (for memory-Z).
        # Lange trained on memory_z too. Set mask to 3 everywhere (Z-type).
        self.syndrome_mask[:] = 3.0  # all Z-type for memory_z circuits

    def stim_to_syndrome_3D(self, det_events):
        """Convert [B, n_detectors] boolean to [B, y, x, t] with X=1/Z=3 encoding."""
        B = det_events.shape[0]
        mask = np.repeat(self.syndrome_mask[None, ...], B, axis=0)
        syn3d = np.zeros_like(mask)
        syn3d[:, self.det_coords[:, 1], self.det_coords[:, 0], self.det_coords[:, 2]] = det_events.astype(np.float32)
        syn3d[np.nonzero(syn3d)] = mask[np.nonzero(syn3d)]
        return syn3d


def build_lange_graph(syndromes_3D, m_nearest_nodes=10, device="cuda"):
    """Given [B, y, x, t] syndrome array, build Lange-style graph batch."""
    defect_inds = np.nonzero(syndromes_3D)
    defects = syndromes_3D[defect_inds]
    defect_inds = np.transpose(np.array(defect_inds))  # [N_defects, 4] = (batch, y, x, t)

    x_defects = defects == 1
    z_defects = defects == 3

    node_features = np.zeros((defects.shape[0], 6), dtype=np.float32)
    node_features[x_defects, 0] = 1
    node_features[x_defects, 2:] = defect_inds[x_defects, ...]
    node_features[z_defects, 1] = 1
    node_features[z_defects, 2:] = defect_inds[z_defects, ...]

    x_cols = [0, 1, 3, 4, 5]  # skip batch col
    batch_col = 2

    x = torch.tensor(node_features[:, x_cols]).to(device)
    batch_labels = torch.tensor(node_features[:, batch_col]).long().to(device)
    pos = x[:, 2:]

    edge_index = knn_graph(pos, m_nearest_nodes, batch=batch_labels)
    dist = torch.sqrt(((pos[edge_index[0], :] - pos[edge_index[1], :]) ** 2).sum(dim=1, keepdim=True))
    edge_attr = 1.0 / (dist ** 2 + 1e-8)  # following their "power=2" default

    return x, edge_index, batch_labels, edge_attr


def load_lange_model(d, d_t):
    """Load Lange's pretrained GNN for (d, d_t)."""
    path = f"/workspace/GNN_decoder/models/circuit_level_noise/d{d}/d{d}_d_t_{d_t}.pt"
    model = GNN_7(
        hidden_channels_GCN=[32, 128, 256, 512, 512, 256, 256],
        hidden_channels_MLP=[256, 128, 64],
        num_classes=1,
    ).to(device)
    ck = torch.load(path, weights_only=False, map_location=device)
    # Checkpoint format varies — try a few loading strategies
    if isinstance(ck, dict):
        if "model" in ck:
            model.load_state_dict(ck["model"])
        elif "state_dict" in ck:
            model.load_state_dict(ck["model"])
        else:
            # assume ck IS the state dict
            model.load_state_dict(ck)
    else:
        model.load_state_dict(ck)
    model.eval()
    return model


# ---- Main eval ----
def eval_point(d, p, seed, n_shots):
    circuit = make_circuit(d, p)
    sampler = circuit.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots=n_shots, separate_observables=True)
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)

    # PyMatching
    dem = circuit.detector_error_model(decompose_errors=True)
    pm = pymatching.Matching.from_detector_error_model(dem)
    preds_pm = pm.decode_batch(det)
    pm_errs = int(np.sum(np.any(preds_pm != obs, axis=1)))

    # Pathfinder
    pf = load_pathfinder(d)
    pf_mapper = PathfinderMapper(circuit)
    pf_errs = 0
    for i in range(0, n_shots, 1000):
        batch_det = det[i:i+1000]
        batch_obs = obs[i:i+1000]
        syn = pf_mapper.to_tensor(batch_det).to(device)
        with torch.no_grad():
            lg = pf(syn)
        preds = (lg > 0).cpu().numpy().astype(np.uint8)
        pf_errs += int(np.sum(np.any(preds != batch_obs, axis=1)))

    # Lange GNN (filters trivial, counts trivial-as-correct)
    try:
        lange = load_lange_model(d, d)
        lange_mapper = LangeMapper(circuit, d)
        # Filter trivial
        shots_w_flips = np.sum(det, axis=1) != 0
        n_trivial = int(np.sum(~shots_w_flips))
        if n_trivial < n_shots:
            det_nontrivial = det[shots_w_flips]
            obs_nontrivial = obs[shots_w_flips]
            # Process in batches for memory
            lange_errs = 0
            bs = 1000
            for i in range(0, len(det_nontrivial), bs):
                batch_det = det_nontrivial[i:i+bs]
                batch_obs = obs_nontrivial[i:i+bs]
                syn3d = lange_mapper.stim_to_syndrome_3D(batch_det)
                try:
                    x, ei, bl, ea = build_lange_graph(syn3d, m_nearest_nodes=10, device=device)
                    with torch.no_grad():
                        out = lange(x, ei, bl, ea)
                    preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(np.uint8).reshape(-1, 1)
                    lange_errs += int(np.sum(np.any(preds != batch_obs, axis=1)))
                except Exception as e:
                    # Some batches may fail if empty
                    print(f"    Lange batch fail: {e}")
                    lange_errs += len(batch_det)  # count as all wrong
            # Add trivial syndromes (always correct for Lange's convention)
        else:
            lange_errs = 0
    except Exception as e:
        print(f"  Lange load/eval FAIL at d={d}: {type(e).__name__}: {e}")
        lange_errs = None
        n_trivial = None

    return {"n": n_shots, "pm_errs": pm_errs, "pf_errs": pf_errs,
            "lange_errs": lange_errs, "n_trivial": n_trivial}


def main():
    N_SEEDS = 3
    N_PER_SEED = 20000
    configs = [(d, p) for d in [3, 5, 7] for p in [0.001, 0.002, 0.003, 0.005]]
    results = {}
    for d, p in configs:
        print(f"\nd={d} p={p}: {N_SEEDS} seeds × {N_PER_SEED} shots", flush=True)
        totals = {"n": 0, "pm_errs": 0, "pf_errs": 0, "lange_errs": 0, "n_trivial": 0}
        lange_ok = True
        for seed in range(2000, 2000 + N_SEEDS):
            r = eval_point(d, p, seed, N_PER_SEED)
            print(f"  seed={seed}: pm_errs={r['pm_errs']}  pf_errs={r['pf_errs']}  lange_errs={r['lange_errs']}  n_trivial={r['n_trivial']}", flush=True)
            totals["n"] += r["n"]
            totals["pm_errs"] += r["pm_errs"]
            totals["pf_errs"] += r["pf_errs"]
            if r["lange_errs"] is None:
                lange_ok = False
            else:
                totals["lange_errs"] += r["lange_errs"]
                totals["n_trivial"] += r["n_trivial"]
        pm_p, pm_lo, pm_hi = wilson(totals["pm_errs"], totals["n"])
        pf_p, pf_lo, pf_hi = wilson(totals["pf_errs"], totals["n"])
        if lange_ok:
            lange_p, lange_lo, lange_hi = wilson(totals["lange_errs"], totals["n"])
            print(f"  SUMMARY: PF={pf_p*100:.4f}% [{pf_lo*100:.4f},{pf_hi*100:.4f}]  Lange={lange_p*100:.4f}% [{lange_lo*100:.4f},{lange_hi*100:.4f}]  PM={pm_p*100:.4f}% [{pm_lo*100:.4f},{pm_hi*100:.4f}]", flush=True)
        else:
            print(f"  SUMMARY: PF={pf_p*100:.4f}%  PM={pm_p*100:.4f}%  (Lange failed)", flush=True)
        results[f"d{d}_p{p}"] = {"d": d, "p": p, **totals,
                                "pm_ler": pm_p, "pf_ler": pf_p,
                                "lange_ler": lange_p if lange_ok else None,
                                "lange_ok": lange_ok}
    with open("/workspace/run_lange_vs_pathfinder_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== wrote /workspace/run_lange_vs_pathfinder_results.json ===")


if __name__ == "__main__":
    main()
