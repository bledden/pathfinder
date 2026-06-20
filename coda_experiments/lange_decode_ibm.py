"""Lange-on-IBM eval, using the verbatim LangeMapper + build_lange_graph from
run_lange_vs_pathfinder.py (the known-working code path that produced
§5.11/§5.12 100K-shot Lange numbers in the paper).

Closes Gap 4 properly without duplicating the LangeWrapper rewrite.
"""
import sys, json
import numpy as np
import stim
import torch
import torch.nn

sys.path.insert(0, '.')
sys.path.insert(0, '/Users/bledden/Documents/pathfinder/coda_experiments/GNN_decoder')

from decode_ibm_result import bitarray_packed_to_bools, wilson_ci
from torch_geometric.nn import knn_graph
from src.gnn_models import GNN_7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LANGE_BASE = '/Users/bledden/Documents/pathfinder/coda_experiments/GNN_decoder/models/circuit_level_noise'


class LangeMapper:
    """Verbatim from bench/results/h200_session2/run_lange_vs_pathfinder.py."""
    def __init__(self, circuit, d):
        coords = circuit.get_detector_coordinates()
        nd = circuit.num_detectors
        det_coords = np.array([coords[i] for i in range(nd)])
        det_coords_int = (det_coords * [1, 1, 1]).astype(np.int64)
        max_x = det_coords_int[:, 0].max() + 1
        max_y = det_coords_int[:, 1].max() + 1
        max_t = det_coords_int[:, 2].max() + 1
        self.syndrome_mask = np.zeros((max_y, max_x, max_t), dtype=np.float32)
        self.det_coords = det_coords_int
        self.d = d
        # memory_z: all detectors measure Z stabilizers
        self.syndrome_mask[:] = 3.0

    def stim_to_syndrome_3D(self, det_events):
        B = det_events.shape[0]
        mask = np.repeat(self.syndrome_mask[None, ...], B, axis=0)
        syn3d = np.zeros_like(mask)
        syn3d[:, self.det_coords[:, 1], self.det_coords[:, 0], self.det_coords[:, 2]] = det_events.astype(np.float32)
        syn3d[np.nonzero(syn3d)] = mask[np.nonzero(syn3d)]
        return syn3d


def build_lange_graph(syndromes_3D, m_nearest_nodes=10):
    defect_inds = np.nonzero(syndromes_3D)
    defects = syndromes_3D[defect_inds]
    defect_inds = np.transpose(np.array(defect_inds))   # (N_defects, 4) = (batch, y, x, t)

    x_defects = defects == 1
    z_defects = defects == 3

    node_features = np.zeros((defects.shape[0], 6), dtype=np.float32)
    node_features[x_defects, 0] = 1
    node_features[x_defects, 2:] = defect_inds[x_defects, ...]
    node_features[z_defects, 1] = 1
    node_features[z_defects, 2:] = defect_inds[z_defects, ...]

    x_cols = [0, 1, 3, 4, 5]
    batch_col = 2
    x = torch.tensor(node_features[:, x_cols]).to(device)
    batch_labels = torch.tensor(node_features[:, batch_col]).long().to(device)
    pos = x[:, 2:]
    edge_index = knn_graph(pos, m_nearest_nodes, batch=batch_labels)
    dist = torch.sqrt(((pos[edge_index[0], :] - pos[edge_index[1], :]) ** 2).sum(dim=1, keepdim=True))
    edge_attr = 1.0 / (dist ** 2 + 1e-8)
    return x, edge_index, batch_labels, edge_attr


def load_lange_model(d, d_t):
    path = f"{LANGE_BASE}/d{d}/d{d}_d_t_{d_t}.pt"
    model = GNN_7(
        hidden_channels_GCN=[32, 128, 256, 512, 512, 256, 256],
        hidden_channels_MLP=[256, 128, 64],
        num_classes=1,
    ).to(device)
    ck = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(ck['model'])
    model.train(False)
    return model, path


def lange_predict(model, det_events, mapper, chunk=500):
    B = det_events.shape[0]
    sigm = torch.nn.Sigmoid()
    preds = np.zeros((B, 1), dtype=np.uint8)
    for i in range(0, B, chunk):
        end = min(i + chunk, B)
        de = det_events[i:end]
        any_flip = de.sum(axis=1) != 0
        if not np.any(any_flip):
            continue
        de_nt = de[any_flip]
        syn3d = mapper.stim_to_syndrome_3D(de_nt)
        x, ei, batch, ea = build_lange_graph(syn3d)
        with torch.no_grad():
            # GNN_7.forward signature: (x, edge_index, batch, edge_attr)
            out = model(x, ei, batch, ea)
            probs = sigm(out).cpu().numpy()
        sub = np.zeros((de_nt.shape[0], 1), dtype=np.uint8)
        sub[:, 0] = (probs.squeeze() > 0.5).astype(np.uint8)
        idx_in_chunk = np.where(any_flip)[0]
        preds[i + idx_in_chunk] = sub
    return preds


def run_one(D, R):
    print(f"\n=== IBM d={D} r={R} : Lange (verbatim run_lange_vs_pathfinder mapper) ===")
    obj = json.load(open(f'ibm_d{D}r{R}_result.json'))
    r = obj['result']
    packed = np.array(r['shots_array_packed'], dtype=np.uint8)
    measurements = bitarray_packed_to_bools(packed, r['n_shots'], r['n_bits'])
    print(f"  shots={measurements.shape[0]}, bits={measurements.shape[1]}")

    clean = stim.Circuit.generated('surface_code:rotated_memory_z', distance=D, rounds=R)
    m2d = clean.compile_m2d_converter()
    det, obs = m2d.convert(measurements=measurements, separate_observables=True)
    n_shots = det.shape[0]
    print(f"  detectors: {det.shape}, det_flip_rate={det.mean():.4f}")

    mapper = LangeMapper(clean, D)
    model, ckpt_path = load_lange_model(D, R)
    print(f"  loaded Lange ckpt: {ckpt_path}")

    pred = lange_predict(model, det.astype(np.uint8), mapper)
    wrong = np.any(pred != obs, axis=1)
    k = int(wrong.sum())
    ler, lo, hi = wilson_ci(k, n_shots)
    print(f"\n  Lange LER: {ler*100:6.3f}% CI=[{lo*100:.3f}%, {hi*100:.3f}%] ({k}/{n_shots} errors)")
    return {
        'distance': D, 'rounds': R, 'n_shots': n_shots,
        'det_flip_rate': float(det.mean()),
        'obs_flip_rate': float(obs.mean()),
        'lange': {'ler': ler, 'ci': [lo, hi], 'errors': k,
                  'ckpt': ckpt_path, 'd_t': R},
    }


def main():
    print("Running Lange (published, d_t=R) on IBM Heron r2 data...")
    out_all = {}
    for (D, R) in [(3, 3), (5, 5)]:
        out_all[f'd{D}r{R}'] = run_one(D, R)
        with open(f'ibm_d{D}r{R}_lange_v2.json', 'w') as f:
            json.dump(out_all[f'd{D}r{R}'], f, indent=2)

    print("\n=== SUMMARY (Lange added to head-to-head) ===")
    print(f"{'(d, r)':<10}{'PM':<12}{'PFWL3S':<18}{'Lange (pub)':<15}")
    print(f"{'-'*55}")
    print(f"{'d=3 r=3':<10}{'28.560':<12}{'28.980 (3-seed)':<18}"
          f"{out_all['d3r3']['lange']['ler']*100:<15.3f}")
    print(f"{'d=5 r=5':<10}{'45.680':<12}{'47.270 (calib)':<18}"
          f"{out_all['d5r5']['lange']['ler']*100:<15.3f}")


if __name__ == '__main__':
    main()
