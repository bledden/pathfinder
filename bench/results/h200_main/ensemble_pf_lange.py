"""Pathfinder + Lange ensemble scoring (task #41).

At matched 4-parameter noise, records per-shot predictions from Pathfinder,
Lange, and PyMatching. Reports individual LERs plus several ensemble
strategies: all-three-wrong (oracle lower bound), 3-way majority vote,
and confidence-thresholded gating where Pathfinder is used when confident
and another decoder (Lange or PM) otherwise.

Pathfinder checkpoint defaults to fixed_d{d} (the 4-parameter retrain) if
present, otherwise falls back to the Table-1 model.

Usage:
  python3 ensemble_pf_lange.py --pathfinder-dir /workspace/pathfinder/train/checkpoints
"""
import sys, os, json, argparse
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace/GNN_decoder")
import numpy as np
import torch, stim, pymatching
from torch_geometric.nn import knn_graph
from src.gnn_models import GNN_7
from model import NeuralDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, ctr - half), min(1.0, ctr + half)


class LangeWrapper:
    def __init__(self, d, d_t, m_nearest_nodes=10, power=2):
        self.d = d; self.d_t = d_t
        self.m = m_nearest_nodes; self.power = power
        self.model = GNN_7(
            hidden_channels_GCN=[32, 128, 256, 512, 512, 256, 256],
            hidden_channels_MLP=[256, 128, 64], num_classes=1,
        ).to(device)
        path = f"/workspace/GNN_decoder/models/circuit_level_noise/d{d}/d{d}_d_t_{d_t}.pt"
        ck = torch.load(path, weights_only=False, map_location=device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.sigmoid = torch.nn.Sigmoid()

    def init_from_circuit(self, circuit):
        coords = circuit.get_detector_coordinates()
        det_c = np.array(list(coords.values()))
        det_c[:, :2] = det_c[:, :2] / 2
        self.det_c = det_c.astype(np.uint8)
        sz = self.d + 1
        sx = np.zeros((sz, sz), dtype=np.uint8)
        sx[::2, 1:sz - 1:2] = 1
        sx[1::2, 2::2] = 1
        sz_mask = np.rot90(sx) * 3
        self.syn_mask = np.dstack([sx + sz_mask] * (self.d_t + 1))

    def stim_to_3d(self, det):
        mask = np.repeat(self.syn_mask[None, ...], det.shape[0], 0)
        s = np.zeros_like(mask)
        s[:, self.det_c[:, 1], self.det_c[:, 0], self.det_c[:, 2]] = det
        s[np.nonzero(s)] = mask[np.nonzero(s)]
        return s

    def predict_batch(self, det):
        B = det.shape[0]
        preds_all = np.zeros((B, 1), dtype=np.uint8)
        any_flip = np.sum(det, axis=1) != 0
        if not np.any(any_flip):
            return preds_all
        det_nt = det[any_flip]
        s3d = self.stim_to_3d(det_nt).astype(np.float32)
        inds = np.nonzero(s3d)
        defs = s3d[inds]
        inds_t = np.transpose(np.array(inds))
        x_def = defs == 1; z_def = defs == 3
        nf = np.zeros((defs.shape[0], 6), dtype=np.float32)
        nf[x_def, 0] = 1; nf[x_def, 2:] = inds_t[x_def, ...]
        nf[z_def, 1] = 1; nf[z_def, 2:] = inds_t[z_def, ...]
        x_cols = [0, 1, 3, 4, 5]
        x = torch.tensor(nf[:, x_cols]).to(device)
        batch = torch.tensor(nf[:, 2]).long().to(device)
        pos = x[:, 2:]
        ei = knn_graph(pos, self.m, batch=batch)
        dist = torch.sqrt(((pos[ei[0], :] - pos[ei[1], :]) ** 2).sum(dim=1, keepdim=True))
        edge_attr = 1.0 / (dist ** self.power)
        with torch.no_grad():
            out = self.model(x, ei, edge_attr, batch)
        probs = self.sigmoid(out).cpu().numpy()
        preds_nt = (probs > 0.5).astype(np.uint8)
        preds_all[any_flip] = preds_nt
        return preds_all


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

    def to_tensor(self, det):
        B = det.shape[0]
        T, H, W = self.grid
        t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        d = torch.from_numpy(det.astype(np.float32))
        for i in range(self.nd):
            t[:, 0, self.det_idx[i, 0], self.det_idx[i, 1], self.det_idx[i, 2]] = d[:, i]
        return t


def load_pathfinder(d, ckpt_dir):
    candidates = [
        f"{ckpt_dir}/fixed_d{d}/best_model.pt",
        f"{ckpt_dir}/d{d}_final/best_model.pt" if d == 7 else f"{ckpt_dir}/d{d}_muon/best_model.pt",
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"  loading Pathfinder from {p}", flush=True)
            ck = torch.load(p, weights_only=False, map_location=device)
            m = NeuralDecoder(ck["config"]).to(device)
            m.load_state_dict(ck["model_state_dict"])
            m.eval()
            return m, p
    raise FileNotFoundError(f"No Pathfinder checkpoint found for d={d} in {ckpt_dir}")


def make_circuit(d, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)


def run_point(d, p, ckpt_dir, n_shots, seed):
    c = make_circuit(d, p)
    sampler = c.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots=n_shots, separate_observables=True)
    det = det.astype(np.uint8); obs = obs.astype(np.uint8)

    dem = c.detector_error_model(decompose_errors=True)
    pm = pymatching.Matching.from_detector_error_model(dem)
    pm_preds = pm.decode_batch(det).astype(np.uint8)

    pf_model, pf_src = load_pathfinder(d, ckpt_dir)
    pfm = PathfinderMapper(c)
    pf_preds = np.zeros_like(obs)
    pf_logits = np.zeros((n_shots, 1), dtype=np.float32)
    for i in range(0, n_shots, 1000):
        bd = det[i:i + 1000]
        syn = pfm.to_tensor(bd).to(device)
        with torch.no_grad():
            lg = pf_model(syn)
        pf_logits[i:i + 1000] = lg.cpu().numpy()
        pf_preds[i:i + 1000] = (lg > 0).cpu().numpy().astype(np.uint8)

    lw = LangeWrapper(d, d)
    lw.init_from_circuit(c)
    lange_preds = np.zeros_like(obs)
    bs = 500
    for i in range(0, n_shots, bs):
        lange_preds[i:i + bs] = lw.predict_batch(det[i:i + bs])

    pf_wrong = np.any(pf_preds != obs, axis=1)
    la_wrong = np.any(lange_preds != obs, axis=1)
    pm_wrong = np.any(pm_preds != obs, axis=1)

    all_three_wrong = pf_wrong & la_wrong & pm_wrong

    maj = ((pf_preds.astype(int) + lange_preds.astype(int) + pm_preds.astype(int)) >= 2).astype(np.uint8)
    maj_wrong = np.any(maj != obs, axis=1)

    thresholds = [1.0, 2.0, 3.0, 4.0]
    gate_lange = {}
    gate_pm = {}
    for T in thresholds:
        conf = np.abs(pf_logits).max(axis=1) > T
        picked_l = np.where(conf[:, None], pf_preds, lange_preds)
        picked_m = np.where(conf[:, None], pf_preds, pm_preds)
        gate_lange[T] = int(np.sum(np.any(picked_l != obs, axis=1)))
        gate_pm[T] = int(np.sum(np.any(picked_m != obs, axis=1)))

    return {
        "n": n_shots,
        "pf": int(pf_wrong.sum()),
        "lange": int(la_wrong.sum()),
        "pm": int(pm_wrong.sum()),
        "all_three_wrong": int(all_three_wrong.sum()),
        "majority": int(maj_wrong.sum()),
        "gate_pf_then_lange": gate_lange,
        "gate_pf_then_pm": gate_pm,
        "pathfinder_src": pf_src,
        "both_wrong_pf_lange": int((pf_wrong & la_wrong).sum()),
        "pf_right_lange_wrong": int((~pf_wrong & la_wrong).sum()),
        "pf_wrong_lange_right": int((pf_wrong & ~la_wrong).sum()),
    }


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--pathfinder-dir", type=str, default="/workspace/pathfinder/train/checkpoints")
    a.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    a.add_argument("--noise-rates", type=float, nargs="+", default=[0.003, 0.005, 0.007, 0.010])
    a.add_argument("--n-per-seed", type=int, default=20000)
    a.add_argument("--n-seeds", type=int, default=3)
    a.add_argument("--output", type=str, default="/workspace/ensemble_results.json")
    args = a.parse_args()

    results = {}
    for d in args.distances:
        for p in args.noise_rates:
            key = f"d{d}_p{p}"
            print(f"\n=== {key} ===", flush=True)
            tot = None
            for seed in range(3000, 3000 + args.n_seeds):
                r = run_point(d, p, args.pathfinder_dir, args.n_per_seed, seed)
                print(f"  seed={seed}: pf={r['pf']}  lange={r['lange']}  pm={r['pm']}  maj={r['majority']}  all3={r['all_three_wrong']}", flush=True)
                if tot is None:
                    tot = {k: (v if isinstance(v, (int, float, str)) else dict(v)) for k, v in r.items()}
                else:
                    for k in ["n", "pf", "lange", "pm", "all_three_wrong", "majority", "both_wrong_pf_lange", "pf_right_lange_wrong", "pf_wrong_lange_right"]:
                        tot[k] += r[k]
                    for T in r["gate_pf_then_lange"]:
                        tot["gate_pf_then_lange"][T] += r["gate_pf_then_lange"][T]
                        tot["gate_pf_then_pm"][T] += r["gate_pf_then_pm"][T]

            n = tot["n"]
            pf_p, pf_lo, pf_hi = wilson(tot["pf"], n)
            la_p, la_lo, la_hi = wilson(tot["lange"], n)
            pm_p, pm_lo, pm_hi = wilson(tot["pm"], n)
            maj_p, maj_lo, maj_hi = wilson(tot["majority"], n)
            all3_p, _, _ = wilson(tot["all_three_wrong"], n)
            print(f"  -> PF={pf_p*100:.4f}% Lange={la_p*100:.4f}% PM={pm_p*100:.4f}% Majority={maj_p*100:.4f}% OracleLB={all3_p*100:.4f}%", flush=True)
            gate_le = {str(T): wilson(tot["gate_pf_then_lange"][T], n)[0] for T in tot["gate_pf_then_lange"]}
            gate_pm_out = {str(T): wilson(tot["gate_pf_then_pm"][T], n)[0] for T in tot["gate_pf_then_pm"]}

            results[key] = {
                "d": d, "p": p, "n": n,
                "pf_ler": pf_p, "pf_ci": [pf_lo, pf_hi],
                "lange_ler": la_p, "lange_ci": [la_lo, la_hi],
                "pm_ler": pm_p, "pm_ci": [pm_lo, pm_hi],
                "majority_ler": maj_p, "majority_ci": [maj_lo, maj_hi],
                "oracle_lb_all_three_wrong": all3_p,
                "gate_pf_then_lange": gate_le,
                "gate_pf_then_pm": gate_pm_out,
                "pathfinder_src": tot["pathfinder_src"],
                "both_wrong_pf_lange": tot["both_wrong_pf_lange"],
                "pf_right_lange_wrong": tot["pf_right_lange_wrong"],
                "pf_wrong_lange_right": tot["pf_wrong_lange_right"],
            }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
