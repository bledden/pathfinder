"""
Ensemble of Pathfinder + Lange + PyMatching with confidence-based gating.
Use each decoder's highest-confidence prediction. OR-oracle baseline shows
upper bound if we could always pick the right decoder.

Exploits the fact that all three decoders have near-disjoint failure modes.
"""
import sys, os, json, time
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace/GNN_decoder")
sys.path.insert(0, "/workspace")
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


def make_circuit(d, p):
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)


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

    def get_graph(self, det):
        s3d = self.stim_to_syn3d(det).astype(np.float32)
        inds = np.nonzero(s3d)
        defs = s3d[inds]
        inds = np.transpose(np.array(inds))
        xd = defs == 1; zd = defs == 3
        nf = np.zeros((defs.shape[0], 6), dtype=np.float32)
        nf[xd, 0] = 1; nf[xd, 2:] = inds[xd, ...]
        nf[zd, 1] = 1; nf[zd, 2:] = inds[zd, ...]
        x_cols = [0, 1, 3, 4, 5]
        x = torch.tensor(nf[:, x_cols]).to(device)
        batch = torch.tensor(nf[:, 2]).long().to(device)
        pos = x[:, 2:]
        ei = knn_graph(pos, self.m_nearest_nodes, batch=batch)
        dist = torch.sqrt(((pos[ei[0],:] - pos[ei[1],:])**2).sum(dim=1, keepdim=True))
        ea = 1.0 / (dist ** self.power + 1e-8)
        return x, ei, batch, ea

    def predict_logits(self, det):
        """Returns raw logits (before sigmoid) for each non-trivial shot."""
        if det.shape[0] == 0:
            return np.zeros((0, 1), dtype=np.float32)
        x, ei, batch, ea = self.get_graph(det.astype(np.uint8))
        with torch.no_grad():
            out = self.model(x, ei, batch, ea)
        return out.cpu().numpy()


def load_pathfinder(d):
    paths = {3: "/workspace/pathfinder/train/checkpoints/best_model.pt",
             5: "/workspace/pathfinder/train/checkpoints/d5_muon/best_model.pt",
             7: "/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt"}
    ck = torch.load(paths[d], weights_only=False, map_location=device)
    m = NeuralDecoder(ck["config"]).to(device); m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


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


def eval_ensemble(d, p, seed, n_shots, pathfinder, lange, pfm):
    c = make_circuit(d, p)
    sampler = c.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots=n_shots, separate_observables=True)
    det = det.astype(np.uint8); obs = obs.astype(np.uint8)

    dem = c.detector_error_model(decompose_errors=True)
    pm = pymatching.Matching.from_detector_error_model(dem)
    pm_preds = pm.decode_batch(det)

    # Pathfinder logits (on all shots)
    pf_logits_all = np.zeros((n_shots, 1), dtype=np.float32)
    for i in range(0, n_shots, 1000):
        bd = det[i:i+1000]
        syn = pfm.to_tensor(bd).to(device)
        with torch.no_grad():
            lg = pathfinder(syn).cpu().numpy()
        pf_logits_all[i:i+1000] = lg
    pf_preds = (pf_logits_all > 0).astype(np.uint8)
    pf_conf = np.abs(pf_logits_all).flatten()

    # Lange logits (only non-trivial)
    shots_nt = np.sum(det, axis=1) != 0
    det_nt = det[shots_nt]
    lange_logits_all = np.zeros((n_shots, 1), dtype=np.float32)
    lange_preds = np.zeros((n_shots, 1), dtype=np.uint8)
    lange_conf = np.zeros(n_shots, dtype=np.float32)
    lange.init_from_circuit(c)
    bs = 500
    lg_idx_map = np.where(shots_nt)[0]
    for i in range(0, len(det_nt), bs):
        bd = det_nt[i:i+bs]
        lg = lange.predict_logits(bd)
        lange_logits_all[lg_idx_map[i:i+bs]] = lg
    lange_preds = (lange_logits_all > 0).astype(np.uint8)
    # Pseudo-confidence for Lange: sigmoid-prob → |2*prob - 1|
    lange_prob = 1 / (1 + np.exp(-lange_logits_all))
    lange_conf = np.abs(2 * lange_prob - 1).flatten()
    # Trivial shots: confidence = 1 (they're always right)
    lange_conf[~shots_nt] = 1.0
    lange_preds[~shots_nt] = 0  # trivial → predict no flip (obs_flips are 0 for trivials iff no X error)

    # Error counts
    pm_errs = int(np.sum(np.any(pm_preds != obs, axis=1)))
    pf_errs = int(np.sum(np.any(pf_preds != obs, axis=1)))
    lange_errs = int(np.sum(np.any(lange_preds != obs, axis=1)))

    # Ensemble: pick decoder with highest confidence
    # For PM, "confidence" is 1 (PM is deterministic). Treat as tiebreaker.
    # Build confidence array [3, n_shots]: [PF_conf, Lange_conf, PM_conf=0.5]
    # Pick the highest-confidence neural prediction if above threshold; else PM.

    # Strategy 1: max-confidence neural + PM fallback
    ens_strategies = {}

    for thresh_pf, thresh_lange in [(2.0, 0.5), (3.0, 0.7), (5.0, 0.9)]:
        ens_preds = np.zeros_like(obs)
        for i in range(n_shots):
            pf_c = pf_conf[i]
            lg_c = lange_conf[i]
            if pf_c > thresh_pf and lg_c > thresh_lange:
                # Both confident: use whichever has higher confidence
                # Normalize lange_conf to similar range as pf_conf
                if pf_c > 10 * lg_c:
                    ens_preds[i] = pf_preds[i]
                else:
                    ens_preds[i] = lange_preds[i]
            elif pf_c > thresh_pf:
                ens_preds[i] = pf_preds[i]
            elif lg_c > thresh_lange:
                ens_preds[i] = lange_preds[i]
            else:
                ens_preds[i] = pm_preds[i]
        ens_errs = int(np.sum(np.any(ens_preds != obs, axis=1)))
        ens_strategies[f"pf{thresh_pf}_lg{thresh_lange}"] = ens_errs

    # Simple voting ensemble (3-way majority where possible)
    vote_preds = np.zeros_like(obs)
    for i in range(n_shots):
        votes = [pf_preds[i][0], lange_preds[i][0], pm_preds[i][0]]
        vote_preds[i, 0] = 1 if sum(votes) >= 2 else 0
    vote_errs = int(np.sum(np.any(vote_preds != obs, axis=1)))

    # OR-oracle
    or_errs = 0
    for i in range(n_shots):
        pf_ok = np.all(pf_preds[i] == obs[i])
        lg_ok = np.all(lange_preds[i] == obs[i])
        pm_ok = np.all(pm_preds[i] == obs[i])
        if not (pf_ok or lg_ok or pm_ok):
            or_errs += 1

    return {"n": n_shots, "pm_errs": pm_errs, "pf_errs": pf_errs, "lange_errs": lange_errs,
            "vote_errs": vote_errs, "or_oracle_errs": or_errs, "ensemble": ens_strategies}


def main():
    configs = [(d, p) for d in [3, 5, 7] for p in [0.003, 0.005, 0.007, 0.010]]
    N_SEEDS = 3
    N_PER_SEED = 10000
    results = {}
    for d, p in configs:
        print(f"\n=== d={d} p={p} ===", flush=True)
        c = make_circuit(d, p)
        pfm = PFMapper(c)
        pf = load_pathfinder(d)
        lg = LangeWrapper(d, d)

        tot = {"n": 0, "pm": 0, "pf": 0, "lange": 0, "vote": 0, "or": 0, "ens": {}}
        for seed in range(3000, 3000 + N_SEEDS):
            r = eval_ensemble(d, p, seed, N_PER_SEED, pf, lg, pfm)
            print(f"  seed={seed}: pm={r['pm_errs']} pf={r['pf_errs']} lange={r['lange_errs']} vote={r['vote_errs']} or={r['or_oracle_errs']} ens={r['ensemble']}", flush=True)
            tot["n"] += r["n"]
            tot["pm"] += r["pm_errs"]; tot["pf"] += r["pf_errs"]; tot["lange"] += r["lange_errs"]
            tot["vote"] += r["vote_errs"]; tot["or"] += r["or_oracle_errs"]
            for k, v in r["ensemble"].items():
                tot["ens"].setdefault(k, 0)
                tot["ens"][k] += v

        n = tot["n"]
        pm_p, pm_lo, pm_hi = wilson(tot["pm"], n)
        pf_p, _, _ = wilson(tot["pf"], n)
        lg_p, _, _ = wilson(tot["lange"], n)
        v_p, v_lo, v_hi = wilson(tot["vote"], n)
        or_p, or_lo, or_hi = wilson(tot["or"], n)
        print(f"  PF={pf_p*100:.4f}%  Lange={lg_p*100:.4f}%  PM={pm_p*100:.4f}%  Vote={v_p*100:.4f}% [{v_lo*100:.4f},{v_hi*100:.4f}]  OR-oracle={or_p*100:.4f}%", flush=True)
        for k, v in tot["ens"].items():
            e_p, e_lo, e_hi = wilson(v, n)
            print(f"    Ensemble {k}: {e_p*100:.4f}% [{e_lo*100:.4f},{e_hi*100:.4f}]", flush=True)
        results[f"d{d}_p{p}"] = {"d": d, "p": p, "total_n": n, **tot,
                                "pm_ler": pm_p, "pf_ler": pf_p, "lange_ler": lg_p,
                                "vote_ler": v_p, "or_oracle_ler": or_p}

    with open("/workspace/ensemble_all_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
