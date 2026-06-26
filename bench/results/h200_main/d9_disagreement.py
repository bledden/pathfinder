"""d=9 Triad voter-disagreement matrix (Coda Q2).

Runs the three Triad voters — PFWL3S-H256-d9 (3-seed logit avg), Lange-d9 GNN,
PyMatching — on the SAME d=9 shots at the winning operational rates (p=0.007, 0.010),
and records the full 8-cell 3-way agreement matrix indexed by (pf_wrong, lange_wrong,
pm_wrong). For a single logical observable, majority-vote is wrong iff >=2 voters are
wrong, so the 8 cells fully determine the Triad LER AND show whether the win comes from
INDEPENDENT errors (many single-voter failures that get outvoted) or is a coincidence.

Local: PFWL3S on MPS, Lange GNN + knn_graph on CPU (torch_geometric), PM on CPU.
"""
import sys, os, json, argparse, math
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO + "/train")
sys.path.insert(0, REPO + "/coda_experiments/GNN_decoder")
import numpy as np, torch, stim, pymatching
from torch_geometric.nn import knn_graph
from src.gnn_models import GNN_7
from model import NeuralDecoder

PF_DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # MPS OK at batch<=250; corrupts d=9 conv at batch>500
LA_DEV = torch.device("cpu")  # torch_geometric knn_graph is safest on CPU


def knn_graph_np(pos_np, k, batch_np):
    """Pure-numpy replacement for torch_geometric.nn.knn_graph (no pyg-lib needed).
    Returns edge_index [2,E] with row0=source(neighbor), row1=target(node), loop=False,
    flow='source_to_target' — matching torch_geometric defaults, k-NN within each batch graph."""
    srcs, tgts = [], []
    for b in np.unique(batch_np):
        idx = np.where(batch_np == b)[0]
        if len(idx) < 2:
            continue
        P = pos_np[idx]
        diff = P[:, None, :] - P[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        np.fill_diagonal(d2, np.inf)
        kk = min(k, len(idx) - 1)
        nn = np.argpartition(d2, kk - 1, axis=1)[:, :kk]   # kk nearest local indices per node
        tgts.append(np.repeat(idx, kk))                    # target = the node
        srcs.append(idx[nn.reshape(-1)])                   # source = its neighbor
    if not srcs:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(np.stack([np.concatenate(srcs), np.concatenate(tgts)]), dtype=torch.long)


def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0, 0.0
    p = k / n; den = 1 + z * z / n; ctr = (p + z * z / (2 * n)) / den
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return p, max(0, ctr - half), min(1, ctr + half)


class LangeWrapper:
    def __init__(self, d, d_t, m_nearest_nodes=10, power=2):
        self.d, self.d_t, self.m, self.power = d, d_t, m_nearest_nodes, power
        self.model = GNN_7(hidden_channels_GCN=[32, 128, 256, 512, 512, 512, 512],
                           hidden_channels_MLP=[512, 256, 128, 64, 32, 16], num_classes=1).to(LA_DEV)
        path = f"{REPO}/coda_experiments/GNN_decoder/models/circuit_level_noise/d{d}/d{d}_d_t_{d_t}.pt"
        ck = torch.load(path, weights_only=False, map_location=LA_DEV)
        self.model.load_state_dict(ck["model"]); self.model.eval()
        self.sig = torch.nn.Sigmoid()

    def init_from_circuit(self, circuit):
        coords = circuit.get_detector_coordinates()
        det_c = np.array(list(coords.values())); det_c[:, :2] = det_c[:, :2] / 2
        self.det_c = det_c.astype(np.uint8)
        sz = self.d + 1
        sx = np.zeros((sz, sz), dtype=np.uint8); sx[::2, 1:sz - 1:2] = 1; sx[1::2, 2::2] = 1
        self.syn_mask = np.dstack([sx + np.rot90(sx) * 3] * (self.d_t + 1))

    def _to3d(self, det):
        mask = np.repeat(self.syn_mask[None, ...], det.shape[0], 0)
        s = np.zeros_like(mask)
        s[:, self.det_c[:, 1], self.det_c[:, 0], self.det_c[:, 2]] = det
        s[np.nonzero(s)] = mask[np.nonzero(s)]
        return s

    def predict_batch(self, det):
        B = det.shape[0]; preds = np.zeros((B, 1), dtype=np.uint8)
        flip = np.sum(det, axis=1) != 0
        if not np.any(flip): return preds
        s3d = self._to3d(det[flip]).astype(np.float32)
        inds = np.nonzero(s3d); defs = s3d[inds]; it = np.transpose(np.array(inds))
        nf = np.zeros((defs.shape[0], 6), dtype=np.float32)
        xd, zd = defs == 1, defs == 3
        nf[xd, 0] = 1; nf[xd, 2:] = it[xd]; nf[zd, 1] = 1; nf[zd, 2:] = it[zd]
        x = torch.tensor(nf[:, [0, 1, 3, 4, 5]]).to(LA_DEV)
        batch = torch.tensor(nf[:, 2]).long().to(LA_DEV); pos = x[:, 2:]
        ei = knn_graph_np(pos.cpu().numpy(), self.m, batch.cpu().numpy()).to(LA_DEV)
        dist = torch.sqrt(((pos[ei[0]] - pos[ei[1]]) ** 2).sum(1, keepdim=True))
        ea = 1.0 / (dist ** self.power)
        with torch.no_grad():
            out = self.model(x, ei, batch, ea)  # GNN_7.forward(x, edge_index, batch, edge_attr)
        preds[flip] = (self.sig(out).cpu().numpy() > 0.5).astype(np.uint8)
        return preds


class PFMapper:
    def __init__(self, c):
        nd = c.num_detectors; co = c.get_detector_coordinates()
        ac = np.array([co[i] for i in range(nd)]); sp, tm = ac[:, :-1], ac[:, -1]
        tu = np.sort(np.unique(tm)); xu = np.sort(np.unique(sp[:, 0]))
        yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
        self.grid = (len(tu), len(yu), len(xu))
        tmm = {v: i for i, v in enumerate(tu)}; xm = {v: i for i, v in enumerate(xu)}; ym = {v: i for i, v in enumerate(yu)}
        di = np.zeros((nd, 3), dtype=np.int64)
        for k in range(nd):
            cc = co[k]; di[k] = [tmm[cc[-1]], ym.get(cc[1], 0) if len(cc) > 2 else 0, xm[cc[0]]]
        self.di, self.nd = di, nd

    def to_tensor(self, det):
        B = det.shape[0]; T, H, W = self.grid
        t = torch.zeros(B, 1, T, H, W, dtype=torch.float32); d = torch.from_numpy(det.astype(np.float32))
        for i in range(self.nd): t[:, 0, self.di[i, 0], self.di[i, 1], self.di[i, 2]] = d[:, i]
        return t


def make_circuit(d, p):
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)


def load_pfwl3s(d):
    ms = []
    for s in (0, 1, 2):
        p = f"{REPO}/bench/results/h200_main/tierC1/pathfinder_wide_long_d{d}_seed{s}/best_model.pt"
        ck = torch.load(p, weights_only=False, map_location=PF_DEV)
        m = NeuralDecoder(ck["config"]).to(PF_DEV); m.load_state_dict(ck["model_state_dict"]); m.eval()
        ms.append(m)
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=9)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.007, 0.010])
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--out", type=str, default=REPO + "/bench/results/h200_main/d9_disagreement_matrix.json")
    a = ap.parse_args()
    print(f"PF_DEV={PF_DEV} LA_DEV={LA_DEV} d={a.d} rates={a.rates} n={a.n}/rate", flush=True)
    pf_models = load_pfwl3s(a.d)
    out = {"d": a.d, "n_per_rate": a.n, "rates": {}}
    for p in a.rates:
        c = make_circuit(a.d, p)
        pm = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
        pfm = PFMapper(c); lw = LangeWrapper(a.d, a.d); lw.init_from_circuit(c)
        cells = np.zeros(8, dtype=np.int64)  # index = pf_w*4 + la_w*2 + pm_w
        done = 0
        for off in range(0, a.n, a.chunk):
            bs = min(a.chunk, a.n - off)
            det, obs = c.compile_detector_sampler(seed=7000 + off).sample(shots=bs, separate_observables=True)
            det = det.astype(np.uint8); obs = obs.astype(np.uint8)
            pm_pred = pm.decode_batch(det).astype(np.uint8)
            # PFWL3S 3-seed logit average
            acc = np.zeros((bs, obs.shape[1]), dtype=np.float32)
            for m in pf_models:
                lg = np.zeros((bs, obs.shape[1]), dtype=np.float32)
                for i in range(0, bs, 250):  # <=250: MPS-safe for d=9
                    syn = pfm.to_tensor(det[i:i + 250]).to(PF_DEV)
                    with torch.no_grad(): lg[i:i + 250] = m(syn).cpu().numpy()
                acc += lg
            pf_pred = (acc > 0).astype(np.uint8)
            la_pred = lw.predict_batch(det)
            pf_w = np.any(pf_pred != obs, axis=1).astype(np.int64)
            la_w = np.any(la_pred != obs, axis=1).astype(np.int64)
            pm_w = np.any(pm_pred != obs, axis=1).astype(np.int64)
            code = pf_w * 4 + la_w * 2 + pm_w
            for cc in range(8): cells[cc] += int(np.sum(code == cc))
            done += bs
            if off % 5000 == 0: print(f"  p={p} {done}/{a.n}", flush=True)
        n = int(cells.sum())
        pop = [bin(i).count("1") for i in range(8)]
        pf_n = sum(cells[i] for i in range(8) if (i >> 2) & 1)
        la_n = sum(cells[i] for i in range(8) if (i >> 1) & 1)
        pm_n = sum(cells[i] for i in range(8) if i & 1)
        maj_n = sum(cells[i] for i in range(8) if pop[i] >= 2)
        all3 = int(cells[7]); none = int(cells[0])
        labels = {i: f"pf{'X' if (i>>2)&1 else '.'}_la{'X' if (i>>1)&1 else '.'}_pm{'X' if i&1 else '.'}" for i in range(8)}
        res = {
            "n": n,
            "cells": {labels[i]: int(cells[i]) for i in range(8)},
            "pf_ler": wilson(pf_n, n), "lange_ler": wilson(la_n, n),
            "pm_ler": wilson(pm_n, n), "majority_ler": wilson(maj_n, n),
            "all_three_wrong": wilson(all3, n), "all_correct": none,
            # Triad recoveries (single-voter failures outvoted by the other two)
            "pf_solo_fail_recovered": int(cells[0b100]),   # pf wrong, la+pm right -> maj right
            "lange_solo_fail_recovered": int(cells[0b010]),
            "pm_solo_fail_recovered": int(cells[0b001]),
            # cases majority still loses (>=2 wrong), broken down
            "both_neural_wrong_pm_right": int(cells[0b110]),  # PM independent-correct (but outvoted)
            "pf_pm_wrong_lange_right": int(cells[0b101]),
            "lange_pm_wrong_pf_right": int(cells[0b011]),
        }
        # pairwise error correlation (phi) to show (in)dependence
        def phi(a_bit, b_bit):
            A = np.array([(i >> a_bit) & 1 for i in range(8)]); B = np.array([(i >> b_bit) & 1 for i in range(8)])
            n11 = sum(cells[i] for i in range(8) if A[i] and B[i]); n10 = sum(cells[i] for i in range(8) if A[i] and not B[i])
            n01 = sum(cells[i] for i in range(8) if not A[i] and B[i]); n00 = sum(cells[i] for i in range(8) if not A[i] and not B[i])
            den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
            return (n11 * n00 - n10 * n01) / den if den else 0.0
        res["phi_pf_lange"] = phi(2, 1); res["phi_pf_pm"] = phi(2, 0); res["phi_lange_pm"] = phi(1, 0)
        out["rates"][f"p{p}"] = res
        print(f"  ==> p={p}: PF={pf_n/n*100:.3f}% Lange={la_n/n*100:.3f}% PM={pm_n/n*100:.3f}% MAJ={maj_n/n*100:.3f}% "
              f"all3={all3/n*100:.3f}% | phi(pf,la)={res['phi_pf_lange']:.3f} phi(pf,pm)={res['phi_pf_pm']:.3f} phi(la,pm)={res['phi_lange_pm']:.3f}", flush=True)
        json.dump(out, open(a.out, "w"), indent=2)  # checkpoint after each rate
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
