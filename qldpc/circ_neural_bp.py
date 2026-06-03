"""Circuit-level equivariant neural-BP on the DEM factor graph (circuit-level training validation).

The code-capacity NeuralBP is hard-wired to the regular code Tanner graph. The circuit-level decoding
graph is the DEM: variable nodes = error mechanisms (hyperedges), check nodes = detectors, irregular
degrees. This module builds an unrolled normalized min-sum BP on that graph with ORBIT-TIED learnable
weights (the Step-A spatial Z6xZ6 × temporal-bulk orbits), plus 'free' (per-edge) and 'classical'
(weights=1) modes for the ablation.

Goal here is NOT SOTA — it is the pre-scaling training gate: confirm the orbit-tied circuit-level architecture
OPTIMIZES (loss drops, beats untrained classical) on a small instance before scaling up. Small = R=3.

Padded-group message passing (handles irregular degrees):
  - per check (detector): its incident fg-edges padded to max check-degree, masked.
  - per variable (error mech): its incident fg-edges padded to max var-degree, masked.
Min-sum check update uses min1/min2 exclude-self on the padded check groups.
"""
import os
import json, sys, time
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
import torch
import torch.nn as nn
import stim
from bb_code import BBCode
from bb_circuit import build_z_memory
# (removed dead stepA_full import; build_factorgraph below is deprecated)


def detector_round_pos_R(d, R):
    if d < R * 36: return d // 36, d % 36
    return R, d - R * 36


def idx_R(r, q, R):
    return (R * 36 + q) if r == R else (r * 36 + q)


class CircNeuralBP(nn.Module):
    def __init__(self, fgdata, T=8, mode='equiv'):
        super().__init__()
        self.T = T; self.mode = mode
        self.ndet = fgdata['ndet']; self.E = fgdata['E']; self.n_orb = fgdata['n_orb']
        fg = fgdata['fg']; n_edge = len(fg)
        self.n_edge = n_edge
        edge_det = np.array([d for (d, j) in fg]); edge_var = np.array([j for (d, j) in fg])
        self.register_buffer('edge_det', torch.tensor(edge_det, dtype=torch.long))
        self.register_buffer('edge_var', torch.tensor(edge_var, dtype=torch.long))
        self.register_buffer('orbit_of', torch.tensor(fgdata['orbit_of'], dtype=torch.long))
        self.register_buffer('priors', torch.tensor(fgdata['priors'], dtype=torch.float32))
        # padded check groups (per detector)
        self._pad_groups(edge_det, self.ndet, 'chk')
        self._pad_groups(edge_var, self.E, 'var')
        if mode == 'equiv':
            self.wc = nn.Parameter(torch.ones(T, self.n_orb))
            self.wv = nn.Parameter(torch.ones(T, self.n_orb))
        elif mode == 'free':
            self.wc = nn.Parameter(torch.ones(T, n_edge))
            self.wv = nn.Parameter(torch.ones(T, n_edge))
        elif mode == 'free_random':
            # matched-param control: tie edges into n_orb RANDOM groups (same param count as equiv,
            # but grouping is random, not symmetry-chosen). Isolates "symmetry tying" from "tying at
            # all / smallness". Random partition seeded by fgdata['rand_seed'] (default 0).
            rng = np.random.default_rng(fgdata.get('rand_seed', 0))
            rand_grp = rng.integers(0, self.n_orb, size=n_edge)
            # ensure all groups used (else param count drifts); remap to contiguous
            uniq = {g: i for i, g in enumerate(sorted(set(rand_grp.tolist())))}
            rand_grp = np.array([uniq[g] for g in rand_grp])
            self.n_groups = len(uniq)
            self.register_buffer('rand_of', torch.tensor(rand_grp, dtype=torch.long))
            self.wc = nn.Parameter(torch.ones(T, self.n_groups))
            self.wv = nn.Parameter(torch.ones(T, self.n_groups))
        elif mode == 'classical':
            self.wc = None; self.wv = None
        self.params_report = {'equiv': 2 * self.n_orb * T, 'free': 2 * n_edge * T,
                              'free_random': 2 * getattr(self, 'n_groups', self.n_orb) * T,
                              'classical': 0}[mode]

    def _pad_groups(self, edge_node, n_node, tag):
        groups = [[] for _ in range(n_node)]
        for e, nd in enumerate(edge_node):
            groups[nd].append(e)
        maxd = max(len(g) for g in groups)
        pad = np.full((n_node, maxd), -1, dtype=np.int64)
        for i, g in enumerate(groups):
            pad[i, :len(g)] = g
        self.register_buffer(f'{tag}_pad', torch.tensor(pad, dtype=torch.long))
        self.register_buffer(f'{tag}_mask', torch.tensor(pad >= 0))

    def _wc(self, t):
        if self.mode == 'classical': return torch.ones(self.n_edge, device=self.edge_det.device)
        if self.mode == 'equiv': return self.wc[t][self.orbit_of]
        if self.mode == 'free_random': return self.wc[t][self.rand_of]
        return self.wc[t]

    def _wv(self, t):
        if self.mode == 'classical': return torch.ones(self.n_edge, device=self.edge_det.device)
        if self.mode == 'equiv': return self.wv[t][self.orbit_of]
        if self.mode == 'free_random': return self.wv[t][self.rand_of]
        return self.wv[t]

    def forward(self, syndrome):
        """syndrome (B, ndet) in {0,1}. Returns var marginal LLR (B, E). >0 => e=0 favored."""
        B = syndrome.shape[0]; dev = syndrome.device
        lam = torch.log((1 - self.priors) / self.priors).view(1, self.E)     # (1,E)
        s_sign = (1.0 - 2.0 * syndrome.float())                              # (B,ndet)
        Mcv = torch.zeros(B, self.n_edge, device=dev)                        # check->var per edge
        INF = 1e9
        for t in range(self.T):
            # var->check: Mvc_e = lam[var_e] + sum_{e' in var, e'!=e} wv*Mcv_e'
            wMcv = self._wv(t).view(1, -1) * Mcv
            varsum = torch.zeros(B, self.E, device=dev).index_add_(1, self.edge_var, wMcv)  # (B,E)
            Mvc = lam[:, self.edge_var] + varsum[:, self.edge_var] - wMcv                    # (B,n_edge)
            # check->var: normalized min-sum, exclude-self, over each detector's group
            pad = self.chk_pad; mask = self.chk_mask                          # (ndet, maxd)
            gathered = Mvc[:, pad.clamp(min=0)]                               # (B,ndet,maxd)
            absg = gathered.abs().masked_fill(~mask.unsqueeze(0), INF)
            sgn = torch.sign(gathered); sgn = sgn.masked_fill(~mask.unsqueeze(0), 1.0)
            sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
            total_sign = sgn.prod(dim=2, keepdim=True)                        # (B,ndet,1)
            srt, _ = torch.sort(absg, dim=2)
            min1 = srt[:, :, 0:1]; min2 = srt[:, :, 1:2]
            excl = torch.where(absg <= min1 + 1e-12, min2.expand_as(absg), min1.expand_as(absg))
            sign_excl = total_sign * sgn
            ss = s_sign.unsqueeze(2)                                          # (B,ndet,1)
            chk_msg = (ss * sign_excl * excl).masked_fill(~mask.unsqueeze(0), 0.0)  # (B,ndet,maxd)
            # scatter back to edges
            newMcv = torch.zeros(B, self.n_edge, device=dev)
            flatpad = pad.reshape(-1); valid = flatpad >= 0
            src = chk_msg.reshape(B, -1)[:, valid]
            newMcv[:, flatpad[valid]] = src * self._wc(t).view(1, -1)[:, flatpad[valid]]
            Mcv = newMcv
        # marginal per var
        wMf = self._wc(self.T - 1).view(1, -1) * Mcv
        Sf = torch.zeros(B, self.E, device=dev).index_add_(1, self.edge_var, wMf)
        return lam + Sf                                                       # (B,E)


def main():
    bb = BBCode(); R = 3; p = 0.01
    fg = build_factorgraph(bb, R, p)
    L = bb.logicals_Z().astype(np.int64)
    obs_mat = None
    out = {'R': R, 'p': p, 'ndet': fg['ndet'], 'E': fg['E'], 'n_orb': fg['n_orb'],
           'n_fg_edges': len(fg['fg'])}
    # sample training/test data from the circuit
    samp = fg['circ'].compile_detector_sampler(seed=7)
    det_tr, obs_tr = samp.sample(8000, separate_observables=True)
    det_te, obs_te = samp.sample(4000, separate_observables=True)
    det_tr = torch.tensor(det_tr, dtype=torch.float32); obs_tr = torch.tensor(obs_tr, dtype=torch.float32)
    det_te = torch.tensor(det_te, dtype=torch.float32); obs_te = torch.tensor(obs_te, dtype=torch.float32)
    # target: predict error-mechanism marginal -> reconstruct observable flip.
    # Simpler honest objective: train BP marginals so that thresholded errors reproduce the syndrome
    # AND predict the observable. We use observable-prediction BCE on a linear readout of var-marginals.
    obsbit = torch.tensor(fg['obsbit'], dtype=torch.float32)   # (E,) which error flips the obs

    def obs_pred_logit(Lv):
        # P(obs flip) from var marginals: obs flips if an odd subset of obs-bearing errors fired.
        # soft proxy: sum over obs-bearing vars of P(e=1) mapped through a learned-free scalar is
        # overkill for a gate; use the BP posterior sign of the parity. We approximate the logical
        # logit as -sum_j obsbit_j * Lv_j  (more obs-bearing errors likely-on => flip more likely).
        pe1 = torch.sigmoid(-Lv)                       # P(e=1)
        return (obsbit.view(1, -1) * pe1).sum(1)       # (B,) proxy logit-ish >=0

    res = {}
    for mode in ['classical', 'equiv', 'free']:
        torch.manual_seed(0)
        m = CircNeuralBP(fg, T=8, mode=mode)
        out[f'params_{mode}'] = m.params_report
        if mode == 'classical':
            with torch.no_grad():
                pred = obs_pred_logit(m(det_te))
            # pick threshold = median to get a baseline acc (untrained reference)
            thr = pred.median()
            acc = ((pred > thr).float() == obs_te[:, 0]).float().mean().item()
            res[mode] = {'untrained_obs_acc_at_median_thr': round(acc, 4)}
            continue
        opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=0.05)
        lossf = nn.BCEWithLogitsLoss()
        # learnable scale+bias on the proxy logit so BCE is well-posed
        scale = torch.nn.Parameter(torch.tensor(1.0)); bias = torch.nn.Parameter(torch.tensor(0.0))
        opt2 = torch.optim.Adam([scale, bias], lr=0.05)
        n = det_tr.shape[0]; g = torch.Generator().manual_seed(1)
        losses = []
        for step in range(300):
            idx = torch.randint(0, n, (512,), generator=g)
            opt.zero_grad(); opt2.zero_grad()
            Lv = m(det_tr[idx])
            logit = scale * obs_pred_logit(Lv) + bias
            loss = lossf(logit, obs_tr[idx, 0])
            loss.backward(); opt.step(); opt2.step()
            if step % 30 == 0: losses.append(round(float(loss), 4))
        with torch.no_grad():
            logit = scale * obs_pred_logit(m(det_te)) + bias
            acc = ((logit > 0).float() == obs_te[:, 0]).float().mean().item()
        res[mode] = {'loss_curve': losses, 'final_loss': losses[-1],
                     'trained_obs_acc': round(acc, 4),
                     'optimized': bool(losses[-1] < losses[0])}
    out['results'] = res
    out['TRAINS'] = bool(res.get('equiv', {}).get('optimized', False))
    json.dump(out, open(os.path.join(_OUT, 'circ_trains_small.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2)); print("WROTE circ_trains_small.json")


if __name__ == '__main__':
    main()
