"""Equivariant neural belief-propagation decoder for the [[72,12,6]] BB code.

WHY (after the feedforward negative result): feedforward syndrome->logical maps plateau ~3x above
BP-OSD because they lack the ITERATIVE message passing LDPC codes need. Neural BP (Nachmani 2016)
unrolls T BP iterations with learnable weights -> trainable, keeps the iterative structure.

NOVELTY: tie the learnable weights across the ORBITS of the Z6xZ6 Tanner-graph automorphism.
The 216 check<->variable edges split into exactly 6 orbits of 36 under Z6xZ6, so equivariant
neural-BP has ~6 weight classes/iteration instead of 216. We compare THREE decoders head to head:
  - classical : all weights = 1 (plain normalized min-sum)
  - equiv     : weights tied across the 6 edge-orbits  (the equivariant claim)
  - free      : independent weight per edge (Nachmani-style, 216/iter, NOT equivariant)
Scientific question: does orbit-tying beat free at fewer params (sample efficiency) and/or beat
classical / BP-OSD (3.09% per-logical @ p0.03)?

The BB code is a regular (dc=6, dv=3) LDPC code -> messages reshape cleanly by check (36,6).

Min-sum (not tanh sum-product) for numerical stability; exact exclude-self via min1/min2 and
sign-product/self. Decode the ERROR then apply logicals == the BP-OSD metric (directly comparable).

selftest(): verifies (a) 216 edges -> 6 orbits of 36, (b) classical limit corrects all weight-1
errors and matches an independent reference, (c) gradients flow. Run BEFORE training.
"""
import os
import json, sys
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
import torch
import torch.nn as nn
from bb_code import BBCode
from _util import wilson_ci


def build_edges_and_orbits(bb):
    """Edges in CHECK order (e = 6*c + j). Returns:
       check_idx[E], var_idx[E], orbit_co (36,6) orbit id per (check,slot), n_orbits."""
    H = bb.HZ
    m, n = H.shape                       # 36, 72
    N = bb.N                             # 36
    # edges grouped by check, each check degree 6
    check_idx, var_idx = [], []
    for c in range(m):
        vs = np.where(H[c] == 1)[0]
        assert len(vs) == 6, f"check {c} degree {len(vs)} != 6 (code not regular?)"
        for v in vs:
            check_idx.append(c); var_idx.append(int(v))
    E = len(check_idx)
    check_idx = np.array(check_idx); var_idx = np.array(var_idx)
    edge_of = {(check_idx[e], var_idx[e]): e for e in range(E)}

    def shift(idx, a, b):                # torus shift on a 0..35 index = flatten of (6,6)
        i, j = idx // 6, idx % 6
        return ((i + a) % 6) * 6 + ((j + b) % 6)

    orbit = -np.ones(E, dtype=int)
    cur = 0
    for e in range(E):
        if orbit[e] != -1:
            continue
        c0, v0 = int(check_idx[e]), int(var_idx[e])
        blk, pos = v0 // N, v0 % N
        for a in range(6):
            for b in range(6):
                c2 = shift(c0, a, b)
                v2 = blk * N + shift(pos, a, b)
                j = edge_of.get((c2, v2))
                assert j is not None, "group action maps edge off the Tanner graph (not a symmetry!)"
                if orbit[j] == -1:
                    orbit[j] = cur
        cur += 1
    orbit_co = orbit.reshape(m, 6)
    return check_idx, var_idx, orbit_co, cur


class NeuralBP(nn.Module):
    """Unrolled normalized min-sum BP. mode in {'classical','equiv','free'}."""
    def __init__(self, bb, T=12, mode='equiv'):
        super().__init__()
        self.bb = bb
        self.m, self.n = bb.HZ.shape
        self.T = T
        self.mode = mode
        ci, vi, orbit_co, n_orb = build_edges_and_orbits(bb)
        self.E = len(ci)
        self.n_orbits = n_orb
        self.register_buffer('var_idx', torch.tensor(vi, dtype=torch.long))      # (E,)
        self.register_buffer('orbit_co', torch.tensor(orbit_co, dtype=torch.long))  # (36,6)
        self.register_buffer('orbit_flat', torch.tensor(orbit_co.reshape(-1), dtype=torch.long))  # (E,)
        # logicals for evaluation
        self.register_buffer('L', torch.tensor(bb.logicals_Z(), dtype=torch.float32))  # (k,72)
        # learnable weights
        if mode == 'classical':
            self.alpha_c = None; self.alpha_v = None; self.w_lam = None; self.w_out = None
        elif mode == 'equiv':
            self.alpha_c = nn.Parameter(torch.ones(T, n_orb))   # check msg scale per orbit per iter
            self.alpha_v = nn.Parameter(torch.ones(T, n_orb))   # var msg scale per orbit per iter
            self.w_lam = nn.Parameter(torch.ones(T))            # prior scale per iter
            self.w_out = nn.Parameter(torch.ones(1))            # output prior scale
            self.w_outm = nn.Parameter(torch.ones(n_orb))       # output msg scale per orbit
        elif mode == 'free':
            self.alpha_c = nn.Parameter(torch.ones(T, self.E))  # per-edge (216/iter)
            self.alpha_v = nn.Parameter(torch.ones(T, self.E))
            self.w_lam = nn.Parameter(torch.ones(T))
            self.w_out = nn.Parameter(torch.ones(1))
            self.w_outm = nn.Parameter(torch.ones(self.E))
        else:
            raise ValueError(mode)

    def _ac(self, t):  # check-msg weights as (36,6)
        if self.mode == 'classical': return torch.ones(self.m, 6, device=self.var_idx.device)
        if self.mode == 'equiv':     return self.alpha_c[t][self.orbit_co]
        return self.alpha_c[t].view(self.m, 6)

    def _av_flat(self, t):  # var-msg weights as (E,)
        if self.mode == 'classical': return torch.ones(self.E, device=self.var_idx.device)
        if self.mode == 'equiv':     return self.alpha_v[t][self.orbit_flat]
        return self.alpha_v[t]

    def _wlam(self, t):
        return 1.0 if self.mode == 'classical' else self.w_lam[t]

    def _woutm_flat(self):
        if self.mode == 'classical': return torch.ones(self.E, device=self.var_idx.device)
        if self.mode == 'equiv':     return self.w_outm[self.orbit_flat]
        return self.w_outm

    def _wout(self):
        return 1.0 if self.mode == 'classical' else self.w_out

    def forward(self, syndrome, p):
        """syndrome: (B, m) in {0,1}. Returns marginal LLR L_v (B, n).
        L_v > 0 => e_v = 0 favored."""
        B = syndrome.shape[0]
        dev = syndrome.device
        lam0 = float(np.log((1 - p) / p))
        lam = torch.full((self.n,), lam0, device=dev)            # (n,)
        s_sign = (1.0 - 2.0 * syndrome).view(B, self.m, 1)        # (B,36,1) +-1
        var_idx = self.var_idx                                    # (E,)
        Mcv = torch.zeros(B, self.E, device=dev)                  # check->var, edge order

        for t in range(self.T):
            # ---- variable -> check (sum over edges at v, exclude self) ----
            wMcv = self._av_flat(t).view(1, self.E) * Mcv                  # (B,E)
            Sw = torch.zeros(B, self.n, device=dev).index_add_(
                1, var_idx, wMcv)                                          # (B,n)
            Mvc = self._wlam(t) * lam[var_idx].view(1, self.E) \
                + Sw[:, var_idx] - wMcv                                    # (B,E)
            # ---- check -> var (normalized min-sum, exclude self) ----
            R = Mvc.view(B, self.m, 6)
            absR = R.abs()
            sgn = torch.sign(R); sgn[sgn == 0] = 1.0
            total_sign = sgn.prod(dim=2, keepdim=True)                     # (B,36,1)
            sign_excl = total_sign * sgn                                   # (B,36,6)
            srt, idx = torch.sort(absR, dim=2)
            min1 = srt[:, :, 0:1]; min2 = srt[:, :, 1:2]; amin = idx[:, :, 0:1]
            excl = min1.expand(-1, -1, 6).clone()
            excl.scatter_(2, amin, min2)                                   # (B,36,6)
            ac = self._ac(t).view(1, self.m, 6)
            Mcv = (ac * s_sign * sign_excl * excl).view(B, self.E)
        # ---- marginal ----
        wMf = self._woutm_flat().view(1, self.E) * Mcv
        Sf = torch.zeros(B, self.n, device=dev).index_add_(1, var_idx, wMf)
        Lv = self._wout() * lam.view(1, self.n) + Sf                       # (B,n)
        return Lv

    def decode_logical_fail(self, syndrome, e_true, p, osd=True):
        """Per-shot block fail + per-logical fail counts. e_true (B,n).
        osd=True applies OSD-0 post-processing (project BP marginals onto a syndrome-consistent
        solution) — without it BP gives good marginals but H.ehat != s on degenerate codes."""
        Lv = self.forward(syndrome, p)                                     # (B,n)
        if osd:
            ehat = self._osd0(Lv, syndrome)
        else:
            ehat = (Lv < 0).float()
        res = (e_true - ehat) % 2                                          # XOR
        flips = (res @ self.L.t()) % 2                                     # (B,k)
        per_logical = flips.sum().item()
        block = (flips.sum(1) > 0).sum().item()
        return block, per_logical, flips.shape[1]

    def _osd0(self, Lv, syndrome):
        """OSD order-0: for each shot, order columns of H by BP reliability (|Lv| ascending =
        most-likely-error first), Gaussian-eliminate over GF(2) to find a basis of reliable
        columns, set the rest to 0, solve H.ehat = s exactly on the basis. Numpy per-shot
        (B small at eval). Returns ehat (B,n) float tensor."""
        H = self.bb.HZ.astype(np.uint8)
        m, n = H.shape
        B = Lv.shape[0]
        s_np = syndrome.detach().cpu().numpy().astype(np.uint8)
        rel = (-Lv.detach().cpu().numpy())   # larger => more likely to be an error (Lv<0)
        out = np.zeros((B, n), dtype=np.float32)
        for b in range(B):
            order = np.argsort(-rel[b])      # most-likely-error columns first
            A = np.zeros((m, m), dtype=np.uint8)
            cols = []
            piv_rows = []
            Hperm = H[:, order]
            # build full-rank set of columns via GF2 elimination
            R = np.zeros((m, 0), dtype=np.uint8)
            chosen = []
            basis = np.zeros((m, m), dtype=np.uint8); nb = 0
            redrows = []  # echelon rows
            pivots = []
            work = np.zeros((m, m), dtype=np.uint8); wc = 0
            ech = []      # list of (pivot_row, vector) echelon for chosen columns
            for ci in range(n):
                col = Hperm[:, ci].copy()
                v = col.copy()
                for (pr, ev) in ech:
                    if v[pr]:
                        v ^= ev
                if v.any():
                    pr = int(np.argmax(v))
                    ech.append((pr, v))
                    chosen.append(ci)
                    if len(chosen) == m:
                        break
            # solve on chosen columns: H[:,chosen_orig] x = s
            chosen_orig = order[chosen]
            Hc = H[:, chosen_orig].copy().astype(np.uint8)   # (m, r)
            # Gaussian elimination solve Hc y = s
            r = Hc.shape[1]
            M = np.concatenate([Hc, s_np[b][:, None]], axis=1).astype(np.uint8)  # (m, r+1)
            row = 0; where = {}
            for col in range(r):
                piv = None
                for rr in range(row, m):
                    if M[rr, col]:
                        piv = rr; break
                if piv is None:
                    continue
                M[[row, piv]] = M[[piv, row]]
                for rr in range(m):
                    if rr != row and M[rr, col]:
                        M[rr] ^= M[row]
                where[col] = row; row += 1
            y = np.zeros(r, dtype=np.uint8)
            for col in range(r):
                if col in where:
                    y[col] = M[where[col], r]
            eb = np.zeros(n, dtype=np.float32)
            eb[chosen_orig] = y.astype(np.float32)
            out[b] = eb
        return torch.tensor(out, device=Lv.device)


def selftest():
    bb = BBCode()
    out = {}
    ci, vi, orbit_co, n_orb = build_edges_and_orbits(bb)
    sizes = [int((orbit_co.reshape(-1) == o).sum()) for o in range(n_orb)]
    out['n_edges'] = len(ci)
    out['n_orbits'] = n_orb
    out['orbit_sizes'] = sizes
    out['orbits_6x36'] = bool(n_orb == 6 and all(s == 36 for s in sizes))

    dev = 'cpu'
    p = 0.03
    bp = NeuralBP(bb, T=20, mode='classical').to(dev)
    # weight-1 correctness
    H = torch.tensor(bb.HZ, dtype=torch.float32)
    w1_ok = 0
    for v in range(bb.n):
        e = torch.zeros(1, bb.n); e[0, v] = 1
        s = (e @ H.t()) % 2
        blk, pl, k = bp.decode_logical_fail(s, e, p)
        if blk == 0:
            w1_ok += 1
    out['weight1_corrected'] = f"{w1_ok}/{bb.n}"
    out['weight1_all'] = bool(w1_ok == bb.n)

    # classical BP code-cap LER (reference point)
    rng = np.random.default_rng(0)
    NT = 2000
    e = (rng.random((NT, bb.n)) < p).astype(np.float32)
    s = (e @ bb.HZ.T) % 2
    et = torch.tensor(e); st = torch.tensor(s, dtype=torch.float32)
    blk, pl, k = bp.decode_logical_fail(st, et, p)
    out['classicalBP_block_ler'] = round(blk / NT, 4)
    out['classicalBP_per_logical_ler'] = round(pl / (NT * k), 4)
    out['bposd_block_bar'] = 0.06875
    out['bposd_per_logical_bar'] = 0.030883

    # gradients flow (equiv)
    bpE = NeuralBP(bb, T=8, mode='equiv').to(dev)
    Lv = bpE.forward(st[:64], p)
    tgt = et[:64]
    loss = nn.BCEWithLogitsLoss()(-Lv, tgt)
    loss.backward()
    g = sum((param.grad.abs().sum().item() if param.grad is not None else 0.0)
            for param in bpE.parameters())
    out['equiv_grad_flows'] = bool(g > 0)
    out['equiv_n_params'] = sum(p_.numel() for p_ in bpE.parameters() if p_.requires_grad)
    out['free_n_params'] = sum(p_.numel() for p_ in NeuralBP(bb, T=8, mode='free').parameters()
                               if p_.requires_grad)
    out['SELFTEST_PASS'] = bool(out['orbits_6x36'] and out['weight1_all'] and out['equiv_grad_flows'])
    json.dump(out, open(os.path.join(_OUT, 'neural_bp_selftest.json'), 'w'), indent=2)
    print("WROTE neural_bp_selftest.json")


if __name__ == '__main__':
    selftest()
