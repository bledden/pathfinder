"""Vector-message (expressive) neural BP for circuit-level BB decoding.

This is the *architecture-strength* arm of the qLDPC close-out ("B-vector"). The shipped
``CircNeuralBP`` learns only scalar damping multipliers on normalized min-sum — barely more
expressive than classical BP. Stage 1 showed ``equiv`` ties ``free_random`` there, but that cannot
distinguish "the symmetry prior is useless" from "the model class was too weak to host any prior."

This module generalizes the message to a D-dimensional learned vector with per-group (D x D)
transforms and a tanh nonlinearity on both the variable and check updates, plus a learned readout
head. The weight-tying control is preserved exactly:

    equiv        -> (D x D) transform tied across the symmetry orbits      (n_orb groups)
    free_random  -> tied across a random partition of the SAME size        (n_orb groups, matched)
    free         -> per-edge transform                                     (n_edge groups)

equiv and free_random therefore have *identical* parameter counts (2 * n_orb * T * D*D), so the
decisive question "does symmetry-chosen tying beat random tying within an expressive class?" is a
clean matched-parameter comparison.

STATUS: prototype. CPU-smoke-tested for output shape, matched param counts, and loss-decrease
(``python circ_neural_bp_vec.py``). The decisive run (equiv vs free_random at matched params,
R=6, p in {0.005, 0.01}, paired seeds) requires GPU and is pending the pod. Drop-in compatible with
``trains_canon.eval_ler`` / ``sample_dem`` (returns a per-mechanism LLR of shape (B, E), same sign
convention as ``CircNeuralBP``: >0 favors e=0).
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn


def groups_for_mode(fgd, mode, rand_seed=0):
    """Return (group_index_per_edge, n_groups) for the requested weight-tying mode."""
    n_edge = len(fgd['fg'])
    if mode == 'equiv':
        g = np.asarray(fgd['orbit_of'], dtype=np.int64)
        return g, int(g.max()) + 1
    if mode == 'free':
        return np.arange(n_edge, dtype=np.int64), n_edge
    if mode == 'free_random':
        n_orb = fgd['n_orb']
        rng = np.random.default_rng(rand_seed)
        g = rng.integers(0, n_orb, size=n_edge)
        uniq = {v: i for i, v in enumerate(sorted(set(g.tolist())))}
        return np.array([uniq[v] for v in g], dtype=np.int64), len(uniq)
    raise ValueError(f"unknown mode {mode!r} (use equiv / free / free_random)")


class CircNeuralBPVec(nn.Module):
    """Vector-message learned BP. mode in {equiv, free, free_random}; D = message width."""

    def __init__(self, fgd, T=8, D=4, mode='equiv', rand_seed=0):
        super().__init__()
        self.T = T
        self.D = D
        self.mode = mode
        self.ndet = fgd['ndet']
        self.E = fgd['E']
        fg = fgd['fg']
        self.n_edge = len(fg)
        edge_det = np.array([d for (d, j) in fg])
        edge_var = np.array([j for (d, j) in fg])
        self.register_buffer('edge_det', torch.tensor(edge_det, dtype=torch.long))
        self.register_buffer('edge_var', torch.tensor(edge_var, dtype=torch.long))
        self.register_buffer('priors', torch.tensor(fgd['priors'], dtype=torch.float32))
        grp, ng = groups_for_mode(fgd, mode, rand_seed)
        self.n_groups = ng
        self.register_buffer('grp', torch.tensor(grp, dtype=torch.long))
        # shared input embedding of the scalar prior LLR into D channels (symmetric; not a tied param)
        self.w_emb = nn.Parameter(torch.randn(D) * 0.1)
        # per-group (D x D) transforms, per iteration — the TIED params (the matched-control quantity)
        scale = 1.0 / float(np.sqrt(D))
        self.Wc = nn.Parameter(torch.randn(T, ng, D, D) * scale)
        self.Wv = nn.Parameter(torch.randn(T, ng, D, D) * scale)
        # shared readout head D -> 1
        self.head = nn.Parameter(torch.randn(D) * scale)
        self.head_b = nn.Parameter(torch.zeros(1))
        # number of orbit/random/edge-tied params (what we hold matched across equiv vs free_random)
        self.params_report = int(2 * ng * T * D * D)

    def forward(self, syndrome):
        """syndrome (B, ndet) in {0,1}. Returns per-mechanism LLR (B, E); >0 favors e=0."""
        B = syndrome.shape[0]
        dev = syndrome.device
        lam = torch.log((1 - self.priors) / self.priors)                 # (E,)
        Lam_e = lam[self.edge_var][:, None] * self.w_emb[None, :]         # (n_edge, D) prior per edge
        Lam_e = Lam_e.unsqueeze(0).expand(B, -1, -1)                      # (B, n_edge, D)
        s_sign = (1.0 - 2.0 * syndrome.float())                          # (B, ndet)
        Mcv = torch.zeros(B, self.n_edge, self.D, device=dev)            # check->var messages

        for t in range(self.T):
            Wv_e = self.Wv[t][self.grp]                                   # (n_edge, D, D)
            Wc_e = self.Wc[t][self.grp]
            # variable -> check: aggregate incoming check msgs at each var, exclude self, transform
            var_tot = torch.zeros(B, self.E, self.D, device=dev).index_add_(1, self.edge_var, Mcv)
            agg_v = var_tot[:, self.edge_var, :] - Mcv                    # (B, n_edge, D) exclude-self
            Mvc = torch.tanh(Lam_e + torch.einsum('bed,edk->bek', agg_v, Wv_e))
            # check -> variable: aggregate var msgs at each detector, exclude self, sign, transform
            det_tot = torch.zeros(B, self.ndet, self.D, device=dev).index_add_(1, self.edge_det, Mvc)
            agg_c = det_tot[:, self.edge_det, :] - Mvc                    # (B, n_edge, D)
            sgn = s_sign[:, self.edge_det][:, :, None]                    # (B, n_edge, 1)
            Mcv = torch.tanh(sgn * torch.einsum('bed,edk->bek', agg_c, Wc_e))

        # readout: per-variable belief = prior + aggregated final check msgs, projected to a scalar LLR
        var_tot = torch.zeros(B, self.E, self.D, device=dev).index_add_(1, self.edge_var, Mcv)
        belief = lam[None, :, None] * self.w_emb[None, None, :] + var_tot   # (B, E, D)
        return torch.einsum('bed,d->be', belief, self.head) + self.head_b   # (B, E)


def _synthetic_fgd(seed=0, ndet=6, E=8, n_orb=3):
    """Tiny hand-built factor-graph dict to smoke-test mechanics without stim/the real DEM."""
    rng = np.random.default_rng(seed)
    fg = []
    for j in range(E):
        deg = int(rng.integers(2, 4))
        for d in rng.choice(ndet, size=deg, replace=False):
            fg.append((int(d), j))
    orbit_of = rng.integers(0, n_orb, size=len(fg)).astype(np.int64)
    return dict(ndet=ndet, E=E, n_orb=n_orb, fg=fg, orbit_of=orbit_of,
                priors=(0.01 + 0.02 * rng.random(E)).astype(np.float32))


def _smoke():
    torch.manual_seed(0)
    fgd = _synthetic_fgd()
    B, D, T = 32, 4, 4
    rng = np.random.default_rng(1)
    synd = torch.tensor((rng.random((B, fgd['ndet'])) < 0.3).astype(np.float32))
    err = torch.tensor((rng.random((B, fgd['E'])) < fgd['priors'][None, :]).astype(np.float32))

    report = {}
    params = {}
    for mode in ['equiv', 'free_random', 'free']:
        m = CircNeuralBPVec(fgd, T=T, D=D, mode=mode, rand_seed=0)
        out = m(synd)
        assert out.shape == (B, fgd['E']), f"{mode}: bad output shape {out.shape}"
        params[mode] = m.params_report
        # one-arm training-signal check: a few Adam steps should reduce the BCE on -LLR vs error
        opt = torch.optim.Adam(m.parameters(), lr=0.05)
        lf = nn.BCEWithLogitsLoss()
        l0 = lf(-m(synd), err).item()
        for _ in range(50):
            opt.zero_grad(); lf(-m(synd), err).backward(); opt.step()
        l1 = lf(-m(synd), err).item()
        report[mode] = {'params': m.params_report, 'n_groups': m.n_groups,
                        'loss_start': round(l0, 4), 'loss_end': round(l1, 4),
                        'decreased': bool(l1 < l0)}

    checks = {
        'output_shape_ok': True,  # asserted above
        'equiv_eq_free_random_params': params['equiv'] == params['free_random'],
        'free_gt_equiv_params': params['free'] > params['equiv'],
        'all_arms_train': all(report[m_]['decreased'] for m_ in report),
    }
    print(json.dumps({'report': report, 'checks': checks}, indent=2))
    assert all(checks.values()), f"SMOKE FAILED: {checks}"
    print("\nSMOKE PASS: shapes ok, equiv/free_random param-matched, free larger, all arms train.")


if __name__ == '__main__':
    _smoke()
