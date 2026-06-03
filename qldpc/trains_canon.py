"""Circuit-level training validation on the canonical harness. Rebuilds CircNeuralBP fgdata from the
canonical DEM extraction (NOT ad-hoc sample_from_dem), trains equiv/free, decodes via canonical
BP-posterior -> OSD, reports LER paired by seed. Canonical-harness training validation.

R=3 for the small gate. Train target = per-mechanism BCE vs ground-truth error (sampled from DEM
priors -> deterministic detectors+observable). Eval LER = neural posterior -> ldpc BP(max_iter8)+OSD0
on canonical H, observable via canonical Lo.
"""
import os
import json, sys
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
import torch, torch.nn as nn
from bb_code import BBCode
from bb_circuit import build_z_memory
from canon_dem import extract, wilson
from circ_neural_bp import CircNeuralBP
from ldpc import BpOsdDecoder
from stepA_canon import UF

N = 36


def _rfromdem(ex):
    """Infer rounds R from detector count: ndet = R*N + N (R bulk rounds + final block)."""
    R = ex['n_det'] // N - 1
    assert R * N + N == ex['n_det'], f"ndet {ex['n_det']} not R*N+N"
    return R


def _spatperm(R, a, b):
    from stepA_orbits import spatial_perm
    sp = spatial_perm(a, b); P = np.empty(R * N + N, dtype=int)
    for d in range(R * N + N):
        r, q = ((R, d - R * N) if d >= R * N else (d // N, d % N))
        P[d] = (R * N + int(sp[q])) if r == R else (r * N + int(sp[q]))
    return P


def _tempperm(R, shift):
    P = -np.ones(R * N + N, dtype=int)
    for d in range(R * N + N):
        r, q = ((R, d - R * N) if d >= R * N else (d // N, d % N))
        if r == R: continue
        r2 = r + shift
        if 0 <= r2 < R: P[d] = r2 * N + q
    return P


def canon_fgdata(ex):
    """Build CircNeuralBP fgdata from canonical extraction + orbit union-find (spatial+temporal).
    R is inferred from the DEM so this works at any rounds (bridge R=6, sweep R in {4,6,8})."""
    R = _rfromdem(ex)
    edges = ex['edges']                          # list of (frozenset dets, obs tuple)
    detsets = [Hs for (Hs, o) in edges]
    E = len(detsets)
    # fg edges (detector, var index)
    fg = [(d, j) for j, Hs in enumerate(detsets) for d in Hs]
    dset = set(detsets); var_of = {Hs: j for j, Hs in enumerate(detsets)}
    gens = [_spatperm(R, a, b) for a in range(6) for b in range(6)] + [_tempperm(R, 1), _tempperm(R, -1)]
    uf = UF()
    for (d, j) in fg:
        Hs = detsets[j]
        for P in gens:
            if any(P[x] < 0 for x in Hs): continue
            img = frozenset(int(P[x]) for x in Hs)
            if img in dset and P[d] >= 0:
                uf.union((d, j), (int(P[d]), var_of[img]))
    roots = {}; orbit_of = []
    for (d, j) in fg:
        r = uf.find((d, j))
        if r not in roots: roots[r] = len(roots)
        orbit_of.append(roots[r])
    return dict(ndet=ex['n_det'], E=E, n_orb=len(roots), fg=fg,
                orbit_of=np.array(orbit_of), priors=ex['priors'].astype(np.float32),
                detsets=detsets, obsbit=np.array([(o[0] if o else 0) for (_, o) in edges]))


def sample_dem(ex, n, seed):
    rng = np.random.default_rng(seed)
    E = ex['n_err']; nd = ex['n_det']; no = ex['n_obs']
    pri = ex['priors']
    err = (rng.random((n, E)) < pri[None, :]).astype(np.int8)
    H = ex['H']; Lo = ex['Lo']
    synd = (err @ H.T.toarray().astype(np.int64)) % 2
    obs = (err @ Lo.T.toarray().astype(np.int64)) % 2
    return (torch.tensor(synd, dtype=torch.float32), torch.tensor(err, dtype=torch.float32),
            torch.tensor(obs, dtype=torch.float32))


def eval_ler(m, ex, s_te, o_te, dev):
    m.eval()
    with torch.no_grad():
        Lv = m(s_te.to(dev)).cpu().numpy()
    pe1 = 1.0 / (1.0 + np.exp(np.clip(Lv, -30, 30)))
    H = ex['H']; Lo = ex['Lo'].toarray().astype(np.int64)
    dec = BpOsdDecoder(H, error_channel=list(np.clip(pe1.mean(0), 1e-6, 1-1e-6)),
                       max_iter=8, bp_method='ms', osd_method='osd0', osd_order=0)
    s_np = s_te.numpy().astype(np.uint8); o_np = o_te.numpy().astype(np.int64)
    N = s_np.shape[0]; f = 0
    for i in range(N):
        dec.update_channel_probs(list(np.clip(pe1[i], 1e-6, 1-1e-6)))
        c = dec.decode(s_np[i]).astype(np.int64)
        if not np.array_equal(((Lo @ c) % 2).astype(np.int64), o_np[i]):
            f += 1
    return wilson(f, N)


def main():
    bb = BBCode(); p = 0.01
    c = build_z_memory(bb, rounds=R, p=p)
    ex = extract(c.detector_error_model(decompose_errors=False))
    fgd = canon_fgdata(ex)
    dev = torch.device('cpu')
    out = {'harness': 'canonical', 'R': R, 'p': p, 'n_det': ex['n_det'], 'n_err': ex['n_err'],
           'n_orb': fgd['n_orb'], 'seeds': [1, 2, 3], 'results': {}}
    s_tr, e_tr, o_tr = sample_dem(ex, 12000, 7)
    # classical reference (no training)
    cl = CircNeuralBP(fgd, T=8, mode='classical')
    cl_lers = []
    for sd in [1, 2, 3]:
        s_te, e_te, o_te = sample_dem(ex, 4000, 100 + sd)
        cl_lers.append(eval_ler(cl, ex, s_te, o_te, dev)[0])
    out['results']['classical'] = {'params': 0, 'ler_seeds': [round(x, 5) for x in cl_lers],
                                   'ler_mean': round(float(np.mean(cl_lers)), 5)}
    for mode in ['equiv', 'free']:
        lers = []; nparams = None
        for sd in [1, 2, 3]:
            torch.manual_seed(sd)
            m = CircNeuralBP(fgd, T=8, mode=mode).to(dev); nparams = m.params_report
            opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=0.05)
            lf = nn.BCEWithLogitsLoss(); g = torch.Generator().manual_seed(sd)
            for step in range(400):
                idxb = torch.randint(0, s_tr.shape[0], (512,), generator=g)
                opt.zero_grad(); Lv = m(s_tr[idxb].to(dev)); lf(-Lv, e_tr[idxb].to(dev)).backward(); opt.step()
            s_te, e_te, o_te = sample_dem(ex, 4000, 100 + sd)
            lers.append(eval_ler(m, ex, s_te, o_te, dev)[0])
        a = np.array(lers)
        out['results'][mode] = {'params': nparams, 'ler_seeds': [round(x, 5) for x in lers],
                                'ler_mean': round(float(a.mean()), 5), 'ler_std': round(float(a.std()), 5)}
    eq = out['results']['equiv']['ler_mean']; cl_m = out['results']['classical']['ler_mean']
    fr = out['results']['free']['ler_mean']
    out['equiv_beats_classical'] = bool(eq < cl_m)
    out['equiv_beats_free'] = bool(eq <= fr)
    out['equiv_params'] = out['results']['equiv']['params']
    out['free_params'] = out['results']['free']['params']
    out['GATE'] = bool(eq < cl_m and eq <= fr)
    json.dump(out, open(os.path.join(_OUT, 'trains_canon.json'), 'w'), indent=2)
    print(json.dumps(out['results'], indent=2)); print("GATE", out['GATE'])


if __name__ == '__main__':
    main()
