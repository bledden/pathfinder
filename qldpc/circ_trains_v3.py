"""Circuit-level training validation (BP-posterior + OSD readout, LER metric).

An earlier readout showed the architecture optimizes (loss drops, equiv>free) but raw-marginal RECALL was poor and
classical BP beat it — the known BP-without-OSD failure mode (trapping sets). The code-capacity result
that worked ALWAYS used OSD on top of BP. So v3 measures the real thing: feed each decoder's BP
posterior into OSD (project to a syndrome-consistent correction), compute LOGICAL ERROR RATE, and
compare equiv+OSD vs free+OSD vs classical+OSD at R=3. Gate is green if equiv+OSD trains to <= classical+OSD LER.

OSD via ldpc BpOsdDecoder over the DEM check matrix H (ndet x E), max_iter=0 (OSD only on the neural
prior, like the code-cap OSD-7 control), order 10. Neural posterior -> per-mechanism P(e=1) ->
update_channel_probs -> decode -> residual obs flip -> LER.
"""
import os
import json, sys
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
import torch, torch.nn as nn
import scipy.sparse as sp
from bb_code import BBCode
from circ_neural_bp import build_factorgraph, CircNeuralBP
from circ_trains_v2 import sample_from_dem
from ldpc import BpOsdDecoder
from _util import wilson_ci


def build_H(fg):
    """DEM check matrix H (ndet x E): H[d,j]=1 if detector d in mechanism j's detset."""
    ndet = fg['ndet']; detsets = fg['detsets']; E = len(detsets)
    rows, cols = [], []
    for j, Hset in enumerate(detsets):
        for d in Hset:
            rows.append(d); cols.append(j)
    return sp.csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(ndet, E))


def osd_ler(fg, H, posterior_pe1, synd, obs):
    """posterior_pe1 (N,E) P(e=1) per mechanism; OSD-project per shot; return LER (logical)."""
    obsbit = fg['obsbit'].astype(np.int64)
    Hd = H.toarray().astype(np.int64)
    N = synd.shape[0]
    dec = BpOsdDecoder(H.astype(np.uint8), error_channel=list(np.clip(posterior_pe1.mean(0), 1e-6, 1-1e-6)),
                       max_iter=0, bp_method='ms', osd_method='osd_cs', osd_order=10)
    fails = 0
    for i in range(N):
        dec.update_channel_probs(list(np.clip(posterior_pe1[i], 1e-6, 1 - 1e-6)))
        c = dec.decode(synd[i].astype(np.uint8)).astype(np.int64)
        if (int((c @ obsbit) % 2) != int(obs[i])):
            fails += 1
    return fails, N


def main():
    bb = BBCode(); R = 3; p = 0.01
    fg = build_factorgraph(bb, R, p)
    H = build_H(fg)
    out = {'R': R, 'p': p, 'ndet': fg['ndet'], 'E': fg['E'], 'n_orb': fg['n_orb'],
           'metric': 'logical error rate via BP-posterior -> OSD-cs order10 (max_iter0)'}
    s_tr, e_tr, o_tr = sample_from_dem(fg, 12000, 7)
    s_te, e_te, o_te = sample_from_dem(fg, 6000, 99)
    o_te_np = o_te.numpy().astype(np.int64); s_te_np = s_te.numpy().astype(np.uint8)

    def posterior(m):
        m.eval()
        with torch.no_grad():
            Lv = m(s_te)                       # (N,E) LLR >0 => e=0
        return (1.0 / (1.0 + np.exp(Lv.numpy()))).astype(np.float64)   # P(e=1)

    res = {}
    for mode in ['classical', 'equiv', 'free']:
        torch.manual_seed(0)
        m = CircNeuralBP(fg, T=8, mode=mode)
        out[f'params_{mode}'] = m.params_report
        if mode != 'classical':
            opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=0.05)
            lf = nn.BCEWithLogitsLoss(); g = torch.Generator().manual_seed(1); n = s_tr.shape[0]
            for step in range(400):
                idx = torch.randint(0, n, (512,), generator=g)
                opt.zero_grad(); Lv = m(s_tr[idx]); lf(-Lv, e_tr[idx]).backward(); opt.step()
        pe1 = posterior(m)
        fails, N = osd_ler(fg, H, pe1, s_te_np, o_te_np)
        ler, lo, hi = wilson_ci(fails, N)
        res[mode] = dict(ler=round(ler, 5), ci=[round(lo, 5), round(hi, 5)], fails=fails, n=N)
        print(f"{mode}+OSD10: LER {ler:.5f} [{lo:.5f},{hi:.5f}] ({fails}/{N})")
        json.dump({**out, 'results': res}, open(os.path.join(_OUT, 'circ_trains_v3.json'), 'w'), indent=2)
    out['results'] = res
    eq = res['equiv']['ler']; cl = res['classical']['ler']; fr = res['free']['ler']
    out['equiv_le_classical'] = bool(eq <= cl)
    out['equiv_le_free'] = bool(eq <= fr)
    out['GATE_GREEN'] = bool(eq <= cl)   # honest gate: trained equiv+OSD at least matches classical+OSD
    json.dump(out, open(os.path.join(_OUT, 'circ_trains_v3.json'), 'w'), indent=2)
    print("equiv<=classical:", out['equiv_le_classical'], "| equiv<=free:", out['equiv_le_free'])
    print("WROTE circ_trains_v3.json")


if __name__ == '__main__':
    main()
