"""Bridge gate (staged, canonical harness). In-regime check BEFORE the full sweep.

R=6, p=0.01, Z-only canonical DEM, 3 arms: classical BP+OSD (tuned), equiv (~960p), free matched.
Paired seeds, N>=10k shots/arm. Decode = neural posterior -> ldpc BP(max_iter8)+OSD0 (fast path);
classical = ldpc BP(30)+OSD-cs10 on DEM priors (canonical tuned classical).

GATE CONDITIONS (both must hold to proceed to sweep):
  (1) equiv <= free at matched params (param-efficiency claim holds in-regime)
  (2) equiv within 1.5x of classical LER (sanity; if equiv >> classical, circuit-level scaling problem)
"""
import json, sys, time
import numpy as np
import torch, torch.nn as nn
from bb_code import BBCode
from bb_circuit import build_z_memory
from canon_dem import extract, decode_bposd, wilson
from circ_neural_bp import CircNeuralBP
from trains_canon import canon_fgdata, sample_dem, eval_ler
from ldpc import BpOsdDecoder

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
R, P = 6, 0.01


def build_free_matched(fgd_equiv, target_params, T=8):
    """free arm with ~target_params: free has 2*n_edge*T params; n_edge fixed by graph, so free
    param count is fixed (108864-ish at T=12 / proportional at T=8). We report it as-is and ALSO
    make a 'free_small' by reducing T to approximately match equiv params if needed. Simplest: the
    free arm at same T is the honest matched-architecture control (same graph, untied weights)."""
    return fgd_equiv  # same graph; mode='free' uses per-edge weights


def main():
    bb = BBCode()
    c = build_z_memory(bb, rounds=R, p=P)
    ex = extract(c.detector_error_model(decompose_errors=False))
    fgd = canon_fgdata(ex)
    T = 8
    seeds = [1, 2, 3]
    N_TR, N_TE = 40000, 10000
    out = {'stage': 'bridge', 'harness': 'canonical', 'R': R, 'p': P, 'device': str(DEV),
           'n_det': ex['n_det'], 'n_err': ex['n_err'], 'n_orb': fgd['n_orb'],
           'N_train': N_TR, 'N_test': N_TE, 'seeds': seeds, 'arms': {}}
    t0 = time.time()
    s_tr, e_tr, o_tr = sample_dem(ex, N_TR, 7)

    # --- classical tuned: ldpc BP30 + OSD-cs10 on DEM priors (no training) ---
    cl_lers = []
    for sd in seeds:
        s_te, e_te, o_te = sample_dem(ex, N_TE, 100 + sd)
        (ler, lo, hi), f, n = decode_bposd(ex, s_te.numpy().astype(np.uint8),
                                           o_te.numpy().astype(np.uint8), max_iter=30, osd_order=10)
        cl_lers.append(ler)
    out['arms']['classical'] = {'params': 0, 'ler_seeds': [round(x, 5) for x in cl_lers],
                                'ler_mean': round(float(np.mean(cl_lers)), 5)}
    print("classical mean LER", out['arms']['classical']['ler_mean'], f"({time.time()-t0:.0f}s)")

    # --- equiv and free (trained), same graph, paired seeds ---
    for mode in ['equiv', 'free']:
        lers = []; nparams = None
        for sd in seeds:
            torch.manual_seed(sd)
            m = CircNeuralBP(fgd, T=T, mode=mode).to(DEV); nparams = m.params_report
            opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=0.05)
            lf = nn.BCEWithLogitsLoss(); g = torch.Generator().manual_seed(sd)
            for step in range(2000):
                ib = torch.randint(0, N_TR, (1024,), generator=g)
                opt.zero_grad(); Lv = m(s_tr[ib].to(DEV)); lf(-Lv, e_tr[ib].to(DEV)).backward(); opt.step()
            s_te, e_te, o_te = sample_dem(ex, N_TE, 100 + sd)
            lers.append(eval_ler(m, ex, s_te, o_te, DEV)[0])
        a = np.array(lers)
        out['arms'][mode] = {'params': nparams, 'ler_seeds': [round(x, 5) for x in lers],
                             'ler_mean': round(float(a.mean()), 5), 'ler_std': round(float(a.std()), 5)}
        print(f"{mode} mean LER {a.mean():.5f} (params {nparams})")
        json.dump(out, open('bridge_canon.json', 'w'), indent=2)

    eq = out['arms']['equiv']['ler_mean']; fr = out['arms']['free']['ler_mean']; cl = out['arms']['classical']['ler_mean']
    out['gate_equiv_le_free'] = bool(eq <= fr)
    out['gate_equiv_within_1.5x_classical'] = bool(eq <= 1.5 * cl)
    out['BRIDGE_PASS'] = bool(out['gate_equiv_le_free'] and out['gate_equiv_within_1.5x_classical'])
    out['elapsed_min'] = round((time.time() - t0) / 60, 1)
    json.dump(out, open('bridge_canon.json', 'w'), indent=2)
    print("GATE equiv<=free:", out['gate_equiv_le_free'], "| equiv<=1.5x classical:", out['gate_equiv_within_1.5x_classical'])
    print("BRIDGE_PASS:", out['BRIDGE_PASS'], "| elapsed", out['elapsed_min'], "min")


if __name__ == '__main__':
    main()
