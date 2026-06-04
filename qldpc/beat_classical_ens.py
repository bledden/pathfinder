"""Bigger beat-classical attempt: multi-seed ENSEMBLE (logit-averaged neural-BP, the PFWL3S trick)
with more capacity, at p in {0.005, 0.01}, vs the canonical classical bar.

Per p:  classical = canon BP(ms,iter30, DEM priors) + OSD-CS    (identical to canon_dem.decode_bposd)
        per-seed neural = trained neural-BP posterior -> OSD-CS
        ENSEMBLE = logit-average the K seeds' posteriors -> OSD-CS   (headline)
strict beat = ensemble Wilson CI strictly below classical Wilson CI.

Low-p gets more shots (rarer failures need more samples for CI resolution). Config via BC_* env.
Writes beat_classical_ens.json.
"""
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from bb_code import BBCode
from bb_circuit import build_z_memory
from canon_dem import extract, wilson
from trains_canon import canon_fgdata, sample_dem
from circ_neural_bp_vec import CircNeuralBPVec
from ldpc import BpOsdDecoder

_RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'beat_classical_ens.json')
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

R = int(os.environ.get('BC_R', 6))
P_POINTS = [float(x) for x in os.environ.get('BC_PS', '0.005,0.01').split(',')]
SEEDS = [int(x) for x in os.environ.get('BC_SEEDS', '1,2,3').split(',')]
D = int(os.environ.get('BC_D', 32))
T = int(os.environ.get('BC_T', 12))
MODE = os.environ.get('BC_MODE', 'equiv')
STEPS = int(os.environ.get('BC_STEPS', 6000))
BATCH = int(os.environ.get('BC_BATCH', 1024))
N_TRAIN = int(os.environ.get('BC_NTRAIN', 40000))
NTEST_LOW = int(os.environ.get('BC_NTEST_LOW', 15000))   # p<=0.005
NTEST_HI = int(os.environ.get('BC_NTEST_HI', 10000))     # p>0.005
OSD_ORDER = int(os.environ.get('BC_OSD', 7))
LR0 = float(os.environ.get('BC_LR', 0.02)); LR1 = LR0 / 10.0


def cosine_lr(step, total):
    return LR1 + 0.5 * (LR0 - LR1) * (1 + math.cos(math.pi * step / max(total, 1)))


def _loop(dec, s_np, o_np, Lo, probs=None):
    f = 0; N = s_np.shape[0]
    for i in range(N):
        if probs is not None:
            dec.update_channel_probs(list(np.clip(probs[i], 1e-8, 1 - 1e-8)))
        c = dec.decode(s_np[i]).astype(np.int64)
        if not np.array_equal(((Lo @ c) % 2).astype(np.int64), o_np[i]):
            f += 1
    return f, N


def classical_fails(ex, s_te, o_te):
    H = ex['H']; Lo = ex['Lo'].toarray().astype(np.int64)
    dec = BpOsdDecoder(H, error_channel=list(np.clip(ex['priors'], 1e-6, 1 - 1e-6)), max_iter=30,
                       bp_method='ms', osd_method='osd_cs', osd_order=OSD_ORDER)
    return _loop(dec, s_te.numpy().astype(np.uint8), o_te.numpy().astype(np.int64), Lo)


def neural_fails(pe1, ex, s_te, o_te):
    H = ex['H']; Lo = ex['Lo'].toarray().astype(np.int64)
    dec = BpOsdDecoder(H, error_channel=list(np.clip(pe1.mean(0), 1e-8, 1 - 1e-8)),
                       max_iter=0, osd_method='osd_cs', osd_order=OSD_ORDER)
    return _loop(dec, s_te.numpy().astype(np.uint8), o_te.numpy().astype(np.int64), Lo, probs=pe1)


def train(fgd, ex, seed):
    torch.manual_seed(seed)
    m = CircNeuralBPVec(fgd, T=T, D=D, mode=MODE, rand_seed=seed).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR0); lf = nn.BCEWithLogitsLoss()
    s_tr, e_tr, _ = sample_dem(ex, N_TRAIN, 7000 + seed); s_tr = s_tr.to(DEV); e_tr = e_tr.to(DEV)
    g = torch.Generator().manual_seed(seed)
    for step in range(STEPS):
        for grp in opt.param_groups:
            grp['lr'] = cosine_lr(step, STEPS)
        idx = torch.randint(0, s_tr.shape[0], (BATCH,), generator=g).to(DEV)
        opt.zero_grad(); lf(-m(s_tr[idx]), e_tr[idx]).backward(); opt.step()
    return m


def llr(m, s_te):
    m.eval()
    with torch.no_grad():
        return m(s_te.to(DEV)).cpu().numpy()


def main():
    bb = BBCode()
    out = {'desc': 'beat-classical ENSEMBLE: %d-seed logit-avg neural-BP (D=%d,T=%d,%d steps) -> OSD-CS%d '
           'vs canonical classical' % (len(SEEDS), D, T, STEPS, OSD_ORDER),
           'R': R, 'D': D, 'T': T, 'mode': MODE, 'steps': STEPS, 'osd_order': OSD_ORDER,
           'n_train': N_TRAIN, 'seeds': SEEDS, 'device': str(DEV), 'points': {}}
    for p in P_POINTS:
        ex = extract(build_z_memory(bb, rounds=R, p=p).detector_error_model(decompose_errors=False))
        fgd = canon_fgdata(ex)
        Nt = NTEST_LOW if p <= 0.005 else NTEST_HI
        s_te, e_te, o_te = sample_dem(ex, Nt, 90001)
        cf, cn = classical_fails(ex, s_te, o_te)
        print(f"p={p}: classical {cf}/{cn} = {cf/cn:.5f}", flush=True)
        Lvs = []; per_seed = []; nparams = None
        for sd in SEEDS:
            m = train(fgd, ex, sd); nparams = m.params_report
            Lv = llr(m, s_te); Lvs.append(Lv)
            pe1 = 1.0 / (1.0 + np.exp(np.clip(Lv, -30, 30)))
            nf, nn_ = neural_fails(pe1, ex, s_te, o_te)
            per_seed.append(round(nf / nn_, 5))
            print(f"  seed{sd}: {nf}/{nn_} = {nf/nn_:.5f}", flush=True)
        Lv_ens = np.mean(Lvs, axis=0)
        pe1_ens = 1.0 / (1.0 + np.exp(np.clip(Lv_ens, -30, 30)))
        ef, en = neural_fails(pe1_ens, ex, s_te, o_te)
        clw = wilson(cf, cn); enw = wilson(ef, en)
        strict = bool(enw[2] < clw[1]); tie = bool(not strict and not (clw[2] < enw[1]))
        out['points'][f'p{p}'] = {'n_test': Nt, 'neural_params': nparams,
                                  'classical_ler': round(clw[0], 5), 'classical_ci': [round(clw[1], 5), round(clw[2], 5)],
                                  'ensemble_ler': round(enw[0], 5), 'ensemble_ci': [round(enw[1], 5), round(enw[2], 5)],
                                  'per_seed_ler': per_seed, 'indiv_mean': round(float(np.mean(per_seed)), 5),
                                  'ensemble_strict_beats': strict, 'statistical_tie': tie}
        json.dump(out, open(_RESULT, 'w'), indent=2)
        print(f"p={p}: classical {clw[0]:.5f} [{clw[1]:.5f},{clw[2]:.5f}] | ENSEMBLE {enw[0]:.5f} "
              f"[{enw[1]:.5f},{enw[2]:.5f}] | indiv_mean {np.mean(per_seed):.5f} | strict_beat {strict} tie {tie}", flush=True)
    out['any_ensemble_strict_beat'] = any(out['points'][f'p{p}']['ensemble_strict_beats'] for p in P_POINTS)
    out['any_tie_or_beat'] = any(out['points'][f'p{p}']['ensemble_strict_beats'] or
                                 out['points'][f'p{p}']['statistical_tie'] for p in P_POINTS)
    json.dump(out, open(_RESULT, 'w'), indent=2)
    print("ANY ENSEMBLE STRICT BEAT:", out['any_ensemble_strict_beat'],
          "| ANY TIE-OR-BEAT:", out['any_tie_or_beat'])


if __name__ == '__main__':
    main()
