"""Serious 'can a well-trained expressive neural decoder beat classical BP-OSD at circuit-level?'
attempt on the [[72,12,6]] BB code.

Unlike sweep_vec.py (a short 600-step symmetry-control GATE, OSD-0), this gives the expressive
vector-message neural-BP a real training budget (cosine-scheduled, thousands of steps, larger D)
and a FAIR, strong OSD, then pits it head-to-head against tuned classical BP-OSD at matched OSD
strength:

    classical = BP(ms, iter30) + OSD-CS order10                      (the established bar)
    neural    = trained neural-BP posterior -> OSD-CS order10        (max_iter=0: pure OSD on the
                                                                       per-shot neural channel)

Strict beat = neural Wilson CI strictly below classical Wilson CI (pooled over seeds).
Outcomes: strict-beat (new, reopens latency angle) / tie-within-CI (architecture-strength threshold
reached) / clear-loss (the earned architecture-strength negative). All honest.

Config overridable via BC_* env vars (for a fast CPU smoke test). Writes beat_classical.json.
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

_RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'beat_classical.json')
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

R = int(os.environ.get('BC_R', 6))
P_POINTS = [float(x) for x in os.environ.get('BC_PS', '0.005,0.01').split(',')]
SEEDS = [int(x) for x in os.environ.get('BC_SEEDS', '1,2,3').split(',')]
D = int(os.environ.get('BC_D', 24))
T = int(os.environ.get('BC_T', 12))
MODE = os.environ.get('BC_MODE', 'equiv')      # equiv == free_random in capacity-per-param; compact + fast
STEPS = int(os.environ.get('BC_STEPS', 5000))
BATCH = int(os.environ.get('BC_BATCH', 1024))
N_TRAIN = int(os.environ.get('BC_NTRAIN', 30000))
N_TEST = int(os.environ.get('BC_NTEST', 8000))
OSD_ORDER = int(os.environ.get('BC_OSD', 10))
LR0 = float(os.environ.get('BC_LR', 0.02))
LR1 = LR0 / 10.0


def cosine_lr(step, total):
    return LR1 + 0.5 * (LR0 - LR1) * (1 + math.cos(math.pi * step / max(total, 1)))


def _decode_loop(dec, s_np, o_np, Lo, per_shot_probs=None):
    f = 0
    N = s_np.shape[0]
    for i in range(N):
        if per_shot_probs is not None:
            dec.update_channel_probs(list(np.clip(per_shot_probs[i], 1e-8, 1 - 1e-8)))
        c = dec.decode(s_np[i]).astype(np.int64)
        if not np.array_equal(((Lo @ c) % 2).astype(np.int64), o_np[i]):
            f += 1
    return f, N


def eval_classical(ex, s_te, o_te, p):
    # CANONICAL classical bar: identical to canon_dem.decode_bposd — BP(ms, iter30) seeded with the
    # true DEM per-mechanism priors (NOT a uniform error_rate) + OSD-CS. Using a uniform channel here
    # would handicap classical and manufacture a false "beat".
    H = ex['H']; Lo = ex['Lo'].toarray().astype(np.int64)
    dec = BpOsdDecoder(H, error_channel=list(np.clip(ex['priors'], 1e-6, 1 - 1e-6)), max_iter=30,
                       bp_method='ms', osd_method='osd_cs', osd_order=OSD_ORDER)
    return _decode_loop(dec, s_te.numpy().astype(np.uint8), o_te.numpy().astype(np.int64), Lo)


def eval_neural(m, ex, s_te, o_te):
    m.eval()
    with torch.no_grad():
        Lv = m(s_te.to(DEV)).cpu().numpy()
    pe1 = 1.0 / (1.0 + np.exp(np.clip(Lv, -30, 30)))
    H = ex['H']; Lo = ex['Lo'].toarray().astype(np.int64)
    dec = BpOsdDecoder(H, error_channel=list(np.clip(pe1.mean(0), 1e-8, 1 - 1e-8)),
                       max_iter=0, osd_method='osd_cs', osd_order=OSD_ORDER)
    return _decode_loop(dec, s_te.numpy().astype(np.uint8), o_te.numpy().astype(np.int64), Lo,
                        per_shot_probs=pe1)


def train(fgd, ex, seed):
    torch.manual_seed(seed)
    m = CircNeuralBPVec(fgd, T=T, D=D, mode=MODE, rand_seed=seed).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR0)
    lf = nn.BCEWithLogitsLoss()
    s_tr, e_tr, _ = sample_dem(ex, N_TRAIN, 7000 + seed)
    s_tr = s_tr.to(DEV); e_tr = e_tr.to(DEV)
    g = torch.Generator().manual_seed(seed)
    for step in range(STEPS):
        for grp in opt.param_groups:
            grp['lr'] = cosine_lr(step, STEPS)
        idx = torch.randint(0, s_tr.shape[0], (BATCH,), generator=g).to(DEV)
        opt.zero_grad()
        lf(-m(s_tr[idx]), e_tr[idx]).backward()
        opt.step()
    return m


def main():
    bb = BBCode()
    out = {'desc': 'beat-classical: well-trained expressive neural-BP -> OSD-CS%d vs classical BP-OSD-CS%d'
           % (OSD_ORDER, OSD_ORDER), 'R': R, 'D': D, 'T': T, 'mode': MODE, 'steps': STEPS,
           'osd_order': OSD_ORDER, 'n_test': N_TEST, 'n_train': N_TRAIN, 'seeds': SEEDS,
           'device': str(DEV), 'points': {}}
    for p in P_POINTS:
        ex = extract(build_z_memory(bb, rounds=R, p=p).detector_error_model(decompose_errors=False))
        fgd = canon_fgdata(ex)
        cl_f = cl_n = nu_f = nu_n = 0
        per = []
        nparams = None
        for sd in SEEDS:
            s_te, e_te, o_te = sample_dem(ex, N_TEST, 90000 + sd)
            cf, cn = eval_classical(ex, s_te, o_te, p)
            m = train(fgd, ex, sd); nparams = m.params_report
            nf, nn_ = eval_neural(m, ex, s_te, o_te)
            cl_f += cf; cl_n += cn; nu_f += nf; nu_n += nn_
            per.append({'seed': sd, 'classical': round(cf / cn, 5), 'neural': round(nf / nn_, 5)})
            print(f"p={p} seed{sd}: classical {cf/cn:.5f}  neural {nf/nn_:.5f}  (params {nparams})", flush=True)
        clw = wilson(cl_f, cl_n); nuw = wilson(nu_f, nu_n)
        strict = bool(nuw[2] < clw[1])
        tie = bool(not strict and not (clw[2] < nuw[1]))   # CIs overlap => statistical tie
        out['points'][f'p{p}'] = {'neural_params': nparams,
                                  'classical_ler': round(clw[0], 5), 'classical_ci': [round(clw[1], 5), round(clw[2], 5)],
                                  'neural_ler': round(nuw[0], 5), 'neural_ci': [round(nuw[1], 5), round(nuw[2], 5)],
                                  'neural_strict_beats': strict, 'statistical_tie': tie,
                                  'n_pooled': cl_n, 'per_seed': per}
        json.dump(out, open(_RESULT, 'w'), indent=2)
        print(f"p={p}: classical {clw[0]:.5f} [{clw[1]:.5f},{clw[2]:.5f}] | "
              f"neural {nuw[0]:.5f} [{nuw[1]:.5f},{nuw[2]:.5f}] | strict_beat {strict} tie {tie}", flush=True)
    out['any_strict_beat'] = any(out['points'][f'p{p}']['neural_strict_beats'] for p in P_POINTS)
    out['any_tie_or_beat'] = any(out['points'][f'p{p}']['neural_strict_beats'] or
                                 out['points'][f'p{p}']['statistical_tie'] for p in P_POINTS)
    json.dump(out, open(_RESULT, 'w'), indent=2)
    print("ANY STRICT BEAT:", out['any_strict_beat'], "| ANY TIE-OR-BEAT:", out['any_tie_or_beat'])


if __name__ == '__main__':
    main()
