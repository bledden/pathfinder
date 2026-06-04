"""Architecture-strength Stage 1 (vector-message arms) — the qLDPC close-out "B-vector" experiment.

Re-runs the decisive matched-parameter control from sweep.py::stage1, but with the EXPRESSIVE
vector-message model (CircNeuralBPVec) instead of scalar min-sum damping. This answers the question
Stage 1 could not: "does symmetry-chosen weight-tying beat a random partition of the same size when
the model class is genuinely expressive?"

PRE-REGISTERED GATE (decide before looking at results):
  equiv-vec must beat free_random-vec beyond seed noise at BOTH p (all 5 paired seeds AND mean margin
  > 2 std). Outcomes:
    - GATE FAILS (equiv ~= free_random): the equivariance negative is now airtight across model
      classes -> close the program; symmetry tying does no work even with an expressive decoder.
    - GATE PASSES (equiv > free_random): a genuine symmetry signal survives expressiveness -> the
      negative was a weak-model artifact; reopen (extend p/R, write up).
  SEPARATELY report equiv-vec vs tuned classical BP-OSD (the architecture-strength bound): if the
  expressive arm still cannot tie classical, "no neural decoder of this class beats classical here"
  is the honest, strong close.

Matched params: equiv and free_random both use n_orb groups => identical 2*n_orb*T*D*D params.
Run on GPU: python sweep_vec.py    (R=6, p in {0.005,0.01}, 5 paired seeds; ~hours on CPU, minutes on GPU)
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from bb_code import BBCode
from bb_circuit import build_z_memory
from canon_dem import extract
from trains_canon import canon_fgdata, sample_dem, eval_ler
from circ_neural_bp import CircNeuralBP          # classical baseline arm
from circ_neural_bp_vec import CircNeuralBPVec   # expressive equiv / free_random arms

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
# result written next to sweep_stage1.json (committable; qldpc/results/ is gitignored) so the pod
# run can be pulled back with `git add qldpc/sweep_vec_stage1.json && git commit && git push`.
_RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_vec_stage1.json')
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

R = 6
P_POINTS = [0.005, 0.01]
SEEDS = [1, 2, 3, 4, 5]
T = 8
D = 4              # message width: 2*n_orb*T*D*D ~= 10k params at n_orb=40 (>~60x the scalar arm)
N_TRAIN = 12000
N_TEST = 4000
STEPS = 600
BATCH = 512
LR = 0.05


def train_arm(fgd, mode, ex, s_tr, e_tr, seed, rand_seed):
    torch.manual_seed(seed)
    m = CircNeuralBPVec(fgd, T=T, D=D, mode=mode, rand_seed=rand_seed).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    for _ in range(STEPS):
        idx = torch.randint(0, s_tr.shape[0], (BATCH,), generator=g)
        opt.zero_grad()
        lf(-m(s_tr[idx].to(DEV)), e_tr[idx].to(DEV)).backward()
        opt.step()
    return m


def main():
    bb = BBCode()
    out = {'stage': '1-vec', 'desc': 'architecture-strength: expressive equiv-vec vs free_random-vec '
           '(matched params) + classical', 'R': R, 'D': D, 'T': T, 'seeds': SEEDS,
           'device': str(DEV), 'points': {}}
    for p in P_POINTS:
        ex = extract(build_z_memory(bb, rounds=R, p=p).detector_error_model(decompose_errors=False))
        fgd = canon_fgdata(ex)
        s_tr, e_tr, _ = sample_dem(ex, N_TRAIN, 7)
        row = {'n_orb': fgd['n_orb']}

        # classical baseline (untrained tuned min-sum + OSD), per seed
        cl = CircNeuralBP(fgd, T=T, mode='classical').to(DEV)
        cl_lers = []
        for sd in SEEDS:
            s_te, e_te, o_te = sample_dem(ex, N_TEST, 100 + sd)
            cl_lers.append(eval_ler(cl, ex, s_te, o_te, DEV)[0])
        row['classical'] = {'params': 0, 'ler_mean': round(float(np.mean(cl_lers)), 5),
                            'ler_seeds': [round(x, 5) for x in cl_lers]}

        for mode in ['equiv', 'free_random']:
            lers, nparams = [], None
            for sd in SEEDS:
                m = train_arm(fgd, mode, ex, s_tr, e_tr, seed=sd, rand_seed=sd)  # partition varies per seed
                nparams = m.params_report
                s_te, e_te, o_te = sample_dem(ex, N_TEST, 100 + sd)
                lers.append(eval_ler(m, ex, s_te, o_te, DEV)[0])
            a = np.array(lers)
            row[mode] = {'params': nparams, 'ler_mean': round(float(a.mean()), 5),
                         'ler_std': round(float(a.std()), 5), 'ler_seeds': [round(x, 5) for x in lers]}

        eqs = np.array(row['equiv']['ler_seeds']); frs = np.array(row['free_random']['ler_seeds'])
        diff = frs - eqs   # positive => equiv better (lower LER)
        row['equiv_vs_free_random'] = {
            'paired_mean_margin': round(float(diff.mean()), 5),
            'paired_std': round(float(diff.std()), 5),
            'equiv_wins_all_seeds': bool((diff > 0).all()),
            'margin_over_2std': bool(diff.mean() > 2 * diff.std() / np.sqrt(len(diff)))}
        row['equiv_beats_classical'] = bool(row['equiv']['ler_mean'] < row['classical']['ler_mean'])
        out['points'][f'p{p}'] = row
        json.dump(out, open(_RESULT, 'w'), indent=2)
        print(f"p={p}: classical {row['classical']['ler_mean']:.4f}  equiv {row['equiv']['ler_mean']:.4f}"
              f"  free_random {row['free_random']['ler_mean']:.4f}  margin "
              f"{row['equiv_vs_free_random']['paired_mean_margin']:+.4f}  "
              f"wins_all {row['equiv_vs_free_random']['equiv_wins_all_seeds']}", flush=True)

    passed = all(out['points'][f'p{p}']['equiv_vs_free_random']['equiv_wins_all_seeds']
                 and out['points'][f'p{p}']['equiv_vs_free_random']['margin_over_2std'] for p in P_POINTS)
    out['STAGE1_VEC_PASS'] = bool(passed)
    out['verdict'] = ('expressive symmetry signal survives — reopen' if passed else
                      'equiv ties free_random even when expressive — equivariance negative is airtight; close')
    beats_cl = all(out['points'][f'p{p}']['equiv_beats_classical'] for p in P_POINTS)
    out['expressive_arm_beats_classical'] = bool(beats_cl)
    json.dump(out, open(os.path.join(_OUT, 'sweep_vec_stage1.json'), 'w'), indent=2)
    print("STAGE1_VEC_PASS", out['STAGE1_VEC_PASS'], "| expressive beats classical:", beats_cl)


if __name__ == '__main__':
    main()
