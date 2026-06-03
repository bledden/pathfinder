"""GATE EXPERIMENT: does equiv neural-BP edge tuned Relay-BP on NOVEL syndromes across noise rates?

The single-p (p=0.03) result: equiv neural-BP novel 3.48% vs Relay-BP novel 3.61%, paired-win on all
3 seeds, margin +0.13pp (~2sigma). That is "promising, not decisive." This tightens it by sweeping
p in {0.02, 0.03, 0.04}, matched seeds, paired per-seed comparison + pooled CIs. A consistent
paired sign across p is far stronger than one point; a collapse kills the signal cheaply (CPU only).

Honest scope: this is CODE-CAPACITY. It is a GATE for whether the circuit-level (deployment-regime)
scaled experiment is worth running, not a deployment claim itself.

Reuses verify_nbp.py's train()/gen()/neural_eval() and relay_novel.py's relay_eval() verbatim so the
neural and classical pipelines are identical to the already-recorded runs.
"""
import os
import json, sys, time
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
from bb_code import BBCode
from neural_bp import NeuralBP
import verify_nbp as V          # train, gen, neural_eval
import relay_novel as RN        # relay_eval
from _util import wilson_ci


def main():
    bb = BBCode()
    T = 12; n_train = 20000; steps = 4000; seeds = [1, 2, 3]; N_TEST = 30000
    ps = [0.02, 0.03, 0.04]
    out = {'n_train': n_train, 'steps': steps, 'seeds': seeds, 'n_test': N_TEST,
           'subset': 'novel (per-seed train mask)', 'decoder': 'equiv neural-BP vs Relay-BP (tuned)',
           'scope': 'CODE-CAPACITY gate for scaled-experiment decision', 'points': {}}
    for p in ps:
        s_te, e_te = V.gen(bb, p, N_TEST, 99999)
        eq_novel, relay_novel, paired = [], [], []
        for sd in seeds:
            # train equiv at this p, this seed
            m, trset = V.train(NeuralBP(bb, T=T, mode='equiv'), bb, p, n_train, steps, sd)
            mask = np.array([tuple(r) not in trset for r in s_te.tolist()])
            sN, eN = s_te[mask], e_te[mask]
            # neural on novel
            ne = V.neural_eval(bb, m, sN, eN, p)['per_logical']
            # relay on the SAME novel mask
            fl, blk, N, k, sm = RN.relay_eval(bb, sN, eN, p)
            re = wilson_ci(fl, N * k)[0]
            eq_novel.append(ne); relay_novel.append(re); paired.append(re - ne)
        eqm, rem = float(np.mean(eq_novel)), float(np.mean(relay_novel))
        pm = float(np.mean(paired))
        out['points'][f'p{p}'] = dict(
            equiv_novel_seeds=[round(x, 5) for x in eq_novel], equiv_novel_mean=round(eqm, 5),
            relay_novel_seeds=[round(x, 5) for x in relay_novel], relay_novel_mean=round(rem, 5),
            paired_relay_minus_equiv=[round(x, 5) for x in paired], paired_mean=round(pm, 5),
            equiv_wins_all_seeds=bool(all(d > 0 for d in paired)),
            margin_pp=round(pm * 100, 4))
        json.dump(out, open(os.path.join(_OUT, 'tighten_multip_novel.json'), 'w'), indent=2)
        print(f"p={p}: equiv {eqm:.5f} relay {rem:.5f} | paired margin {pm*100:+.3f}pp "
              f"| equiv wins all seeds: {all(d>0 for d in paired)}")
    # verdict
    allwin = all(out['points'][f'p{p}']['equiv_wins_all_seeds'] for p in ps)
    out['equiv_wins_every_seed_every_p'] = bool(allwin)
    out['VERDICT'] = ('signal holds across p -> justifies the scaled experiment' if allwin
                      else 'signal inconsistent across p -> do not scale; likely noise')
    json.dump(out, open(os.path.join(_OUT, 'tighten_multip_novel.json'), 'w'), indent=2)
    print("VERDICT:", out['VERDICT'])
    print("WROTE tighten_multip_novel.json")


if __name__ == '__main__':
    main()
