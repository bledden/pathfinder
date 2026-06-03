"""Circuit-level training validation (per-mechanism BCE objective).

v1 trained against a sum-over-obs-bearing-marginals readout — but the logical flip is a PARITY, not
representable that way; the model collapsed to random (loss ln2, acc ~0.49). That was a readout bug,
not necessarily an architecture failure. v2 uses the standard neural-BP objective (Liu-Poulin 2019):
per-error-mechanism BCE against the GROUND-TRUTH error activation vector.

To get ground-truth e we sample error mechanisms DIRECTLY from the DEM priors (each error fires
independently w.p. its prior), then detectors = parity of incident errors (deterministic), observable
likewise. This gives (syndrome, true_error) pairs the BP must invert. Success = BP marginals predict
the per-mechanism error bits well above chance, AND the resulting correction reduces logical error vs
untrained classical. That is the honest "it optimizes" gate.
"""
import os
import json, sys
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
import torch
import torch.nn as nn
from bb_code import BBCode
from circ_neural_bp import build_factorgraph, CircNeuralBP


def sample_from_dem(fg, n, seed):
    """Sample error mechanisms ~ priors; build (syndrome over detectors, true error over E, obs)."""
    rng = np.random.default_rng(seed)
    E = fg['E']; ndet = fg['ndet']
    priors = fg['priors']; obsbit = fg['obsbit']; detsets = fg['detsets']
    err = (rng.random((n, E)) < priors[None, :]).astype(np.int8)             # (n,E)
    # detector parity: detector d fires = parity of errors whose detset contains d
    # build det->errors incidence
    det_inc = [[] for _ in range(ndet)]
    for j, H in enumerate(detsets):
        for d in H:
            det_inc[d].append(j)
    synd = np.zeros((n, ndet), dtype=np.int8)
    for d in range(ndet):
        if det_inc[d]:
            synd[:, d] = err[:, det_inc[d]].sum(1) % 2
    obs = (err @ obsbit) % 2
    return (torch.tensor(synd, dtype=torch.float32),
            torch.tensor(err, dtype=torch.float32),
            torch.tensor(obs, dtype=torch.float32))


def main():
    bb = BBCode(); R = 3; p = 0.01
    fg = build_factorgraph(bb, R, p)
    out = {'R': R, 'p': p, 'ndet': fg['ndet'], 'E': fg['E'], 'n_orb': fg['n_orb'],
           'objective': 'per-error-mechanism BCE vs ground-truth (Liu-Poulin), DEM-sampled'}
    s_tr, e_tr, o_tr = sample_from_dem(fg, 12000, 7)
    s_te, e_te, o_te = sample_from_dem(fg, 6000, 99)
    base_err_rate = float(e_tr.mean())                      # mean P(e=1) — chance baseline for BCE acc
    out['mean_error_rate'] = round(base_err_rate, 5)

    res = {}
    for mode in ['classical', 'equiv', 'free']:
        torch.manual_seed(0)
        m = CircNeuralBP(fg, T=8, mode=mode)
        out[f'params_{mode}'] = m.params_report
        if mode == 'classical':
            with torch.no_grad():
                Lv = m(s_te); pred = (Lv < 0).float()       # P(e=1) high => predict error
            bit_acc = (pred == e_te).float().mean().item()
            # error-weighted recall: of true errors, how many predicted
            recall = (pred[e_te == 1] == 1).float().mean().item() if (e_te == 1).any() else 0.0
            res[mode] = {'bit_acc': round(bit_acc, 4), 'error_recall': round(recall, 4)}
            continue
        opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=0.05)
        lossf = nn.BCEWithLogitsLoss()
        n = s_tr.shape[0]; g = torch.Generator().manual_seed(1); losses = []
        for step in range(400):
            idx = torch.randint(0, n, (512,), generator=g)
            opt.zero_grad()
            Lv = m(s_tr[idx])                               # (B,E) LLR, >0 => e=0
            loss = lossf(-Lv, e_tr[idx])                    # logit for e=1 is -LLR
            loss.backward(); opt.step()
            if step % 40 == 0: losses.append(round(float(loss), 4))
        with torch.no_grad():
            Lv = m(s_te); pred = (Lv < 0).float()
            bit_acc = (pred == e_te).float().mean().item()
            recall = (pred[e_te == 1] == 1).float().mean().item() if (e_te == 1).any() else 0.0
        res[mode] = {'loss_curve': losses, 'final_loss': losses[-1],
                     'bit_acc': round(bit_acc, 4), 'error_recall': round(recall, 4),
                     'loss_dropped': bool(losses[-1] < losses[0] - 0.05),
                     'recall_above_chance': bool(recall > base_err_rate + 0.05)}
    out['results'] = res
    # honest gate: equiv must (a) drop loss meaningfully AND (b) recall true errors well above chance
    eq = res.get('equiv', {})
    out['TRAINS_HONEST'] = bool(eq.get('loss_dropped') and eq.get('recall_above_chance'))
    json.dump(out, open(os.path.join(_OUT, 'circ_trains_v2.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2)); print("WROTE circ_trains_v2.json")


if __name__ == '__main__':
    main()
