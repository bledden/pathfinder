"""Sample-efficiency training: EquivBBDecoder vs matched-capacity MLPDecoder on BB code-capacity.

The hypothesis: group-equivariance (Z_l x Z_m weight sharing) buys SAMPLE EFFICIENCY,
not per-shot LER. So we train BOTH models at several training-set sizes n_train and compare
test per-logical LER vs n_train. Win = equiv reaches matched LER at fewer samples.

Decoder framing (code-capacity, X errors, detect with HZ, k=12 logicals):
  - INPUT: syndrome s = (e @ HZ^T) % 2,  shape (N=36,)
  - TARGET: per-logical flip label y = (e @ L^T) % 2, shape (k=12,)  [L = Z-logicals]
  - The net predicts P(logical j flips | s); BCEWithLogits multi-label.
  - At test: predicted logical-flip = (logit>0); per-logical LER = mean over (shot, logical) of
    (pred != y). This is exactly the BP-OSD per-logical metric -> directly comparable to the bar.
    NOTE: predicting the logical-coset directly (not a physical correction) is the standard neural
    QEC formulation (AlphaQubit, Lange) and is well-defined because y is a deterministic fn of s
    ONLY up to stabilizer degeneracy — the net learns the MAP estimate, same target a decoder needs.

Self-validation mode runs tiny + asserts: (a) loss decreases, (b) test LER << random(0.5),
(c) both models forward/backward clean, before scaling up.
"""
import os
import json, sys, time, argparse
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
import torch
import torch.nn as nn
from bb_code import BBCode
from equiv_decoder import EquivBBDecoder, MLPDecoder, count_params
from _util import wilson_ci


def make_data(bb, L, p, n, rng):
    # integer parity (exact; avoids float-matmul NaN warning on Accelerate/MPS)
    e = (rng.random((n, bb.n)) < p).astype(np.int64)
    HZ = bb.HZ.astype(np.int64)
    Li = L.astype(np.int64)
    s = (e @ HZ.T) % 2
    y = (e @ Li.T) % 2
    return torch.tensor(s, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def evaluate(model, s, y, device, bs=4096):
    model.eval()
    wrong = 0; tot = 0
    with torch.no_grad():
        for i in range(0, s.shape[0], bs):
            logits = model(s[i:i+bs].to(device))
            pred = (logits > 0).float().cpu()
            wrong += (pred != y[i:i+bs]).sum().item()
            tot += pred.numel()
    return wrong, tot


def train_one(model, tr_s, tr_y, device, steps, bs, lr, seed):
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    n = tr_s.shape[0]
    g = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        xb, yb = tr_s[idx].to(device), tr_y[idx].to(device)
        opt.zero_grad()
        loss = lossf(model(xb), yb)
        loss.backward(); opt.step()
        if step % max(steps // 10, 1) == 0:
            losses.append(round(float(loss), 4))
    return losses


def run(mode):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    bb = BBCode()
    L = bb.logicals_Z().astype(np.float32)  # (12, 72)
    k = L.shape[0]
    rng = np.random.default_rng(0)
    p = 0.03

    # fixed large test set (held out)
    te_s, te_y = make_data(bb, L, p, 20000, np.random.default_rng(999))

    if mode == 'validate':
        # tiny sanity: small train, few steps, assert learning
        tr_s, tr_y = make_data(bb, L, p, 5000, rng)
        eq = EquivBBDecoder(hidden=32, depth=3)
        mlp = MLPDecoder(width=180, depth=3)
        out = {'p': p, 'k': k, 'equiv_params': count_params(eq), 'mlp_params': count_params(mlp),
               'baseline_bposd_per_logical_p0.03': 0.030883, 'random_ler': 0.5}
        for name, model in [('equiv', eq), ('mlp', mlp)]:
            losses = train_one(model, tr_s, tr_y, device, steps=400, bs=256, lr=2e-3, seed=1)
            w, t = evaluate(model, te_s, te_y, device)
            ler, lo, hi = wilson_ci(w, t)
            out[name] = dict(losses=losses, test_per_logical_ler=ler, ci=[lo, hi],
                             learned=bool(losses[-1] < losses[0] and ler < 0.4))
        out['VALIDATE_PASS'] = bool(out['equiv']['learned'] and out['mlp']['learned'])
        json.dump(out, open(os.path.join(_OUT, 'train_validate.json'), 'w'), indent=2)
        print("WROTE train_validate.json")

    elif mode == 'sweep':
        # FAIR sample-efficiency sweep (fixes 2 confounds from the first run):
        #  (1) TRUE param-match: auto-pick MLP width so mlp_params ~= equiv_params (was 210k vs 139k).
        #  (2) ADEQUATE + EQUAL training: flat STEPS gradient steps for BOTH models at EVERY size
        #      (the old formula collapsed to 3000 for all sizes -> large-n undertrained).
        STEPS = 15000
        equiv_ctor = lambda: EquivBBDecoder(hidden=64, depth=4)
        eqp = count_params(equiv_ctor())
        best_w, best_d = 64, 1 << 30
        for w in range(120, 400):
            d = abs(count_params(MLPDecoder(width=w, depth=4)) - eqp)
            if d < best_d:
                best_d, best_w = d, w
        mlp_ctor = lambda: MLPDecoder(width=best_w, depth=4)
        ctors = {'equiv': equiv_ctor, 'mlp': mlp_ctor}
        sizes = [500, 1000, 2000, 5000, 20000, 100000]
        seeds = [1, 2, 3]
        out = {'p': p, 'k': k, 'steps_per_run': STEPS,
               'equiv_params': eqp, 'mlp_params': count_params(mlp_ctor()), 'mlp_width': best_w,
               'baseline_bposd_per_logical': 0.030883,
               'sizes': sizes, 'seeds': seeds, 'results': {}}
        for ntr in sizes:
            row = {}
            for name, ctor in ctors.items():
                lers = []
                for sd in seeds:
                    # each seed: fresh train data draw + fresh init + fresh batch order
                    tr_s, tr_y = make_data(bb, L, p, ntr, np.random.default_rng(1000 * sd + ntr))
                    model = ctor()
                    train_one(model, tr_s, tr_y, device, steps=STEPS, bs=256, lr=2e-3, seed=sd)
                    w, t = evaluate(model, te_s, te_y, device)
                    lers.append(wilson_ci(w, t)[0])
                arr = np.array(lers)
                row[name] = dict(per_seed_ler=[round(x, 5) for x in lers],
                                 mean_ler=round(float(arr.mean()), 5),
                                 std_ler=round(float(arr.std()), 5), steps=STEPS)
            out['results'][str(ntr)] = row
            json.dump(out, open(os.path.join(_OUT, 'train_sweep_fair.json'), 'w'), indent=2)
            print(f"size {ntr} done: equiv {row['equiv']['mean_ler']:.4f} mlp {row['mlp']['mean_ler']:.4f}")
        print("WROTE train_sweep_fair.json")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', nargs='?', default='validate')
    run(ap.parse_args().mode)
