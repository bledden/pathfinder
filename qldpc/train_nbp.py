"""Train + compare neural-BP variants vs classical BP and BP-OSD on [[72,12,6]] BB, code-cap p=0.03.

Variants: classical (no params), equiv (orbit-tied, ~206 params), free (per-edge, ~7346 params).
Target = per-qubit error (BCE on -marginal-LLR). Metric = per-logical LER (== BP-OSD metric).
Multi-seed. Writes JSON; numbers copied from arrays.

Key questions:
  1. Does training close BP's gap to BP-OSD (3.09% per-logical)?
  2. Does equiv (orbit-tied) match/beat free at 36x fewer params? (sample-efficiency, done right)
  3. Does equiv beat classical BP? (does learning help at all)
"""
import os
import json, sys, time
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
import torch, torch.nn as nn
from bb_code import BBCode
from neural_bp import NeuralBP
from _util import wilson_ci


def make_data(bb, p, n, seed):
    rng = np.random.default_rng(seed)
    e = (rng.random((n, bb.n)) < p).astype(np.float32)
    s = (e @ bb.HZ.astype(np.int64).T) % 2
    return torch.tensor(s, dtype=torch.float32), torch.tensor(e, dtype=torch.float32)


def eval_ler(model, s, e, p, bs=2048):
    model.eval()
    blk = pl = tot = 0; k = None
    with torch.no_grad():
        for i in range(0, s.shape[0], bs):
            b, l, k = model.decode_logical_fail(s[i:i+bs], e[i:i+bs], p)
            blk += b; pl += l; tot += s[i:i+bs].shape[0]
    return wilson_ci(blk, tot)[0], pl / (tot * k)


def train(model, trs, trE, p, steps, bs, lr, seed, dev):
    torch.manual_seed(seed); model.to(dev).train()
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=lr)
    lf = nn.BCEWithLogitsLoss()
    n = trs.shape[0]; g = torch.Generator().manual_seed(seed)
    for st in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        opt.zero_grad()
        Lv = model(trs[idx].to(dev), p)
        loss = lf(-Lv, trE[idx].to(dev))     # logit for e=1 is -LLR
        loss.backward(); opt.step()


def main():
    dev = torch.device('cpu')   # tiny graph, CPU fine and avoids MPS scatter quirks
    bb = BBCode(); p = 0.03
    T = 12
    te_s, te_e = make_data(bb, p, 20000, 999)
    seeds = [1, 2, 3]
    out = {'p': p, 'T': T, 'seeds': seeds, 'bposd_per_logical': 0.030883,
           'bposd_block': 0.06875, 'results': {}}

    # classical baseline (no training)
    cl = NeuralBP(bb, T=T, mode='classical').to(dev)
    cb, cpl = eval_ler(cl, te_s, te_e, p)
    out['classical'] = dict(per_logical_ler=round(cpl, 5), block_ler=round(cb, 5), params=0)

    n_train = 20000
    for mode in ['equiv', 'free']:
        pls = []; blks = []
        nparams = None; t0 = time.time()
        for sd in seeds:
            trs, trE = make_data(bb, p, n_train, 1000 * sd + 1)
            m = NeuralBP(bb, T=T, mode=mode)
            nparams = sum(q.numel() for q in m.parameters() if q.requires_grad)
            train(m, trs, trE, p, steps=4000, bs=256, lr=0.02, seed=sd, dev=dev)
            b, pl = eval_ler(m, te_s, te_e, p)
            pls.append(pl); blks.append(b)
        a = np.array(pls)
        out['results'][mode] = dict(
            params=nparams, n_train=n_train,
            per_logical_mean=round(float(a.mean()), 5), per_logical_std=round(float(a.std()), 5),
            per_logical_seeds=[round(x, 5) for x in pls],
            block_mean=round(float(np.mean(blks)), 5),
            beats_bposd=bool(a.mean() < 0.030883), min=round((time.time() - t0) / 60, 1))
        json.dump(out, open(os.path.join(_OUT, 'train_nbp.json'), 'w'), indent=2)
        print(f"{mode}: per-logical {a.mean():.4f} (params {nparams}) vs classical {cpl:.4f} vs BP-OSD 0.0309")
    json.dump(out, open(os.path.join(_OUT, 'train_nbp.json'), 'w'), indent=2)
    print("WROTE train_nbp.json")


if __name__ == '__main__':
    main()
