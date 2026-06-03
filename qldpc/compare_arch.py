"""Decisive 3-way: does a STRUCTURE-AWARE equivariant kernel beat the generic equivariant CNN
and the plain MLP? Param-matched, equal training, multi-seed, at the two sizes where the generic
equiv decoder lost most clearly (n=2000 @3.2sigma, n=20000 @6.2sigma).

If struct-equiv beats MLP -> the equivariance thesis is salvageable (the bug was kernel support).
If struct-equiv also loses -> the negative result is robust across architectures (honest kill).
"""
import os
import json, sys
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
import torch, torch.nn as nn
from bb_code import BBCode
from equiv_decoder import EquivBBDecoder, MLPDecoder, count_params
from struct_equiv import StructEquivBBDecoder
from _util import wilson_ci


def make_data(bb, L, p, n, seed):
    rng = np.random.default_rng(seed)
    e = (rng.random((n, bb.n)) < p).astype(np.int64)
    s = (e @ bb.HZ.astype(np.int64).T) % 2
    y = (e @ L.astype(np.int64).T) % 2
    return torch.tensor(s, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def ler(model, s, y, device, bs=8192):
    model.eval(); w = t = 0
    with torch.no_grad():
        for i in range(0, s.shape[0], bs):
            pr = (model(s[i:i+bs].to(device)) > 0).float().cpu()
            w += (pr != y[i:i+bs]).sum().item(); t += pr.numel()
    return wilson_ci(w, t)[0]


def train(model, trs, trY, device, steps, bs, lr, seed):
    torch.manual_seed(seed); model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    lf = nn.BCEWithLogitsLoss(); n = trs.shape[0]
    for st in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        opt.zero_grad(); loss = lf(model(trs[idx].to(device)), trY[idx].to(device))
        loss.backward(); opt.step()


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    bb = BBCode(); L = bb.logicals_Z().astype(np.float32); p = 0.03
    te_s, te_y = make_data(bb, L, p, 20000, 999)

    # param-match all three near the generic-equiv budget (~139k)
    target = count_params(EquivBBDecoder(hidden=64, depth=4))
    # struct hidden to match
    sh, sd_ = 64, 1 << 30
    for h in range(40, 96):
        dd = abs(count_params(StructEquivBBDecoder(hidden=h, depth=4)) - target)
        if dd < sd_: sd_, sh = dd, h
    mw, md_ = 251, 1 << 30
    for w in range(180, 340):
        dd = abs(count_params(MLPDecoder(width=w, depth=4)) - target)
        if dd < md_: md_, mw = dd, w

    ctors = {
        'struct': lambda: StructEquivBBDecoder(hidden=sh, depth=4),
        'generic': lambda: EquivBBDecoder(hidden=64, depth=4),
        'mlp': lambda: MLPDecoder(width=mw, depth=4),
    }
    out = {'p': p, 'steps': 15000, 'seeds': [1, 2, 3],
           'params': {k: count_params(c()) for k, c in ctors.items()},
           'struct_hidden': sh, 'mlp_width': mw,
           'bposd_bar': 0.030883, 'results': {}}
    for ntr in [2000, 20000]:
        row = {}
        for name, ctor in ctors.items():
            lers = []
            for sd in [1, 2, 3]:
                trs, trY = make_data(bb, L, p, ntr, np.random.default_rng(1000 * sd + ntr))
                m = ctor()
                train(m, trs, trY, device, steps=15000, bs=256, lr=2e-3, seed=sd)
                lers.append(ler(m, te_s, te_y, device))
            a = np.array(lers)
            row[name] = dict(mean_ler=round(float(a.mean()), 5), std_ler=round(float(a.std()), 5),
                             per_seed=[round(x, 5) for x in lers])
        out['results'][str(ntr)] = row
        json.dump(out, open(os.path.join(_OUT, 'compare_arch.json'), 'w'), indent=2)
        print(f"n={ntr}: struct {row['struct']['mean_ler']:.4f} generic {row['generic']['mean_ler']:.4f} mlp {row['mlp']['mean_ler']:.4f}")
    print("WROTE compare_arch.json")


if __name__ == '__main__':
    main()
