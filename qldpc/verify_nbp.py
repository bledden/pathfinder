"""DEFINITIVE, leak-aware verification of the neural-BP 'beats BP-OSD' claim.

The train_nbp result (equiv neural-BP 1.28% per-logical, beating BP-OSD 3.09%) is suspiciously good.
Most likely confound: code-capacity i.i.d. noise on a small [[72,12,6]] code repeats common
low-weight syndromes, so a trained decoder can MEMORIZE syndrome->error and OSD-0 cleans up.
BP-OSD does not train, so that would be an unfair comparison.

This script settles it, all on IDENTICAL test shots:
  (1) syndrome-match rate of every decoder's correction (correctness gate; must be ~1.0).
  (2) FULL test per-logical LER.
  (3) LEAK-FREE test per-logical LER: evaluate ONLY on test shots whose syndrome never appeared
      in training. If the win evaporates here, it was memorization.
  (4) real ldpc BP-OSD (ms + osd_cs order7) on the SAME shots = the honest bar.
Modes: classical (untrained), equiv (orbit-tied), free (per-edge). Wilson CIs. Numbers from arrays.
"""
import os
import json, sys, time
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
import torch
import torch.nn as nn
from bb_code import BBCode
from neural_bp import NeuralBP
from ldpc import bposd_decoder
from _util import wilson_ci


def gen(bb, p, n, seed):
    rng = np.random.default_rng(seed)
    e = (rng.random((n, bb.n)) < p).astype(np.int64)
    s = ((e @ bb.HZ.astype(np.int64).T) % 2).astype(np.uint8)
    return s, e.astype(np.uint8)


def train(model, bb, p, n_train, steps, seed):
    s, e = gen(bb, p, n_train, 1000 * seed + 1)
    ts = torch.tensor(s, dtype=torch.float32); te = torch.tensor(e, dtype=torch.float32)
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=0.02)
    lf = nn.BCEWithLogitsLoss(); g = torch.Generator().manual_seed(seed)
    model.train()
    for st in range(steps):
        idx = torch.randint(0, n_train, (256,), generator=g)
        opt.zero_grad(); Lv = model(ts[idx], p); loss = lf(-Lv, te[idx])
        loss.backward(); opt.step()
    train_synds = set(map(tuple, s.tolist()))
    return model, train_synds


def neural_eval(bb, model, s, e, p):
    L = bb.logicals_Z().astype(np.int64); H = bb.HZ.astype(np.int64)
    ts = torch.tensor(s, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        Lv = model(ts, p)
        ehat = model._osd0(Lv, ts).cpu().numpy().astype(np.int64)
    synd_match = float((((ehat @ H.T) % 2) == s).all(axis=1).mean())
    res = (e.astype(np.int64) ^ ehat) % 2
    flips = (res @ L.T) % 2
    N, k = flips.shape
    per_logical_fails = int(flips.sum())
    block_fails = int((flips.sum(1) > 0).sum())
    pl, plo, phi = wilson_ci(per_logical_fails, N * k)
    bl, blo, bhi = wilson_ci(block_fails, N)
    return dict(per_logical=pl, pl_ci=[plo, phi], block=bl, synd_match=round(synd_match, 4), n=N)


def ldpc_eval(bb, s, e, p):
    H = bb.HZ.astype(np.uint8); L = bb.logicals_Z().astype(np.int64)
    dec = bposd_decoder(H, error_rate=p, max_iter=72, bp_method='ms',
                        osd_method='osd_cs', osd_order=7)
    N = s.shape[0]; k = L.shape[0]
    fl = 0; blk = 0; sm = 0
    for i in range(N):
        c = dec.decode(s[i]).astype(np.int64)
        if (((c @ H.T) % 2) == s[i]).all():
            sm += 1
        r = (e[i].astype(np.int64) ^ c) % 2
        f = (L @ r) % 2
        fl += int(f.sum()); blk += int(f.any())
    pl, plo, phi = wilson_ci(fl, N * k)
    bl, blo, bhi = wilson_ci(blk, N)
    return dict(per_logical=pl, pl_ci=[plo, phi], block=bl, synd_match=round(sm / N, 4), n=N)


def main():
    bb = BBCode(); p = 0.03
    T = 12; n_train = 20000; steps = 4000; seeds = [1, 2, 3]
    N_TEST = 30000
    s_te, e_te = gen(bb, p, N_TEST, 99999)
    out = {'p': p, 'T': T, 'n_train': n_train, 'steps': steps, 'seeds': seeds, 'n_test': N_TEST}

    t0 = time.time()
    out['ldpc_bposd'] = ldpc_eval(bb, s_te, e_te, p)
    print(f"ldpc BP-OSD: per-logical {out['ldpc_bposd']['per_logical']:.4f} "
          f"block {out['ldpc_bposd']['block']:.4f} synd_match {out['ldpc_bposd']['synd_match']} "
          f"({time.time()-t0:.0f}s)")
    json.dump(out, open(os.path.join(_OUT, 'verify_nbp.json'), 'w'), indent=2)

    cl = NeuralBP(bb, T=T, mode='classical')
    out['classical'] = neural_eval(bb, cl, s_te, e_te, p)
    print(f"classical BP+OSD0: per-logical {out['classical']['per_logical']:.4f} "
          f"synd_match {out['classical']['synd_match']}")
    json.dump(out, open(os.path.join(_OUT, 'verify_nbp.json'), 'w'), indent=2)

    for mode in ['equiv', 'free']:
        full_pl, dedup_pl, sm, kept = [], [], [], []
        nparams = sum(q.numel() for q in NeuralBP(bb, T=T, mode=mode).parameters() if q.requires_grad)
        for sd in seeds:
            m, trsynds = train(NeuralBP(bb, T=T, mode=mode), bb, p, n_train, steps, sd)
            full = neural_eval(bb, m, s_te, e_te, p)
            full_pl.append(full['per_logical']); sm.append(full['synd_match'])
            mask = np.array([tuple(row) not in trsynds for row in s_te.tolist()])
            kept.append(float(mask.mean()))
            ded = neural_eval(bb, m, s_te[mask], e_te[mask], p)
            dedup_pl.append(ded['per_logical'])
        a, d = np.array(full_pl), np.array(dedup_pl)
        out[mode] = dict(params=nparams,
                         full_per_logical_mean=round(float(a.mean()), 5),
                         full_per_logical_std=round(float(a.std()), 5),
                         full_per_logical_seeds=[round(x, 5) for x in full_pl],
                         dedup_per_logical_mean=round(float(d.mean()), 5),
                         dedup_per_logical_seeds=[round(x, 5) for x in dedup_pl],
                         dedup_frac_kept=round(float(np.mean(kept)), 4),
                         synd_match_mean=round(float(np.mean(sm)), 4),
                         beats_bposd_full=bool(a.mean() < out['ldpc_bposd']['per_logical']),
                         beats_bposd_dedup=bool(d.mean() < out['ldpc_bposd']['per_logical']))
        print(f"{mode}: full {a.mean():.4f} | dedup {d.mean():.4f} (kept {np.mean(kept):.2f}) "
              f"| synd_match {np.mean(sm):.3f} | params {nparams}")
        json.dump(out, open(os.path.join(_OUT, 'verify_nbp.json'), 'w'), indent=2)
    print("WROTE verify_nbp.json")


if __name__ == '__main__':
    main()
