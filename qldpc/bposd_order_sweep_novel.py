"""DETERMINING EXPERIMENT: sweep BP-OSD OSD order on the NOVEL subset.

bposd_novel_subset.py showed BP-OSD order-7 on novel syndromes = 7.72% per-logical, vs equiv
neural-BP 3.48% on the same novel subset -> neural appears to BEAT BP-OSD-7 by ~2.2x. But our own
Diagnostic 3 says OSD order matters enormously on BB codes (Cascade uses order 60). So before any
conclusion we must tune OUR BP-OSD baseline. This sweeps osd_cs order {7,15,30,60} on the novel
subset (seed-1 mask, ~11.4k shots) and reports per-logical LER vs the neural-BP novel number 3.48%.

Outcomes:
  - tuned BP-OSD (order ~60) drops below 3.48% -> §4 null restored, BUT the verdict now correctly
    flows through Diagnostic 3 (the apparent neural win on novel was a baseline-tuning artifact).
  - tuned BP-OSD stays above 3.48% -> neural-BP genuinely beats tuned classical on hard/novel
    syndromes: a POSITIVE result that would materially change the paper. Either way, report honestly.
Same gen()/seeds/test set as verify_nbp.py. Times each order (order-60 combination-sweep is slow).
"""
import os
import json, sys, time
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
from bb_code import BBCode
from ldpc import bposd_decoder
from _util import wilson_ci


def gen(bb, p, n, seed):
    rng = np.random.default_rng(seed)
    e = (rng.random((n, bb.n)) < p).astype(np.int64)
    s = ((e @ bb.HZ.astype(np.int64).T) % 2).astype(np.uint8)
    return s, e.astype(np.uint8)


def eval_order(bb, s, e, order, p):
    H = bb.HZ.astype(np.uint8); L = bb.logicals_Z().astype(np.int64)
    dec = bposd_decoder(H, error_rate=p, max_iter=72, bp_method='ms',
                        osd_method='osd_cs', osd_order=order)
    N = s.shape[0]; k = L.shape[0]
    fl = 0; blk = 0; t0 = time.time()
    for i in range(N):
        c = dec.decode(s[i]).astype(np.int64)
        r = (e[i].astype(np.int64) ^ c) % 2
        f = (L @ r) % 2
        fl += int(f.sum()); blk += int(f.any())
    pl = wilson_ci(fl, N * k); bl = wilson_ci(blk, N)
    return dict(order=order, per_logical=pl[0], pl_ci=[pl[1], pl[2]], block=bl[0],
                n=N, sec=round(time.time() - t0, 1))


def main():
    bb = BBCode(); p = 0.03; n_train = 20000
    s_te, e_te = gen(bb, p, 30000, 99999)
    s_tr, _ = gen(bb, p, n_train, 1000 * 1 + 1)   # seed 1 train mask
    trset = set(map(tuple, s_tr.tolist()))
    mask = np.array([tuple(row) not in trset for row in s_te.tolist()])
    s_n, e_n = s_te[mask], e_te[mask]
    out = {'p': p, 'subset': 'novel (seed-1 mask)', 'n_novel': int(s_n.shape[0]),
           'neural_equiv_novel': 0.03479, 'neural_free_novel': 0.03711,
           'bposd_order7_novel_pooled_ref': 0.07718, 'orders': {}}
    for order in [7, 15, 30, 60]:
        r = eval_order(bb, s_n, e_n, order, p)
        out['orders'][f'osd{order}'] = r
        json.dump(out, open(os.path.join(_OUT, 'bposd_order_sweep_novel.json'), 'w'), indent=2)
        print(f"osd{order}: novel per-logical {r['per_logical']:.5f}  ({r['sec']}s)")
    best = min(out['orders'].values(), key=lambda r: r['per_logical'])
    out['best_order'] = best['order']
    out['best_bposd_novel'] = best['per_logical']
    out['neural_equiv_beats_best_bposd_on_novel'] = bool(0.03479 < best['per_logical'])
    json.dump(out, open(os.path.join(_OUT, 'bposd_order_sweep_novel.json'), 'w'), indent=2)
    print(f"BEST BP-OSD novel = order {best['order']} @ {best['per_logical']:.5f} ; "
          f"equiv neural 0.03479 {'BEATS' if 0.03479 < best['per_logical'] else 'LOSES TO'} it")
    print("WROTE bposd_order_sweep_novel.json")


if __name__ == '__main__':
    main()
