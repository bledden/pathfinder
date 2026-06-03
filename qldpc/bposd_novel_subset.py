"""Control: evaluate BP-OSD on the NOVEL-ONLY subset, apples-to-apples.

verify_nbp.json reports BP-OSD on the FULL test set (2.977% per-logical). The neural dedup numbers
are on novel syndromes (per-seed train mask). To compare like-for-like we must evaluate BP-OSD on the
SAME per-seed novel masks and average over seeds, exactly as the neural dedup mean is built.

Reuses verify_nbp.py's exact gen(), train-set seeds, test set, and BP-OSD config (ldpc bposd_decoder,
ms + osd_cs order 7, max_iter 72). Reports BP-OSD full vs novel per-logical AND block, with the
memorized-subset LER decomposition for the neural decoders (the memorization decomposition decomposition).
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


def bposd_eval(bb, dec, s, e):
    H = bb.HZ.astype(np.int64); L = bb.logicals_Z().astype(np.int64)
    N = s.shape[0]; k = L.shape[0]
    fl = 0; blk = 0
    for i in range(N):
        c = dec.decode(s[i]).astype(np.int64)
        r = (e[i].astype(np.int64) ^ c) % 2
        f = (L @ r) % 2
        fl += int(f.sum()); blk += int(f.any())
    return fl, blk, N, k


def main():
    bb = BBCode(); p = 0.03
    n_train = 20000; seeds = [1, 2, 3]; N_TEST = 30000
    s_te, e_te = gen(bb, p, N_TEST, 99999)   # identical to verify_nbp test set
    H = bb.HZ.astype(np.uint8)
    dec = bposd_decoder(H, error_rate=p, max_iter=72, bp_method='ms',
                        osd_method='osd_cs', osd_order=7)

    # FULL-test BP-OSD (reproduce the 2.977% bar for self-consistency)
    t0 = time.time()
    fl, blk, N, k = bposd_eval(bb, dec, s_te, e_te)
    full_pl = wilson_ci(fl, N * k); full_bl = wilson_ci(blk, N)
    out = {'p': p, 'n_train': n_train, 'seeds': seeds, 'n_test': N_TEST,
           'bposd_config': 'ldpc ms + osd_cs order7, max_iter72',
           'bposd_full': {'per_logical': full_pl[0], 'pl_ci': [full_pl[1], full_pl[2]],
                          'block': full_bl[0], 'fails_pl': fl, 'fails_blk': blk, 'n': N}}
    json.dump(out, open(os.path.join(_OUT, 'bposd_novel_subset.json'), 'w'), indent=2)
    print(f"BP-OSD full: per-logical {full_pl[0]:.5f} ({time.time()-t0:.0f}s)")

    # NOVEL-only BP-OSD, per-seed mask, then pooled
    per_seed = []
    pooled_fl = pooled_blk = pooled_npl = pooled_nblk = 0
    for sd in seeds:
        s_tr, _ = gen(bb, p, n_train, 1000 * sd + 1)   # identical to verify_nbp train set for this seed
        trset = set(map(tuple, s_tr.tolist()))
        mask = np.array([tuple(row) not in trset for row in s_te.tolist()])
        fl_s, blk_s, N_s, k_s = bposd_eval(bb, dec, s_te[mask], e_te[mask])
        pl_s = wilson_ci(fl_s, N_s * k_s)
        per_seed.append({'seed': sd, 'frac_novel': round(float(mask.mean()), 4), 'n_novel': int(N_s),
                         'per_logical': pl_s[0], 'pl_ci': [pl_s[1], pl_s[2]],
                         'block': wilson_ci(blk_s, N_s)[0]})
        pooled_fl += fl_s; pooled_blk += blk_s; pooled_npl += N_s * k_s; pooled_nblk += N_s
        print(f"  seed {sd}: novel n={N_s} BP-OSD per-logical {pl_s[0]:.5f}")
    pooled_pl = wilson_ci(pooled_fl, pooled_npl); pooled_bl = wilson_ci(pooled_blk, pooled_nblk)
    out['bposd_novel_per_seed'] = per_seed
    out['bposd_novel_pooled'] = {'per_logical': pooled_pl[0], 'pl_ci': [pooled_pl[1], pooled_pl[2]],
                                 'block': pooled_bl[0], 'n_shots_pooled': pooled_nblk}
    out['bposd_novel_mean_over_seeds'] = round(float(np.mean([x['per_logical'] for x in per_seed])), 5)

    # Memorization decomposition: neural LER on the memorized (non-novel) subset.
    # full = f_novel*novel + (1-f_novel)*mem  =>  mem = (full - f*novel)/(1-f)
    v = json.load(open(os.path.join(_OUT, 'verify_nbp.json')))
    decomp = {}
    for mode in ['equiv', 'free']:
        full = v[mode]['full_per_logical_mean']; novel = v[mode]['dedup_per_logical_mean']
        f = v[mode]['dedup_frac_kept']
        mem = (full - f * novel) / (1.0 - f)
        decomp[mode] = {'full': full, 'novel': novel, 'frac_novel': f,
                        'memorized_subset_ler': round(float(mem), 6)}
    out['neural_memorized_decomposition'] = decomp
    json.dump(out, open(os.path.join(_OUT, 'bposd_novel_subset.json'), 'w'), indent=2)
    print("WROTE bposd_novel_subset.json  | BP-OSD novel pooled per-logical %.5f" % pooled_pl[0])


if __name__ == '__main__':
    main()
