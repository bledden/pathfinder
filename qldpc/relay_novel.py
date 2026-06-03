"""DETERMINING EXPERIMENT pt.2: Relay-BP on the NOVEL subset, the strong-classical bar.

BP-OSD order-60 on novel = 5.60% per-logical, LOSES to equiv neural-BP 3.48%. But Cascade's headline
baseline is RELAY, and our Phase 0a showed Relay-BP beats BP-OSD-10 by ~3x. So the real question is
whether RELAY-BP on the novel subset drops below the neural 3.48%. If yes -> neural "win" was us
under-tuning (Diagnostic 3 self-catch). If no -> neural-BP genuinely beats tuned classical on hard
syndromes (a real positive result).

Same gen()/seeds/test set/novel mask as verify_nbp & bposd_novel_subset. Relay-BP decodes H_Z
directly (per-qubit prior p over 72 qubits -> correction), exactly as BP-OSD was called there. Block
per-logical LER over the 12 Z-logicals, Wilson CI. Multi-seed mask, pooled.
"""
import os
import json, sys, time
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
import scipy.sparse as sp
from bb_code import BBCode
from _util import wilson_ci


def gen(bb, p, n, seed):
    rng = np.random.default_rng(seed)
    e = (rng.random((n, bb.n)) < p).astype(np.int64)
    s = ((e @ bb.HZ.astype(np.int64).T) % 2).astype(np.uint8)
    return s, e.astype(np.uint8)


def relay_eval(bb, s, e, p):
    from relay_bp import RelayDecoderF32
    H = sp.csr_matrix(bb.HZ.astype(np.uint8))
    L = bb.logicals_Z().astype(np.int64)
    priors = np.full(bb.n, p, dtype=np.float64)
    dec = RelayDecoderF32(H, error_priors=priors,
                          gamma0=0.65, pre_iter=80, num_sets=60, set_max_iter=60,
                          gamma_dist_interval=(-0.24, 0.66), stop_nconv=5)
    N = s.shape[0]; k = L.shape[0]
    fl = 0; blk = 0; sm = 0
    Hd = bb.HZ.astype(np.int64)
    for i in range(N):
        c = np.asarray(dec.decode(s[i].astype(np.uint8))).astype(np.int64)
        if (((c @ Hd.T) % 2) == s[i]).all():
            sm += 1
        r = (e[i].astype(np.int64) ^ c) % 2
        f = (L @ r) % 2
        fl += int(f.sum()); blk += int(f.any())
    return fl, blk, N, k, sm


def main():
    bb = BBCode(); p = 0.03; n_train = 20000; seeds = [1, 2, 3]
    s_te, e_te = gen(bb, p, 30000, 99999)
    out = {'p': p, 'subset': 'novel (per-seed mask), pooled', 'seeds': seeds,
           'neural_equiv_novel': 0.03479, 'bposd60_novel_seed1': 0.05604,
           'relay_config': 'RelayDecoderF32 gamma0=0.65 pre80 sets60 setmax60 (-0.24,0.66) stop5',
           'per_seed': []}
    t0 = time.time()
    # full-test relay for reference
    flf, blkf, Nf, kf, smf = relay_eval(bb, s_te, e_te, p)
    plf = wilson_ci(flf, Nf * kf)
    out['relay_full'] = {'per_logical': plf[0], 'pl_ci': [plf[1], plf[2]],
                         'block': wilson_ci(blkf, Nf)[0], 'synd_match': round(smf / Nf, 4), 'n': Nf}
    print(f"relay FULL per-logical {plf[0]:.5f} synd_match {smf/Nf:.4f} ({time.time()-t0:.0f}s)")
    json.dump(out, open(os.path.join(_OUT, 'relay_novel.json'), 'w'), indent=2)

    pooled_fl = pooled_n = 0
    for sd in seeds:
        s_tr, _ = gen(bb, p, n_train, 1000 * sd + 1)
        trset = set(map(tuple, s_tr.tolist()))
        mask = np.array([tuple(row) not in trset for row in s_te.tolist()])
        fl, blk, N, k, sm = relay_eval(bb, s_te[mask], e_te[mask], p)
        pl = wilson_ci(fl, N * k)
        out['per_seed'].append({'seed': sd, 'n_novel': int(N), 'per_logical': pl[0],
                                'pl_ci': [pl[1], pl[2]], 'block': wilson_ci(blk, N)[0],
                                'synd_match': round(sm / N, 4)})
        pooled_fl += fl; pooled_n += N * k
        print(f"  seed {sd}: relay novel per-logical {pl[0]:.5f}")
        json.dump(out, open(os.path.join(_OUT, 'relay_novel.json'), 'w'), indent=2)
    pooled = wilson_ci(pooled_fl, pooled_n)
    out['relay_novel_pooled'] = {'per_logical': pooled[0], 'pl_ci': [pooled[1], pooled[2]]}
    out['relay_novel_mean_over_seeds'] = round(float(np.mean([x['per_logical'] for x in out['per_seed']])), 5)
    out['equiv_neural_beats_relay_on_novel'] = bool(0.03479 < pooled[0])
    json.dump(out, open(os.path.join(_OUT, 'relay_novel.json'), 'w'), indent=2)
    print(f"RELAY novel pooled {pooled[0]:.5f} ; equiv neural 0.03479 "
          f"{'BEATS' if 0.03479 < pooled[0] else 'LOSES TO'} relay")
    print("WROTE relay_novel.json")


if __name__ == '__main__':
    main()
