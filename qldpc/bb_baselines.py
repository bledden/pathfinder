"""BP-OSD baseline LER on the [[72,12,6]] BB code under code-capacity noise.

Establishes the reference the group-equivariant neural decoder must beat (or match at far
fewer params / training samples — the sample-efficiency hypothesis). BB codes are
exactly what BP-OSD is designed for, so this is the honest bar.

Logical failure criterion (code-capacity, X errors detected by HZ, logical via Z-logicals L):
  decode syndrome s -> correction c with HZ c = s; residual r = e XOR c; FAIL if L r != 0 (any).
Writes JSON; numbers copied from arrays.
"""
import os
import json, sys, time
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
from bb_code import BBCode
from ldpc import bposd_decoder

from _util import wilson_ci


def run(p, shots, seed=0):
    bb = BBCode()
    rng = np.random.default_rng(seed)
    H = bb.HZ.astype(np.uint8)            # X errors detected by Z-checks
    L = bb.logicals_Z().astype(np.uint8)  # (k=12, n) — fixed, exactly k independent reps
    k = L.shape[0]
    e = (rng.random((shots, bb.n)) < p).astype(np.uint8)
    s = (e @ H.T) % 2
    dec = bposd_decoder(H, error_rate=p, max_iter=bb.n, bp_method='ms',
                        osd_method='osd_cs', osd_order=7)
    t0 = time.time()
    block_fails = 0           # ANY of the k logicals flips (block error)
    per_logical_fails = 0     # count over (shot, logical) pairs
    for i in range(shots):
        c = dec.decode(s[i].astype(np.uint8))
        r = (e[i] ^ c.astype(np.uint8)) % 2
        flips = (L @ r) % 2   # (k,) which logicals flipped
        nf = int(flips.sum())
        per_logical_fails += nf
        if nf:
            block_fails += 1
    dt = time.time() - t0
    # per-logical LER = the literature-standard metric (Bravyi et al report per-logical)
    pl_ler, pl_lo, pl_hi = wilson_ci(per_logical_fails, shots * k)
    bl_ler, bl_lo, bl_hi = wilson_ci(block_fails, shots)
    return dict(p=p, shots=shots, k=k,
                per_logical_ler=pl_ler, per_logical_ci=[pl_lo, pl_hi],
                block_ler=bl_ler, block_ci=[bl_lo, bl_hi],
                sec_per_shot=round(dt / shots, 5))


def main():
    out = {'code': '[[72,12,6]] BB', 'decoder': 'BP-OSD (ms, osd_cs order7)', 'noise': 'code-capacity X', 'points': {}}
    for p in [0.02, 0.03, 0.05, 0.07]:
        r = run(p, 20000, seed=1)
        out['points'][f'p{p}'] = r
        json.dump(out, open(os.path.join(_OUT, 'bb_bposd_baseline.json'), 'w'), indent=2)
    print("WROTE bb_bposd_baseline.json")


if __name__ == '__main__':
    main()
