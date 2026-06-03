"""Orbit-structure analysis on the canonical Z-only DEM (the
both-checks DEM). Orbit structure of the decoding factor graph under the honest symmetry group:
spatial Z6xZ6 (global) + temporal +/-1 (bulk-valid). xy-swap and time-reversal EXCLUDED (verified
non-automorphisms; verified here). Reports: new tying factor, per-class
(bulk/cross/boundary) breakdown, generator automorphism audit.

Z-only DEM layout: detectors are Z-check ancilla measurements. R rounds x N=36 + final-readout block.
Same detector_round_pos / idx scheme as before (Z-only memory: N per round, final block at r=R).
"""
import os
import json, sys
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
import stim
from bb_code import BBCode
from bb_circuit import build_z_memory
from canon_dem import extract
from stepA_orbits import spatial_perm

R = 6
N = 36


def drp(d):
    return (R, d - R * N) if d >= R * N else (d // N, d % N)


def idx(r, q):
    return (R * N + q) if r == R else (r * N + q)


def spatial_detperm(a, b):
    sp = spatial_perm(a, b)
    P = np.empty(R * N + N, dtype=int)
    for d in range(R * N + N):
        r, q = drp(d); P[d] = idx(r, int(sp[q]))
    return P


def temporal_detperm(shift):
    P = -np.ones(R * N + N, dtype=int)
    for d in range(R * N + N):
        r, q = drp(d)
        if r == R:
            continue
        r2 = r + shift
        if 0 <= r2 < R:
            P[d] = idx(r2, q)
    return P


def xyswap_detperm():
    spm = np.array([(p % 6) * 6 + (p // 6) for p in range(N)], dtype=int)
    P = np.empty(R * N + N, dtype=int)
    for d in range(R * N + N):
        r, q = drp(d); P[d] = idx(r, int(spm[q]))
    return P


def timereversal_detperm():
    P = -np.ones(R * N + N, dtype=int)
    for d in range(R * N + N):
        r, q = drp(d)
        if r == R:
            continue
        P[d] = idx(R - 1 - r, q)
    return P


class UF:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def etype(H):
    rs = set(drp(d)[0] for d in H)
    if 0 in rs or R in rs:
        return 'boundary' if (rs <= {0} or rs <= {R}) else 'cross'
    return 'bulk'


def is_global_auto(detsets, P):
    for Hs in detsets:
        if any(P[d] < 0 for d in Hs):
            return False
        if frozenset(int(P[d]) for d in Hs) not in detsets:
            return False
    return True


def main():
    bb = BBCode(); p = 0.003
    c = build_z_memory(bb, rounds=R, p=p)
    dem = c.detector_error_model(decompose_errors=False)
    ex = extract(dem)
    detsets = set(Hs for (Hs, obs) in ex['edges'])
    out = {'dem': 'canonical Z-only', 'R': R, 'p': p, 'n_det': ex['n_det'],
           'n_hyperedges': ex['n_err'], 'n_unique_detsets': len(detsets)}

    # generator audit
    spat = [spatial_detperm(a, b) for a in range(6) for b in range(6)]
    out['spatial_Z6xZ6_global'] = f"{sum(is_global_auto(detsets, P) for P in spat)}/36"
    out['xyswap_global'] = bool(is_global_auto(detsets, xyswap_detperm()))
    out['timereversal_global'] = bool(is_global_auto(detsets, timereversal_detperm()))
    out['temporal_shift1_global'] = bool(is_global_auto(detsets, temporal_detperm(1)))

    # honest tying group = spatial(global) + temporal +/-1 (bulk-valid only)
    fg = set()
    for Hs in detsets:
        for d in Hs:
            fg.add((d, Hs))
    gens = list(spat)
    partial = [temporal_detperm(1), temporal_detperm(-1)]
    uf = UF()
    for (d, Hs) in fg:
        for P in gens:
            if any(P[x] < 0 for x in Hs): continue
            img = frozenset(int(P[x]) for x in Hs)
            if img in detsets and P[d] >= 0:
                uf.union((d, Hs), (int(P[d]), img))
        for P in partial:
            if any(P[x] < 0 for x in Hs): continue
            img = frozenset(int(P[x]) for x in Hs)
            if img in detsets and P[d] >= 0:
                uf.union((d, Hs), (int(P[d]), img))
    from collections import defaultdict
    reps = defaultdict(list)
    for x in fg:
        reps[uf.find(x)].append(x)
    obt = {'bulk': 0, 'cross': 0, 'boundary': 0}
    for root, members in reps.items():
        obt[etype(members[0][1])] += 1
    fbt = {'bulk': 0, 'cross': 0, 'boundary': 0}
    for (d, Hs) in fg:
        fbt[etype(Hs)] += 1
    n_orb = len(reps)
    out['n_fg_edges'] = len(fg)
    out['n_orbits'] = n_orb
    out['fg_by_type'] = fbt
    out['orbits_by_type'] = obt
    out['tying_total'] = round(len(fg) / n_orb, 2)
    out['tying_bulk'] = round(fbt['bulk'] / obt['bulk'], 2) if obt['bulk'] else None
    out['tying_cross'] = round(fbt['cross'] / obt['cross'], 2) if obt['cross'] else None
    out['tying_boundary'] = round(fbt['boundary'] / obt['boundary'], 2) if obt['boundary'] else None
    T = 12
    out['params_equiv_T12'] = 2 * n_orb * T
    out['params_free_T12'] = 2 * len(fg) * T
    out['vs_prev_bothchecks'] = {'old_tying_total': 109.6, 'old_n_orbits': 138, 'old_n_fg': 15120}
    out['GATE_tying_ge_10x'] = bool(out['tying_total'] >= 10)
    json.dump(out, open(os.path.join(_OUT, 'stepA_canon.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
