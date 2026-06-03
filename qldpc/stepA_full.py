"""Orbit-structure analysis under the full available symmetry group,
not just spatial. Prior run tied weights spatially-only (36x) and ignored: bulk temporal tying, and
two additional candidate symmetries (time-reversal, x<->y / A<->B code automorphism).

Method: union-find over factor-graph edges (detector d, hyperedge detector-set H). Connect (d,H) to
(P[d], P(H)) for every generator P that is LOCALLY valid for that edge (i.e. P(H) exists as a
hyperedge detector-set). Generators tested:
  - spatial Z6xZ6 (36, global)
  - temporal +1 shift (partial; ties interior bulk rounds)
  - x<->y position swap (the A<->B code automorphism: under x<->y, A=x^3+y+y^2 <-> y^3+x+x^2=B)
  - time-reversal r -> R-1-r (partial)
Each generator is first tested as a topology automorphism (does it map the hyperedge SET onto itself
where defined); orbits are then the union-find classes. Tying factor = fg-edges / orbits, by type.
"""
import os
import json, sys
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
import stim
from bb_code import BBCode
from bb_circuit import build_z_memory
from stepA_orbits import parse_dem, spatial_perm, detector_round_pos

R = 6
N = 36


def det_index(r, p):
    return (R * N + p) if r == R else (r * N + p)


def spatial_detperm(a, b):
    sp = spatial_perm(a, b)
    P = np.empty(R * N + N, dtype=int)
    for d in range(R * N + N):
        r, p = detector_round_pos(d)
        P[d] = det_index(r, int(sp[p]))
    return P


def xyswap_detperm():
    """x<->y torus transpose: position (i,j)->(j,i); same on every round + final block."""
    sp = np.array([(p % 6) * 6 + (p // 6) for p in range(N)], dtype=int)
    P = np.empty(R * N + N, dtype=int)
    for d in range(R * N + N):
        r, p = detector_round_pos(d)
        P[d] = det_index(r, int(sp[p]))
    return P


def temporal_detperm(shift):
    """(r,p)->(r+shift,p) for syndrome rounds; final block and out-of-range map to -1 (invalid)."""
    P = -np.ones(R * N + N, dtype=int)
    for d in range(R * N + N):
        r, p = detector_round_pos(d)
        if r == R:
            continue
        r2 = r + shift
        if 0 <= r2 < R:
            P[d] = det_index(r2, p)
    return P


def timereversal_detperm():
    """(r,p)->(R-1-r,p) for syndrome rounds; final block invalid."""
    P = -np.ones(R * N + N, dtype=int)
    for d in range(R * N + N):
        r, p = detector_round_pos(d)
        if r == R:
            continue
        P[d] = det_index(R - 1 - r, p)
    return P


def is_topo_auto(edges_detsets, P):
    """Global topology automorphism: every hyperedge detset maps (fully defined) onto a detset."""
    for H in edges_detsets:
        if any(P[d] < 0 for d in H):
            return False
        if frozenset(int(P[d]) for d in H) not in edges_detsets:
            return False
    return True


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


def edge_type(H):
    rs = set(detector_round_pos(d)[0] for d in H)
    if 0 in rs or R in rs:
        return 'boundary' if (rs <= {0} or rs <= {R}) else 'cross-boundary'
    return 'bulk'


def main():
    bb = BBCode(); p = 0.003
    circ = build_z_memory(bb, rounds=R, p=p)
    dem = circ.detector_error_model(decompose_errors=False)
    edges = parse_dem(dem)
    detsets = set(d for (d, o, pr) in edges)
    out = {'code': '[[72,12,6]]', 'p': p, 'rounds': R, 'n_detectors': dem.num_detectors,
           'n_unique_hyperedges': len(detsets)}

    # --- which generators are (global) topology automorphisms ---
    gens = {}  # name -> (perm, is_global_auto)
    spat = [spatial_detperm(a, b) for a in range(6) for b in range(6)]
    gens['spatial_Z6xZ6'] = (None, sum(is_topo_auto(detsets, P) for P in spat))
    xy = xyswap_detperm()
    gens['xy_swap_AtoB'] = (xy, is_topo_auto(detsets, xy))
    trev = timereversal_detperm()
    gens['time_reversal'] = (trev, is_topo_auto(detsets, trev))  # global (partial perm) -> will be False
    tshift1 = temporal_detperm(1)
    gens['temporal_shift1_global'] = (tshift1, is_topo_auto(detsets, tshift1))
    out['generator_global_automorphism'] = {
        'spatial_Z6xZ6': f"{gens['spatial_Z6xZ6'][1]}/36",
        'xy_swap_AtoB': bool(gens['xy_swap_AtoB'][1]),
        'time_reversal_global': bool(gens['time_reversal'][1]),
        'temporal_shift1_global': bool(gens['temporal_shift1_global'][1]),
    }

    # --- union-find orbits over fg-edges under ALL locally-valid generators ---
    # generator list for union-find: 36 spatial (global), + xy if global, + temporal+/-1 (partial),
    # + time-reversal (partial). Partial generators only connect where image hyperedge exists.
    uf_gens = list(spat)
    if gens['xy_swap_AtoB'][1]:
        uf_gens.append(xy)
    partial_gens = [temporal_detperm(1), temporal_detperm(-1), timereversal_detperm()]

    fg = set()
    for H in detsets:
        for d in H:
            fg.add((d, H))
    uf = UF()
    def valid_image(P, H):
        if any(P[d] < 0 for d in H):
            return None
        img = frozenset(int(P[d]) for d in H)
        return img if img in detsets else None
    for (d, H) in fg:
        for P in uf_gens:                      # global gens
            img = valid_image(P, H)
            if img is not None:
                uf.union((d, H), (int(P[d]), img))
        for P in partial_gens:                 # partial gens (temporal, time-reversal)
            img = valid_image(P, H)
            if img is not None and P[d] >= 0:
                uf.union((d, H), (int(P[d]), img))

    # count orbits by type (type by representative's hyperedge)
    from collections import defaultdict
    orbit_reps = defaultdict(list)
    for (d, H) in fg:
        orbit_reps[uf.find((d, H))].append((d, H))
    orbits_by_type = {'bulk': 0, 'boundary': 0, 'cross-boundary': 0}
    for root, members in orbit_reps.items():
        # type = type of the hyperedge in the representative
        t = edge_type(members[0][1])
        orbits_by_type[t] += 1
    n_orbits = len(orbit_reps)
    fg_by_type = {'bulk': 0, 'boundary': 0, 'cross-boundary': 0}
    for (d, H) in fg:
        fg_by_type[edge_type(H)] += 1

    out['n_factor_graph_edges'] = len(fg)
    out['n_edge_orbits_total'] = n_orbits
    out['fg_edges_by_type'] = fg_by_type
    out['orbits_by_type'] = orbits_by_type
    out['tying_factor_total'] = round(len(fg) / n_orbits, 2)
    out['tying_factor_bulk'] = round(fg_by_type['bulk'] / orbits_by_type['bulk'], 2) if orbits_by_type['bulk'] else None
    out['tying_factor_boundary'] = round(fg_by_type['boundary'] / orbits_by_type['boundary'], 2) if orbits_by_type['boundary'] else None
    out['tying_factor_cross'] = round(fg_by_type['cross-boundary'] / orbits_by_type['cross-boundary'], 2) if orbits_by_type['cross-boundary'] else None
    T = 12
    out['params_free_T12'] = 2 * len(fg) * T
    out['params_equiv_T12'] = 2 * n_orbits * T
    out['spatial_only_orbits_prev'] = 420  # from stepA_orbits.json, for comparison
    json.dump(out, open(os.path.join(_OUT, 'stepA_full.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2)); print("WROTE stepA_full.json")


if __name__ == '__main__':
    main()
