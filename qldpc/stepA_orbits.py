"""Step A (pre-scaling gate): does the equivariant parameter-efficiency survive to
circuit-level? Compute the DEM factor-graph symmetry/orbit structure for [[72,12,6]] circuit-level,
BEFORE any training or GPU.

Neural-BP ties learnable weights to EDGES of the decoding factor graph. At code-capacity that graph
is the code Tanner graph (36 checks x 72 vars), Z6xZ6 gives 6 orbits of 36 -> 163 params. At
circuit-level the graph is the DEM: detectors x error-mechanisms (hyperedges). The question is the
orbit structure of the DEM factor-graph edges under the symmetries that SURVIVE to circuit-level.

Method (same as the schedule search): the candidate symmetries are known a priori. For each, build
the induced DETECTOR permutation, and test whether it is a DEM automorphism (permutes the hyperedge
set onto itself, preserving detector-incidence and prior). Surviving symmetries = the tying group.

Detector layout (from bb_circuit.py build_z_memory, Z-memory, R=6):
  detectors 0..215  = bulk: round r in 0..5, position p in 0..35 -> index 36*r + p
  detectors 216..251 = final readout consistency, position p in 0..35 -> index 216 + p
  position p = (i,j), i=p//6, j=p%6 on the Z6xZ6 torus.

Outputs:
  1. hyperedge count by type (bulk / boundary / cross-boundary)
  2. which candidate symmetries are DEM automorphisms (the surviving group)
  3. factor-graph edge orbit structure, split bulk/boundary/cross-boundary
  4. param count equiv vs free (edges/iter tied to orbits vs untied)
  5. tying-factor breakdown
  6. (separate script) train-on-small-instance confirmation
Plus two additional candidate symmetries: time-reversal x CSS-swap, extra code automorphisms.
"""
import os
import json, sys, time
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
import stim
from bb_code import BBCode
from bb_circuit import build_z_memory

R = 6
N = 36  # checks per round / positions on torus


def spatial_perm(a, b):
    """Z6xZ6 element (a,b): position p=(i,j) -> ((i+a)%6, (j+b)%6). Returns perm over 0..35."""
    out = np.zeros(N, dtype=int)
    for p in range(N):
        i, j = p // 6, p % 6
        out[p] = ((i + a) % 6) * 6 + ((j + b) % 6)
    return out


def detector_round_pos(d):
    """(round, position) for detector index d. round=6 denotes the final readout block."""
    if d < R * N:
        return d // N, d % N
    return R, d - R * N   # final block tagged round=R


def induced_detector_perm(spatialp, time_shift=0):
    """Build the detector permutation from a spatial Z6xZ6 perm + optional temporal shift.
    Returns perm array P of length n_det with P[d] = image detector, or None if the map sends a
    detector off the lattice (e.g. temporal shift past the boundary)."""
    ndet = R * N + N
    P = -np.ones(ndet, dtype=int)
    for d in range(ndet):
        r, p = detector_round_pos(d)
        p2 = spatialp[p]
        if r == R:
            # final readout block: spatial only, no temporal shift applies
            if time_shift != 0:
                return None  # readout block has no temporal image
            P[d] = R * N + p2
        else:
            r2 = r + time_shift
            if r2 < 0 or r2 >= R:
                return None  # off the bulk -> this symmetry not globally defined
            P[d] = r2 * N + p2
    return P


def parse_dem(dem):
    """Return list of hyperedges: each = (frozenset(detectors), obs_bit, prior)."""
    edges = []
    for inst in dem.flattened():
        if inst.type != 'error':
            continue
        pr = inst.args_copy()[0]
        dets = []
        obs = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs ^= 1
        edges.append((frozenset(dets), obs, round(pr, 12)))
    return edges


def is_automorphism(edges, edge_index, P):
    """TOPOLOGY automorphism — the correct notion for neural-BP weight-tying. Weights tie to
    message-passing graph EDGES (detector<->hyperedge incidence); the per-edge prior is an INPUT
    (channel LLR) and the observable is read out by a separate position-aware head, so neither
    needs to be invariant. (The spatial group PERMUTES logical sectors, so the obs bit changes by
    construction — requiring observable-invariance would be incorrect here.) P is a DEM topology
    automorphism iff it maps every hyperedge's detector-set onto another hyperedge's detector-set."""
    detset = edge_index  # set of frozenset(detectors)
    for dets, obs, pr in edges:
        mapped = frozenset(int(P[d]) for d in dets)
        if mapped not in detset:
            return False
    return True


def main():
    bb = BBCode()
    p = 0.003
    circ = build_z_memory(bb, rounds=R, p=p)
    dem = circ.detector_error_model(decompose_errors=False)
    ndet = dem.num_detectors
    edges = parse_dem(dem)
    # topology automorphism keys on detector-SETS (weights tie to graph edges; prior+obs are not
    # invariant — the spatial group permutes logical sectors, and priors are channel inputs).
    edge_index = set(dets for (dets, o, pr) in edges)
    out = {'code': '[[72,12,6]]', 'p': p, 'rounds': R, 'n_detectors': ndet, 'n_hyperedges': len(edges),
           'note': 'topology automorphism (weight-tying notion): keyed on detector-sets, NOT prior/obs'}

    # --- (1) hyperedge classification by which rounds its detectors span ---
    def edge_type(dets):
        rounds = set(detector_round_pos(d)[0] for d in dets)
        has_init = (0 in rounds)
        has_final = (R in rounds)
        bulk_rounds = [r for r in rounds if 1 <= r <= R - 1]
        if has_init or has_final:
            if rounds <= {0} or rounds <= {R}:
                return 'boundary'
            return 'cross-boundary'
        return 'bulk'
    types = {}
    for dets, o, pr in edges:
        t = edge_type(dets)
        types[t] = types.get(t, 0) + 1
    out['hyperedge_by_type'] = types

    # --- (2) which candidate symmetries are DEM automorphisms ---
    # (a) spatial Z6xZ6, per-round (time_shift=0)
    spatial_ok = 0
    for a in range(6):
        for b in range(6):
            P = induced_detector_perm(spatial_perm(a, b), time_shift=0)
            if P is not None and is_automorphism(edges, edge_index, P):
                spatial_ok += 1
    out['spatial_Z6xZ6_automorphisms'] = f"{spatial_ok}/36"
    # (b) temporal translation by +k (bulk only; will fail globally because final/init differ)
    temporal_ok = []
    for k in range(1, R):
        P = induced_detector_perm(spatial_perm(0, 0), time_shift=k)
        ok = (P is not None) and is_automorphism(edges, edge_index, P)
        temporal_ok.append((k, bool(P is not None), bool(ok)))
    out['temporal_shift_global'] = [{'k': k, 'defined': d, 'is_automorphism': o} for (k, d, o) in temporal_ok]

    # (c) spatial x temporal combined, all (a,b,k)
    spacetime_ok = 0
    spacetime_tot = 0
    for a in range(6):
        for b in range(6):
            for k in range(R):
                P = induced_detector_perm(spatial_perm(a, b), time_shift=k)
                if P is None:
                    continue
                spacetime_tot += 1
                if is_automorphism(edges, edge_index, P):
                    spacetime_ok += 1
    out['spacetime_automorphisms'] = f"{spacetime_ok}/{spacetime_tot} (globally-defined candidates)"

    # --- (3-5) orbit structure of factor-graph EDGES under the surviving group ---
    # factor-graph edge = (detector, hyperedge_id) incidence. Build them, then orbit under the
    # surviving symmetry group (spatial automorphisms; + temporal if any survived globally).
    # Surviving group = spatial autos that passed (and we also tie over the per-round structure:
    # a bulk hyperedge in round r maps to the same orbit as its spatial image in round r).
    surviving = []
    for a in range(6):
        for b in range(6):
            P = induced_detector_perm(spatial_perm(a, b), 0)
            if P is not None and is_automorphism(edges, edge_index, P):
                surviving.append((a, b, P))
    # TOPOLOGY orbit structure. A hyperedge's identity for tying is its detector-SET (a unique
    # message-passing node); multiple DEM errors can share one detector-set (different prior/obs) but
    # are the same factor-graph node. An fg-edge = (detector d, hyperedge-detset H). Symmetry g maps
    # (d, H) -> (P[d], P(H)). Orbits of fg-edges = the weight-tying classes.
    detsets = sorted({dets for (dets, o, pr) in edges}, key=lambda s: tuple(sorted(s)))
    n_unique_hyperedges = len(detsets)
    out['n_unique_hyperedge_detsets'] = n_unique_hyperedges
    fg_edges = set()
    for H in detsets:
        for d in H:
            fg_edges.add((d, H))
    def mapP(P, H):
        return frozenset(int(P[d]) for d in H)
    seen = set()
    orbits_by_type = {'bulk': 0, 'boundary': 0, 'cross-boundary': 0}
    orbit_total = 0
    for (d, H) in fg_edges:
        if (d, H) in seen:
            continue
        orb = set()
        for (a, b, P) in surviving:
            orb.add((int(P[d]), mapP(P, H)))
        seen |= orb
        t = edge_type(H)
        orbits_by_type[t] = orbits_by_type.get(t, 0) + 1
        orbit_total += 1
    nfg = len(fg_edges)
    out['n_factor_graph_edges'] = nfg
    out['n_edge_orbits_total'] = orbit_total
    out['n_edge_orbits_by_type'] = orbits_by_type
    out['tying_factor_total'] = round(nfg / orbit_total, 2) if orbit_total else None
    fg_by_type = {'bulk': 0, 'boundary': 0, 'cross-boundary': 0}
    for (d, H) in fg_edges:
        fg_by_type[edge_type(H)] += 1
    out['fg_edges_by_type'] = fg_by_type
    out['tying_factor_bulk'] = round(fg_by_type['bulk'] / orbits_by_type['bulk'], 2) if orbits_by_type.get('bulk') else None
    out['n_surviving_spatial'] = len(surviving)

    # rough param-count proxy: neural-BP has ~2 weights per fg-edge per iteration (check->var, var->check)
    T_iter = 12
    out['params_free_per_iter'] = 2 * len(fg_edges)
    out['params_equiv_per_iter'] = 2 * orbit_total
    out['params_free_total_T12'] = 2 * len(fg_edges) * T_iter
    out['params_equiv_total_T12'] = 2 * orbit_total * T_iter

    json.dump(out, open(os.path.join(_OUT, 'stepA_orbits.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2))
    print("WROTE stepA_orbits.json")


if __name__ == '__main__':
    main()
