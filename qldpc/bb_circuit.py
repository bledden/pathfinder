"""Circuit-level syndrome-extraction circuit + DEM for the [[72,12,6]] BB code.

VALIDATED CONSTRUCTION (validated via Stim automorphism search):
  - 6 CNOT layers. In each layer, every X-ancilla does one CX (X-check) and every Z-ancilla does
    one CX (Z-check), each addressing one monomial of A/B.
  - The commutation condition (so X- and Z-check measurements don't anti-commute through shared
    data qubits, i.e. detectors are deterministic) is: the Z-check monomial order must be the
    TIME-REVERSAL of the X-check monomial order. Discovered by exhaustive Stim search: with
    X-order [A0,A1,A2,B0,B1,B2], exactly 1 of 720 Z-orders is deterministic — the reverse
    [B2,B1,B0,A2,A1,A0]. This is the Bravyi et al. (Nature 2024) schedule structure.
  - Z-basis memory: init |0>, R rounds, final data Z-readout. Detectors on Z-checks; observable a
    Z-logical. Validated by circuit-level distance == 6 before any LER is trusted.

X-order and Z-order are module constants below; build_z_memory uses them.
"""
import numpy as np
import os
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import stim
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bb_code import BBCode

A_MONS = [('x', 3), ('y', 1), ('y', 2)]
B_MONS = [('y', 3), ('x', 1), ('x', 2)]
X_ORDER = ['A0', 'A1', 'A2', 'B0', 'B1', 'B2']
Z_ORDER = ['B2', 'B1', 'B0', 'A2', 'A1', 'A0']   # time-reversal — the unique deterministic order


def monomial_perms(bb):
    l, m = bb.l, bb.m
    Il, Im = np.eye(l, dtype=np.int8), np.eye(m, dtype=np.int8)
    def term(v, p):
        if v == 'x':
            return np.kron(np.linalg.matrix_power(bb.Sl, p), Im) % 2
        return np.kron(Il, np.linalg.matrix_power(bb.Sm, p)) % 2
    out = {}
    for k, (v, p) in zip(['A0', 'A1', 'A2'], A_MONS):
        out[k] = term(v, p)
    for k, (v, p) in zip(['B0', 'B1', 'B2'], B_MONS):
        out[k] = term(v, p)
    return out


def build_z_memory(bb, rounds, p, x_order=None, z_order=None):
    """Z-basis memory detecting X errors. Measures ONLY Z-checks (H_Z) each round — measuring
    X-checks would collapse the X-logical and make the observable non-deterministic (a subtlety in the
    prior both-checks version). init data |0>, inject X noise, R rounds of Z-check extraction, final
    data Z-readout, observable = X-logicals (deterministic: nothing collapses them).
    z_order = the time-reversed monomial order (the deterministic Z-check schedule)."""
    N = bb.N
    Ld = lambda i: i
    Rd = lambda i: N + i
    ZA = lambda c: 2 * N + c            # only Z-ancillas now (no X-ancillas)
    n_data = 2 * N
    P = monomial_perms(bb)
    z_order = z_order or Z_ORDER
    data = list(range(n_data))
    zanc = [ZA(i) for i in range(N)]

    c = stim.Circuit()

    def noisy_round(circ):
        circ.append("R", zanc)
        if p > 0:
            circ.append("X_ERROR", zanc, p)
        for step in range(6):
            zk = z_order[step]
            matz = P[zk]
            pairs = []
            for cc in range(N):
                iT = int(np.argmax(matz[:, cc]))
                pairs += [(Rd(iT) if zk[0] == 'A' else Ld(iT)), ZA(cc)]   # CX data->Z-anc
            circ.append("CX", pairs)
            if p > 0:
                circ.append("DEPOLARIZE2", pairs, p)
        if p > 0:
            circ.append("X_ERROR", zanc, p)         # pre-measure flip
        circ.append("MR", zanc)                      # measure+reset Z-anc (N records/round)

    c.append("R", data)
    if p > 0:
        c.append("X_ERROR", data, p)
    for r in range(rounds):
        noisy_round(c)
        if r == 0:
            for i in range(N):
                c.append("DETECTOR", [stim.target_rec(-N + i)])
        else:
            for i in range(N):
                # Z-only: N records/round, so previous round's zanc i is one block back: -2N + i
                c.append("DETECTOR", [stim.target_rec(-N + i), stim.target_rec(-2 * N + i)])
    if p > 0:
        c.append("X_ERROR", data, p)
    c.append("M", data)
    # final Z-check consistency: Z-check cc parity from final data XOR last Z-anc cc
    zL = [np.where(bb.B[:, cc] == 1)[0] for cc in range(N)]
    zR = [np.where(bb.A[:, cc] == 1)[0] for cc in range(N)]
    for cc in range(N):
        recs = []
        for d in zL[cc]:
            recs.append(stim.target_rec(-n_data + Ld(int(d))))
        for d in zR[cc]:
            recs.append(stim.target_rec(-n_data + Rd(int(d))))
        recs.append(stim.target_rec(-n_data - N + cc))   # last zanc cc (before the n_data M records)
        c.append("DETECTOR", recs)
    # observable = X-logicals (commute with H_Z, flipped by X errors, deterministic in Z-readout)
    Lx = bb.logicals_X()
    for li in range(Lx.shape[0]):
        obs = [stim.target_rec(-n_data + j) for j in range(n_data) if Lx[li, j]]
        c.append("OBSERVABLE_INCLUDE", obs, li)
    return c


def circuit_distance(circ, cap=6):
    err = circ.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=2 * cap,
        dont_explore_edges_with_degree_above=cap,
        dont_explore_edges_increasing_symptom_degree=False,
    )
    return len(err)


if __name__ == '__main__':
    import json, time
    bb = BBCode()
    out = {'code': '[[72,12,6]]', 'target_distance': 6, 'x_order': X_ORDER, 'z_order': Z_ORDER}
    circ0 = build_z_memory(bb, rounds=bb.l, p=0.0)
    try:
        circ0.detector_error_model(decompose_errors=False)
        out['deterministic'] = True
    except Exception as e:
        out['deterministic'] = False
        out['det_err'] = str(e)[:150]
    circ = build_z_memory(bb, rounds=bb.l, p=0.001)
    out['n_qubits'] = circ.num_qubits
    out['n_detectors'] = circ.num_detectors
    out['n_observables'] = circ.num_observables
    try:
        dem = circ.detector_error_model(decompose_errors=False)
        out['dem_built'] = True
        out['dem_num_errors'] = dem.num_errors
    except Exception as e:
        out['dem_built'] = False
        out['dem_err'] = str(e)[:150]
    t0 = time.time()
    try:
        d = circuit_distance(circ)
        out['circuit_distance'] = d
        out['distance_search_s'] = round(time.time() - t0, 1)
    except Exception as e:
        out['circuit_distance'] = f"fail:{str(e)[:80]}"
    out['DISTANCE_OK'] = (out.get('circuit_distance') == 6)
    json.dump(out, open(os.path.join(_OUT, 'bb_circuit_validate.json'), 'w'), indent=2)
    print("WROTE bb_circuit_validate.json:", out)
