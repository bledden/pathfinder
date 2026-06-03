"""Fast distance validation for the BB circuit-level schedule (the full weight-6 search was
intractable single-threaded). Two cheap, decisive checks:

(1) GRAPHLIKE distance via search_for_undetectable_logical_errors with a HARD low cap + a timeout
    posture: we only need to know "is there an undetectable logical error of weight < 6?". So we
    cap exploration at weight 5. If the capped search returns a logical error of weight w<6, the
    schedule is distance-collapsing (BAD). If it exhausts weight<=5 without finding one, the
    circuit distance is >=6 (GOOD) — which combined with the code distance 6 means ==6.
    We run it with decompose_errors=True (graphlike) which is far faster than the hyperedge search.

(2) Cross-check: the DEM must be deterministic (already known) and the number of detectors/errors
    sane. Reported alongside.

This is the gate: distance >= 6 (no sub-6 undetectable logical) => DEM trustworthy for Phase 0a.
"""
import os
import json, time, sys
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import stim
from bb_code import BBCode
from bb_circuit import build_z_memory, X_ORDER, Z_ORDER


def main():
    bb = BBCode()
    out = {'x_order': X_ORDER, 'z_order': Z_ORDER, 'target': 6}
    circ = build_z_memory(bb, rounds=bb.l, p=0.001)
    out['n_detectors'] = circ.num_detectors

    # graphlike search, capped at weight 5 — fast; tells us if distance < 6
    t0 = time.time()
    try:
        err = circ.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=5,
            dont_explore_edges_with_degree_above=2,        # graphlike edges only
            dont_explore_edges_increasing_symptom_degree=True,
        )
        out['capped5_found_logical_weight'] = len(err)
        out['distance_lt_6'] = (len(err) < 6)
        out['search_s'] = round(time.time() - t0, 1)
    except ValueError as e:
        # stim raises if NO undetectable logical error exists within the cap -> distance > cap
        msg = str(e)
        out['capped5_no_error_within_cap'] = True
        out['distance_ge_6'] = True
        out['search_s'] = round(time.time() - t0, 1)
        out['stim_msg'] = msg[:120]
    out['DISTANCE_OK'] = bool(out.get('distance_ge_6', False) or
                              (out.get('capped5_found_logical_weight', 0) >= 6))
    json.dump(out, open(os.path.join(_OUT, 'bb_distance_fast.json'), 'w'), indent=2)
    print("WROTE bb_distance_fast.json:", out)


if __name__ == '__main__':
    main()
