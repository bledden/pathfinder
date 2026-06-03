"""Canonical DEM extraction — the SINGLE SOURCE OF TRUTH for all circuit-level decoding.

Every
circuit-level number is rebuilt on THIS module. No more hand-rolled build_H / sample_from_dem /
obsbit. Everything comes from stim's DEM via dem.flattened().

Provides:
  extract(dem) -> dict(H, Lo, priors, n_det, n_obs, n_err, edges)   [H: ndet x E csr, Lo: nobs x E]
  decode_bposd(H, Lo, priors, det_shots, obs_shots, max_iter, osd_order) -> block LER (+ Wilson CI)
  regression_test() -> asserts canonical BP-OSD-10 LER <= 0.2% at R=6, p=0.003, N>=10k

The ad-hoc functions in circ_neural_bp.py (build_factorgraph/sample_from_dem) and pod_test.build_H
are deprecated in favor of the canonical extraction below.
"""
import os, sys
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sp
import stim
from bb_code import BBCode
from bb_circuit import build_z_memory


def extract(dem):
    """Canonical hyperedge extraction from a stim DEM (decompose_errors=False)."""
    nd, no = dem.num_detectors, dem.num_observables
    rows_d, cols_d, rows_o, cols_o, priors, edges = [], [], [], [], [], []
    j = 0
    for inst in dem.flattened():
        if inst.type != 'error':
            continue
        pr = inst.args_copy()[0]
        dets, obs = [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
        for d in dets:
            rows_d.append(d); cols_d.append(j)
        for o in obs:
            rows_o.append(o); cols_o.append(j)
        priors.append(pr); edges.append((frozenset(dets), tuple(obs)))
        j += 1
    E = j
    H = sp.csr_matrix((np.ones(len(rows_d), np.uint8), (rows_d, cols_d)), shape=(nd, E))
    Lo = sp.csr_matrix((np.ones(len(rows_o), np.uint8), (rows_o, cols_o)), shape=(no, E))
    return dict(H=H, Lo=Lo, priors=np.array(priors), n_det=nd, n_obs=no, n_err=E, edges=edges)


def wilson(k, n, z=1.96):
    import math
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def decode_bposd(ex, det_shots, obs_shots, max_iter=30, osd_order=10, channel_probs=None):
    """Block LER via ldpc BP-OSD. channel_probs=None -> use DEM priors (canonical classical)."""
    from ldpc import BpOsdDecoder
    H = ex['H']; Lo = ex['Lo'].toarray().astype(np.int64)
    pri = ex['priors'] if channel_probs is None else channel_probs
    dec = BpOsdDecoder(H, error_channel=list(np.clip(pri, 1e-6, 1 - 1e-6)), max_iter=max_iter,
                       bp_method='ms', osd_method='osd_cs', osd_order=osd_order)
    N = det_shots.shape[0]; fails = 0
    for i in range(N):
        e = dec.decode(det_shots[i].astype(np.uint8))
        pred = (Lo @ e) % 2
        if not np.array_equal(pred.astype(np.uint8), obs_shots[i].astype(np.uint8)):
            fails += 1
    return wilson(fails, N), fails, N


def regression_test(N=10000, seed=11):
    """STANDING REGRESSION: canonical BP-OSD-10 must give block LER <= 0.2% at R=6, p=0.003.
    Any pipeline change reruns this; it catches the next harness bug at commit time."""
    bb = BBCode()
    c = build_z_memory(bb, rounds=6, p=0.003)
    dem = c.detector_error_model(decompose_errors=False)
    ex = extract(dem)
    det, obs = c.compile_detector_sampler(seed=seed).sample(N, separate_observables=True)
    (ler, lo, hi), fails, n = decode_bposd(ex, det, obs, max_iter=30, osd_order=10)
    ok = bool(ler <= 0.002)
    return dict(R=6, p=0.003, N=n, fails=fails, block_ler=round(ler, 6),
                ci=[round(lo, 6), round(hi, 6)], n_det=ex['n_det'], n_err=ex['n_err'],
                PASS=ok, threshold=0.002)


if __name__ == '__main__':
    import json
    r = regression_test()
    json.dump(r, open(os.path.join(_OUT, 'canon_regression.json'), 'w'), indent=2)
    print(json.dumps(r, indent=2))
    print("REGRESSION", "PASS" if r['PASS'] else "FAIL")
