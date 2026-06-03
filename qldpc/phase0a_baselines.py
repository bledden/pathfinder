"""Phase 0a — the kill-or-confirm gate. Tuned CLASSICAL baselines on the validated circuit-level
BB DEM, the regime where neural wins are claimed. If tuned classical (BP-OSD order-10 + Relay-BP)
matches/eats the published transformer-BB (2504.13043) LER, the published win was over weak
baselines and the program stops. If tuned classical loses materially, real room exists -> push.

Decoders (all CPU):
  - BP-OSD order-0  (weak baseline reference — what many papers compare against)
  - BP-OSD order-10 combination-sweep (tuned)
  - Relay-BP (relay_bp RelayDecoderF32, IBM's real-time BP SOTA)

Metric: per-shot LOGICAL failure (any of the k=12 observables wrong) = block LER, Wilson CI.
We decode each observable's prediction from the decoder's correction via the DEM's observable map.

NOTE: transformer-BB's published LER is NOT hardcoded — it must be filled from the paper. We emit
'published_transformerBB_LER': null with a TODO so no fabricated comparison enters the record.

Modes:
  pilot  — small shots, all 3 decoders, all p, time each -> decide full-run feasibility
  full   — 1e6 shots @ p=0.001 (and scaled at higher p), the real gate
"""
import os
import json, sys, time, argparse
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
import stim
from bb_code import BBCode
from bb_circuit import build_z_memory
from ldpc import BpOsdDecoder
from _util import wilson_ci

import scipy.sparse as sp


def dem_to_matrices(dem):
    """Build detector-check matrix H (dets x errors), observable matrix O (obs x errors),
    and prior p per error, from a stim DEM (no decomposition)."""
    n_det = dem.num_detectors
    n_obs = dem.num_observables
    rows_d, cols_d, rows_o, cols_o, priors = [], [], [], [], []
    e = 0
    def handle(instr):
        nonlocal e
        for t in instr.targets_copy():
            pass
    for instr in dem.flattened():
        if instr.type == "error":
            p = instr.args_copy()[0]
            dets = []
            obs = []
            for t in instr.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    obs.append(t.val)
            for d in dets:
                rows_d.append(d); cols_d.append(e)
            for o in obs:
                rows_o.append(o); cols_o.append(e)
            priors.append(p)
            e += 1
    n_err = e
    H = sp.csc_matrix((np.ones(len(rows_d), dtype=np.uint8), (rows_d, cols_d)), shape=(n_det, n_err))
    O = sp.csc_matrix((np.ones(len(rows_o), dtype=np.uint8), (rows_o, cols_o)), shape=(n_obs, n_err))
    return H, O, np.array(priors)


def run_bposd(H, O, priors, det, obs, order):
    Hd = H.toarray().astype(np.uint8)
    Od = O.toarray().astype(np.uint8)
    dec = BpOsdDecoder(Hd, error_channel=list(priors), max_iter=30, bp_method='ms',
                       osd_method=('osd0' if order == 0 else 'osd_cs'), osd_order=order)
    N = det.shape[0]
    fails = 0
    for i in range(N):
        corr = dec.decode(det[i].astype(np.uint8))
        pred_obs = (Od @ corr) % 2
        if np.any(pred_obs != obs[i]):
            fails += 1
    ler, lo, hi = wilson_ci(fails, N)
    return dict(fails=fails, n=N, block_ler=ler, ci=[lo, hi])


def run_relaybp(H, O, priors, det, obs):
    try:
        from relay_bp import RelayDecoderF32
    except Exception as e:
        return {'error': f'relay_bp import: {e}'}
    Hd = H.astype(np.float32) if sp.issparse(H) else sp.csr_matrix(H)
    try:
        dec = RelayDecoderF32(
            sp.csr_matrix(H).astype(np.uint8),
            error_priors=np.asarray(priors, dtype=np.float64),
            gamma0=0.65, pre_iter=80, num_sets=60, set_max_iter=60,
            gamma_dist_interval=(-0.24, 0.66), stop_nconv=5,
        )
    except Exception as e:
        return {'error': f'relay ctor: {type(e).__name__}: {str(e)[:160]}'}
    Od = O.toarray().astype(np.uint8)
    N = det.shape[0]; fails = 0
    for i in range(N):
        try:
            corr = dec.decode(det[i].astype(np.uint8))
        except Exception as e:
            return {'error': f'relay decode: {str(e)[:160]}'}
        pred_obs = (Od @ np.asarray(corr).astype(np.uint8)) % 2
        if np.any(pred_obs != obs[i]):
            fails += 1
    ler, lo, hi = wilson_ci(fails, N)
    return dict(fails=fails, n=N, block_ler=ler, ci=[lo, hi])


def sample(circ, shots, seed):
    sampler = circ.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots, separate_observables=True)
    return det.astype(np.uint8), obs.astype(np.uint8)


def main(mode):
    bb = BBCode()
    ps = [0.003, 0.002, 0.001]   # cheap/fast points first -> early checkpoints; 1M @ p=0.001 last
    shots_map = {'pilot': {0.001: 2000, 0.002: 2000, 0.003: 2000},
                 'full':  {0.001: 1000000, 0.002: 300000, 0.003: 100000}}[mode]
    out = {'mode': mode, 'code': '[[72,12,6]]', 'rounds': bb.l, 'k': 12,
           'x_order': ['A0','A1','A2','B0','B1','B2'], 'z_order': ['B2','B1','B0','A2','A1','A0'],
           'published_transformerBB_LER': None,  # TODO fill from arXiv:2504.13043, do NOT fabricate
           'note': 'block LER = any of 12 logicals wrong; CPU; validated DEM (graphlike dist>=6)',
           'points': {}}
    for p in ps:
        circ = build_z_memory(bb, rounds=bb.l, p=p)
        dem = circ.detector_error_model(decompose_errors=False)
        H, O, priors = dem_to_matrices(dem)
        shots = shots_map[p]
        det, obs = sample(circ, shots, seed=1234 + int(p * 1e5))
        row = {'shots': shots, 'n_errors_dem': int(H.shape[1]), 'n_det': int(H.shape[0])}
        t = time.time(); row['bposd0'] = run_bposd(H, O, priors, det, obs, 0); row['bposd0']['s'] = round(time.time()-t, 1)
        t = time.time(); row['bposd10'] = run_bposd(H, O, priors, det, obs, 10); row['bposd10']['s'] = round(time.time()-t, 1)
        t = time.time(); row['relaybp'] = run_relaybp(H, O, priors, det, obs); row['relaybp']['s'] = round(time.time()-t, 1) if 'error' not in row['relaybp'] else None
        out['points'][f'p{p}'] = row
        json.dump(out, open(os.path.join(_OUT, 'phase0a_baselines.json'), 'w'), indent=2)
        b10 = row['bposd10']['block_ler']
        print(f"p={p} shots={shots}: BP-OSD0={row['bposd0']['block_ler']:.4f} BP-OSD10={b10:.4f} relay={row['relaybp'].get('block_ler','ERR')}")
    print("WROTE phase0a_baselines.json")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', nargs='?', default='pilot')
    main(ap.parse_args().mode)
