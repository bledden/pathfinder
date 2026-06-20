"""Correctness gate: soft detectors thresholded at 0.5 must EXACTLY equal Stim hard detectors.
Run from anywhere; uses absolute path to the soft modules."""
import sys
sys.path.insert(0, '/Users/bledden/Documents/pathfinder/coda_experiments')
import numpy as np, stim
from qiskit_ibm_runtime import QiskitRuntimeService
from soft_info import calibrate_iq, soft_detectors_from_p1
from soft_detmap import detector_to_measurements

svc = QiskitRuntimeService()
rec = np.asarray(svc.job(open('/tmp/soft_jobid.txt').read().strip()).result()[0].data.rec)  # (200,17)
p1, meta = calibrate_iq(rec)
hard_from_soft = (p1 > 0.5).astype(bool)
D, R = 3, 1
clean = stim.Circuit.generated('surface_code:rotated_memory_z', distance=D, rounds=R)
det_stim, _ = clean.compile_m2d_converter().convert(measurements=hard_from_soft, separate_observables=True)
dtm, nm, nd, _ = detector_to_measurements(D, R)
soft_det = soft_detectors_from_p1(p1, dtm)
det_soft_hard = (soft_det > 0.5)
match = bool((det_stim == det_soft_hard).all())
agree = float((det_stim == det_soft_hard).mean())
conf = float((np.abs(soft_det - 0.5) * 2).mean())
band = float(((soft_det > 0.1) & (soft_det < 0.9)).mean())
out = (f"GATE soft@0.5 == Stim hard detectors: {match}  (agreement={agree:.5f})\n"
       f"  mean soft confidence |p-.5|*2 = {conf:.4f}  uncertain-band[0.1,0.9] frac = {band:.4f}\n")
open('/tmp/soft_gate.txt', 'w').write(out)
print(out)
