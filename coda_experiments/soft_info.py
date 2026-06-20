"""Soft-information pipeline: per-shot kerneled IQ -> soft measurement LLRs -> soft detectors.

Method (Pattison et al. 2021, "Improved QEC using soft information"; AlphaQubit uses the
same idea): each measurement's IQ point is scored against a per-qubit 2-Gaussian model of
the |0> and |1> readout blobs, giving P(meas=1). A detector is a mod-2 sum of measurements;
its SOFT value is the probabilistic XOR of its constituent measurement probabilities. We
expose both the soft detector probability (for the NN, float in [0,1]) and the hard detector
(threshold at 0.5, == what the standard pipeline produces) so the two are directly comparable.
"""
import numpy as np
from sklearn.mixture import GaussianMixture


def calibrate_iq(rec):
    """rec: (shots, n_meas) complex. Returns per-measurement GMM + which component is |1>.
    Self-calibration: the LESS-populated cluster is labeled |1> (ancillas/data sit mostly in
    |0> over a memory experiment; excited population is the minority). Returns p1[shots,n_meas]."""
    n_shot, n_meas = rec.shape
    p1 = np.zeros((n_shot, n_meas), dtype=np.float64)
    meta = []
    for j in range(n_meas):
        pts = np.column_stack([rec[:, j].real, rec[:, j].imag]).astype(np.float64)
        # normalize IQ scale (raw values ~1e8 overflow the GMM and make it non-deterministic)
        s = np.abs(pts).max()
        if s > 0:
            pts = pts / s
        g = GaussianMixture(2, covariance_type='full', random_state=0, n_init=2).fit(pts)
        post = g.predict_proba(pts)          # (shots,2)
        # label the minority-weight component as |1>
        one = int(np.argmin(g.weights_))
        p1[:, j] = post[:, one]
        meta.append(dict(weights=g.weights_.tolist(), one_comp=one,
                         sep=float(np.linalg.norm(g.means_[0]-g.means_[1]))))
    return p1, meta


def p1_to_llr(p1, eps=1e-6):
    """soft measurement LLR = log P(0)/P(1). Large +ve = confident 0."""
    p1 = np.clip(p1, eps, 1-eps)
    return np.log((1-p1)/p1)


def soft_xor(prob_list):
    """Probabilistic XOR of independent bits: P(parity=1).
    For bits with P(b_i=1)=p_i, P(parity odd) = (1 - prod(1-2p_i))/2."""
    prod = np.ones_like(prob_list[0])
    for p in prob_list:
        prod = prod * (1 - 2*p)
    return (1 - prod)/2


def soft_detectors_from_p1(p1, det_to_meas):
    """p1: (shots, n_meas). det_to_meas: list of lists of measurement indices per detector.
    Returns soft_det (shots, n_det) float in [0,1] = P(detector fired)."""
    n_shot = p1.shape[0]
    n_det = len(det_to_meas)
    soft = np.zeros((n_shot, n_det))
    for d, meas_idx in enumerate(det_to_meas):
        soft[:, d] = soft_xor([p1[:, m] for m in meas_idx])
    return soft
