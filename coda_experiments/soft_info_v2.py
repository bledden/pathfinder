"""Soft calibration v2: match the TRAINING generative model (data_soft.measurements_to_soft_p1).
Training assumes p1 = sigmoid(sep * z) with EQUAL priors, z = signed distance from the blob
midpoint along the |0>-|1> axis in pooled-sigma units. The v1 GMM posterior used fitted cluster
WEIGHTS (priors) + full covariance -> systematically miscalibrated vs training. v2 reuses a GMM
only to FIND the two blob means/assignment, then computes p1 the training way."""
import numpy as np
from sklearn.mixture import GaussianMixture

def calibrate_iq_v2(rec):
    n_shot, n_meas = rec.shape
    p1 = np.zeros((n_shot, n_meas)); meta = []
    for j in range(n_meas):
        pts = np.column_stack([rec[:, j].real, rec[:, j].imag]).astype(np.float64)
        s = np.abs(pts).max() or 1.0
        pts = pts / s
        # robust 2-cluster means via GMM (reg_covar prevents the singular-cov overflow)
        g = GaussianMixture(2, covariance_type='spherical', random_state=0, n_init=3,
                            reg_covar=1e-4).fit(pts)
        lab = g.predict(pts)
        m = g.means_                                  # (2,2)
        # |0> = majority cluster (ancilla/data reset to 0 dominates a memory expt)
        zero = int(np.argmax(g.weights_)); one = 1 - zero
        axis = m[one] - m[zero]; L = np.linalg.norm(axis)
        if L < 1e-9:                                  # blobs collapsed -> no info -> p1=hard-ish
            p1[:, j] = (lab == one).astype(float); meta.append(dict(sep_sigma=0.0)); continue
        u = axis / L
        mid = (m[zero] + m[one]) / 2.0
        proj = (pts - mid) @ u                        # signed dist from midpoint (sign: + toward |1>)
        # pooled within-cluster sigma along the axis
        sig = np.sqrt(max(np.var(proj[lab == zero]) * (lab == zero).mean()
                          + np.var(proj[lab == one]) * (lab == one).mean(), 1e-12))
        sep = L / sig                                 # separation in sigma units
        z = np.clip(sep * (proj / sig), -60, 60)      # == training's sep * I (I in sigma units)
        p1[:, j] = 1.0 / (1.0 + np.exp(-z))
        meta.append(dict(sep_sigma=float(sep), zero_comp=zero))
    return p1, meta
