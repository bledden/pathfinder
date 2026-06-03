"""Soft-aware training data: Stim hard measurements -> simulated per-shot IQ -> soft detectors.

Makes train==test for the soft-readout experiment. For each Stim measurement bit b in {0,1}:
  draw an IQ point from a 2D Gaussian centered at mu_0 (b=0) or mu_1 (b=1), separation SNR*sigma.
Then run the SAME soft_info pipeline used on real IBM data (per-shot P(meas=1) via the known
Gaussians -> probabilistic-XOR soft detectors). The model thus learns to consume soft [0,1]
detector confidences exactly as they appear on real hardware.

SNR is sampled per-measurement from the REAL ibm_fez kerneled distribution (median ~5, tail to
~1.5) so the simulated readout-confidence statistics match the chip. Combined with the
IBM-budget circuit noise (data_ibm_budget) this is a faithful soft-noise training distribution.
"""
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_ibm_budget import IBMBudgetSyndromeDataset, IBMBudgetDataConfig, budget_for


def measurements_to_soft_p1(meas_bits, snr_per_meas, rng):
    """meas_bits: (B, n_meas) bool/int. snr_per_meas: (n_meas,) blob separation in sigma.
    Returns soft P(meas=1): (B, n_meas) float in (0,1), computed the SAME way soft_info does
    on real IQ (2-Gaussian posterior). sigma=1; mu0=-snr/2, mu1=+snr/2 along the I axis,
    Q axis pure noise (no info), matching a 1D-separable readout."""
    B, nm = meas_bits.shape
    sep = snr_per_meas[None, :]                      # (1, nm), in sigma units
    mu1 = sep / 2.0
    mu0 = -sep / 2.0
    true_mu = np.where(meas_bits.astype(bool), mu1, mu0)   # (B, nm)
    I = true_mu + rng.standard_normal((B, nm))       # observed I with unit-sigma noise
    # 2-Gaussian posterior P(1|I) with equal priors, shared sigma=1:
    #   = sigmoid( (mu1-mu0) * (I - (mu1+mu0)/2) )  -- here (mu1+mu0)/2 = 0
    z = np.clip((mu1 - mu0) * (I - 0.0), -60.0, 60.0)   # clip to avoid exp overflow
    p1 = 1.0 / (1.0 + np.exp(-z))
    return p1


def soft_xor_prob(prob_cols):
    prod = np.ones_like(prob_cols[0])
    for p in prob_cols:
        prod = prod * (1 - 2*p)
    return (1 - prod) / 2


@dataclass
class SoftDataConfig:
    distance: int
    rounds: int
    alpha_low: float = 0.7
    alpha_high: float = 1.3
    n_alphas: int = 13
    batch_size: int = 256
    # Calibrated (clean sweep) so TRAINING soft-detector uncertain-frac matches REAL ibm_fez:
    #   d3r3 real 1.35% -> snr_median 8.5 ; d5r5 real 3.92% -> snr_median 6.5
    # train_soft.py passes --snr explicitly per (d,r); this default is just a fallback.
    snr_median: float = 8.5
    snr_min: float = 1.5
    snr_max: float = 7.5
    magnitude_mult: Optional[float] = None


class SoftSyndromeDataset:
    """Wraps IBMBudgetSyndromeDataset but emits SOFT detector tensors (float P in [0,1])
    from simulated per-shot IQ, instead of hard binary detectors. Labels unchanged."""
    def __init__(self, config: SoftDataConfig, rng: Optional[np.random.Generator] = None):
        self.config = config
        self.rng = rng or np.random.default_rng()
        # underlying hard-measurement sampler at the IBM-budget circuit noise
        budcfg = IBMBudgetDataConfig(distance=config.distance, rounds=config.rounds,
                                     alpha_low=config.alpha_low, alpha_high=config.alpha_high,
                                     n_alphas=config.n_alphas, batch_size=config.batch_size,
                                     magnitude_mult=config.magnitude_mult)
        self.bud = IBMBudgetSyndromeDataset(budcfg, rng=self.rng)
        self.grid_shape = self.bud.grid_shape
        self.det_to_grid = self.bud.det_to_grid
        self.n_detectors = self.bud.n_detectors
        # build measurement->detector map for soft detectors
        import stim
        from soft_detmap import detector_to_measurements
        self.det_to_meas, self.n_meas, self.nd, self.circuit = detector_to_measurements(
            config.distance, config.rounds)
        # per-measurement SNR drawn once (fixed chip calibration), clipped to real range
        snr = self.rng.normal(config.snr_median, (config.snr_max-config.snr_min)/4.0, self.n_meas)
        self.snr_per_meas = np.clip(snr, config.snr_min, config.snr_max)
        # we need the measurement-level sampler (not just detectors) to inject IQ
        self.code_type = budcfg.code_type
        self._meas_samplers = {}
        for a, _ in self.bud.samplers.items():
            kw = {k: min(v*float(a), 0.5) for k, v in self.bud.budget.items()}
            circ = stim.Circuit.generated(self.code_type, distance=config.distance,
                                          rounds=config.rounds, **kw)
            self._meas_samplers[a] = circ.compile_sampler()   # measurement sampler
        self._m2d = self.circuit.compile_m2d_converter()
        self.alphas = list(self.bud.samplers.keys())

    def soft_detectors_to_tensor(self, soft_det):
        B = soft_det.shape[0]
        T, H, W = self.grid_shape
        t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        sd = torch.from_numpy(soft_det.astype(np.float32))
        for det_id, (gi, gj, gk) in self.det_to_grid.items():
            if gi < T and gj < H and gk < W and det_id < soft_det.shape[1]:
                t[:, 0, gi, gj, gk] = sd[:, det_id]
        return t

    def sample(self, batch_size: Optional[int] = None):
        bs = batch_size or self.config.batch_size
        a = float(self.alphas[int(self.rng.integers(0, len(self.alphas)))])
        # measurement sampler returns raw bits only (no separate_observables kwarg)
        meas = self._meas_samplers[a].sample(shots=bs).astype(np.uint8)
        # detectors + observables from the TRUE hard bits via the m2d converter
        det_true, obs_true = self._m2d.convert(measurements=meas.astype(bool), separate_observables=True)
        # simulate per-shot IQ -> soft P(meas=1) -> soft detectors
        p1 = measurements_to_soft_p1(meas, self.snr_per_meas, self.rng)
        soft_det = np.zeros((bs, self.nd), dtype=np.float64)
        for d, midx in enumerate(self.det_to_meas):
            soft_det[:, d] = soft_xor_prob([p1[:, m] for m in midx])
        syn = self.soft_detectors_to_tensor(soft_det)
        labels = torch.from_numpy(obs_true.astype(np.float32))
        return syn, labels, a


if __name__ == "__main__":
    cfg = SoftDataConfig(distance=3, rounds=3, n_alphas=5)
    ds = SoftSyndromeDataset(cfg, rng=np.random.default_rng(0))
    print(f"grid={ds.grid_shape} n_det={ds.n_detectors} n_meas={ds.n_meas}")
    print(f"snr per meas: min={ds.snr_per_meas.min():.2f} med={np.median(ds.snr_per_meas):.2f} max={ds.snr_per_meas.max():.2f}")
    x, y, a = ds.sample(2000)
    sv = x.numpy()[x.numpy() > 0]
    print(f"alpha={a:.2f} soft-detector tensor: nonzero mean={sv.mean():.3f} "
          f"frac in (0.1,0.9)={((sv>0.1)&(sv<0.9)).mean():.4f}  (graded soft info present)")
    print(f"label flip rate={y.mean().item():.4f}")
