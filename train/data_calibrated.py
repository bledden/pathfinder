"""IBM-calibrated multi-noise syndrome dataset for real-hardware-robust training.

The original SyndromeDataset trains at a single uniform depolarizing rate p,
which produces models that overfit to that specific noise distribution. On
real IBM Heron r2 hardware (detector flip rate ~0.20-0.35 at d=5 r=5),
single-p models underperform PyMatching because PM's combinatorial matching
is noise-distribution-agnostic.

This dataset:
  1. Samples noise rate p uniformly from [p_low, p_high] per batch.
  2. Optionally scales readout-error rate relative to gate-error rate
     (IBM Heron r2 has measurement errors ~1.5-2x its 2q-gate errors).
  3. Pre-compiles all samplers so training is bottlenecked only on GPU.
"""
import stim
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data import SyndromeDataset, DataConfig


@dataclass
class CalibratedDataConfig:
    distance: int
    rounds: int
    p_low: float = 0.003          # bottom of noise sweep
    p_high: float = 0.025         # top of noise sweep — covers IBM operational regime
    n_rates: int = 15             # number of discrete p values to pre-compile
    readout_scale: float = 1.5    # multiplicative scale on measurement-flip rate
    batch_size: int = 256
    code_type: str = "surface_code:rotated_memory_z"


class CalibratedSyndromeDataset:
    """Multi-noise dataset that pre-compiles N samplers at different p values.

    Each call to sample() picks one p at random (uniformly across the discrete
    rate set) and returns a batch from that sampler. Over many steps the model
    sees the full noise distribution.
    """
    def __init__(self, config: CalibratedDataConfig, rng: Optional[np.random.Generator] = None):
        self.config = config
        self.rng = rng or np.random.default_rng()

        # Pre-compile samplers across the calibrated noise sweep
        self.rates = np.linspace(config.p_low, config.p_high, config.n_rates)
        self.samplers = {}
        for p in self.rates:
            circ = stim.Circuit.generated(
                config.code_type,
                distance=config.distance,
                rounds=config.rounds,
                after_clifford_depolarization=float(p),
                before_measure_flip_probability=float(p * config.readout_scale),
                after_reset_flip_probability=float(p),
                before_round_data_depolarization=float(p),
            )
            self.samplers[float(p)] = circ.compile_detector_sampler()

        # The detector layout is the same for all p (same circuit topology)
        # — use the first sampler's circuit to build the coordinate map
        ref_circ = stim.Circuit.generated(
            config.code_type,
            distance=config.distance, rounds=config.rounds,
            after_clifford_depolarization=float(self.rates[0]),
        )
        self.circuit = ref_circ
        self.n_detectors = ref_circ.num_detectors
        self.n_observables = ref_circ.num_observables
        self._build_coordinate_map()

    def _build_coordinate_map(self):
        """Same coordinate-to-grid mapping as SyndromeDataset (factored out)."""
        coords = self.circuit.get_detector_coordinates()
        all_coords = np.array([coords[i] for i in range(self.n_detectors)])
        spatial = all_coords[:, :-1]
        temporal = all_coords[:, -1]
        t_unique = np.sort(np.unique(temporal))
        t_map = {v: i for i, v in enumerate(t_unique)}
        if spatial.shape[1] >= 2:
            xs = spatial[:, 0]
            ys = spatial[:, 1]
            x_unique = np.sort(np.unique(xs))
            y_unique = np.sort(np.unique(ys))
            x_map = {v: i for i, v in enumerate(x_unique)}
            y_map = {v: i for i, v in enumerate(y_unique)}
            self.grid_shape = (len(t_unique), len(y_unique), len(x_unique))
        else:
            x_unique = np.sort(np.unique(spatial[:, 0]))
            x_map = {v: i for i, v in enumerate(x_unique)}
            y_map = {0: 0}
            self.grid_shape = (len(t_unique), 1, len(x_unique))
        self.det_to_grid = {}
        for det_id in range(self.n_detectors):
            c = coords[det_id]
            t_idx = t_map[c[-1]]
            x_idx = x_map[c[0]]
            y_idx = y_map[c[1]] if len(c) > 2 else 0
            self.det_to_grid[det_id] = (t_idx, y_idx, x_idx)

    def detectors_to_tensor(self, det_events: np.ndarray) -> torch.Tensor:
        B = det_events.shape[0]
        T, H, W = self.grid_shape
        tensor = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        for det_id, (gi, gj, gk) in self.det_to_grid.items():
            if gi < T and gj < H and gk < W and det_id < det_events.shape[1]:
                tensor[:, 0, gi, gj, gk] = torch.from_numpy(
                    det_events[:, det_id].astype(np.float32)
                )
        return tensor

    def sample(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Pick a random p, sample a batch.

        Returns: (syndromes [B,1,T,H,W], labels [B,n_obs], p_used)
        """
        bs = batch_size or self.config.batch_size
        p_idx = int(self.rng.integers(0, len(self.rates)))
        p = float(self.rates[p_idx])
        sampler = self.samplers[p]
        det_events, obs_flips = sampler.sample(shots=bs, separate_observables=True)
        syndromes = self.detectors_to_tensor(det_events)
        labels = torch.from_numpy(obs_flips.astype(np.float32))
        return syndromes, labels, p


class CurriculumMultiNoise:
    """Curriculum: anneal both the noise-rate range AND the readout scale.

    Stage 1 (0-20%): narrow band at low noise — p ∈ [p_low, p_low + (p_high-p_low)*0.2]
    Stage 2 (20-60%): expand band gradually
    Stage 3 (60-100%): full [p_low, p_high] sweep

    This lets the model build up basic syndrome recognition before grappling
    with the high-noise regime that IBM operates in.
    """
    def __init__(self, base_config: CalibratedDataConfig, total_steps: int):
        self.base = base_config
        self.total = total_steps

    def get_config(self, step: int) -> CalibratedDataConfig:
        frac = step / max(self.total, 1)
        if frac < 0.2:
            band = 0.2
        elif frac < 0.6:
            band = 0.2 + 0.8 * (frac - 0.2) / 0.4
        else:
            band = 1.0
        p_high_now = self.base.p_low + (self.base.p_high - self.base.p_low) * band
        cfg = CalibratedDataConfig(**{**self.base.__dict__, 'p_high': p_high_now})
        return cfg


if __name__ == "__main__":
    # Smoke test
    cfg = CalibratedDataConfig(distance=3, rounds=3, n_rates=5)
    ds = CalibratedSyndromeDataset(cfg)
    print(f"Grid shape: {ds.grid_shape}, n_detectors: {ds.n_detectors}")
    print(f"Pre-compiled samplers at p = {list(ds.samplers.keys())}")
    for _ in range(3):
        x, y, p = ds.sample(8)
        print(f"  p={p:.4f}  x.shape={tuple(x.shape)}  y.shape={tuple(y.shape)}  "
              f"flip_rate={x.mean().item():.4f}")
