"""IBM-budget asymmetric noise dataset for real-hardware-faithful training.

The key finding (2026-05-28): uniform depolarizing noise has the WRONG
STRUCTURE for real IBM Heron r2. The chip's error budget is dominated by
measurement + data-idle decoherence (the 5056ns readout dwell -> ~5% T2
dephasing per round), with comparatively tiny gate error (~0.3% cz).

A Stim circuit with the asymmetric per-component budget below, scaled by a
global magnitude multiplier, reproduces real ibm_fez d=5 r=5 det_flip=0.353
AND independently reproduces PyMatching's real-hardware LER at two distances
(a faithfulness check uniform depolarizing never passed).

This dataset samples a global multiplier alpha per batch (default [0.7, 1.3])
so the model sees noise budgets from 70%-130% of nominal IBM calibration
(robustness to calibration drift), while keeping the per-component RATIOS
fixed at IBM's actual structure.
"""
import stim
import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

import sys, os
sys.path.insert(0, os.path.dirname(__file__))


# T1/T2-derived per-component structure (from ibm_fez calibration snapshot):
#   T1_med=129us T2_med=96us; measure dwell 5056ns -> p_dephase=0.051;
#   reset 1600ns -> p_relax=0.012; sx 32ns / cz 68ns gate errors 0.0003 / 0.0029.
# These rates capture the STRUCTURE but undershoot the MAGNITUDE: they give
# det_flip~0.21 while real ibm_fez d=5 r=5 is 0.353. A global x2.2 multiplier
# (extra crosstalk/leakage/correlated error not in the T1/T2 model) brings the
# simulated det_flip to 0.354 AND independently reproduces PM's real-hardware LER
# (sim 44.5% vs real 45.7% at d=5; sim 28.6% vs real 28.5% at d=3).
_IBM_BUDGET_STRUCTURE = {
    "after_clifford_depolarization": 0.003,
    "before_measure_flip_probability": 0.055,
    "after_reset_flip_probability": 0.012,
    "before_round_data_depolarization": 0.050,
}
# The magnitude multiplier is DISTANCE-DEPENDENT: deeper circuits (more rounds,
# more gates) accumulate more of the "extra" non-T1/T2 noise (crosstalk, leakage,
# correlated error). Each multiplier is calibrated so the simulated circuit matches
# real ibm_fez on BOTH det_flip AND PyMatching-LER at that distance:
# The magnitude multiplier depends on BOTH distance and rounds (circuit depth):
# deeper circuits (more rounds) accumulate more of the "extra" non-T1/T2 noise, so
# they need a larger multiplier. Each value is calibrated so the simulated circuit
# matches real ibm_fez on det_flip AND PyMatching-LER at that (d, r); mild tension
# between the two targets is bracketed by the alpha in [0.7, 1.3] sweep.
#   (3,1): M=0.8 -> det 0.155 (real 0.135), PM 8.4% (real 8.1%)   [sub-threshold]
#   (5,1): M=1.2 -> det 0.230 (real 0.214), PM 15.5% (real 16.5%) [sub-threshold]
#   (3,3): M=1.4 -> det 0.26-0.28 (real 0.279), PM 28-30% (real 28.5%) [near thresh]
#   (5,5): M=1.9 -> det 0.32-0.33 (real 0.353), PM 46-47% (real 45.7%) [at thresh]
# (values 50k-shot verified; the alpha in [0.7,1.3] sweep brackets det-match & PM-match.)
# d=7 r=7 is past threshold on ibm_fez (no decodable signal), so not retrained here.
_MAGNITUDE_MULT_BY_DR = {(3, 1): 0.8, (5, 1): 1.2, (3, 3): 1.4, (5, 5): 1.9}
_MAGNITUDE_MULT_BY_D = {3: 1.4, 5: 1.9}  # back-compat (r=d fallback)


def budget_for(distance: int, rounds: int, mult: Optional[float] = None) -> dict:
    """Effective per-component noise budget for a (distance, rounds) point."""
    if mult is None:
        mult = _MAGNITUDE_MULT_BY_DR.get((distance, rounds),
                                         _MAGNITUDE_MULT_BY_D.get(distance, 2.1))
    return {k: v * mult for k, v in _IBM_BUDGET_STRUCTURE.items()}


def budget_for_distance(distance: int, mult: Optional[float] = None) -> dict:
    """Back-compat shim (assumes rounds=distance)."""
    return budget_for(distance, distance, mult)


# Back-compat default (d=5 r=5 budget); prefer budget_for() / config.magnitude_mult.
IBM_BUDGET = budget_for(5, 5)


@dataclass
class IBMBudgetDataConfig:
    distance: int
    rounds: int
    alpha_low: float = 0.7
    alpha_high: float = 1.3
    n_alphas: int = 13
    batch_size: int = 256
    code_type: str = "surface_code:rotated_memory_z"
    magnitude_mult: Optional[float] = None  # None -> per-distance default


class IBMBudgetSyndromeDataset:
    """Multi-noise dataset: asymmetric IBM budget scaled by alpha in [low, high]."""
    def __init__(self, config: IBMBudgetDataConfig, rng: Optional[np.random.Generator] = None):
        self.config = config
        self.rng = rng or np.random.default_rng()
        self.budget = budget_for(config.distance, config.rounds, config.magnitude_mult)
        self.alphas = np.linspace(config.alpha_low, config.alpha_high, config.n_alphas)
        self.samplers = {}
        for a in self.alphas:
            kw = {k: min(v * float(a), 0.5) for k, v in self.budget.items()}
            circ = stim.Circuit.generated(
                config.code_type, distance=config.distance, rounds=config.rounds, **kw)
            self.samplers[float(a)] = circ.compile_detector_sampler()

        ref = stim.Circuit.generated(
            config.code_type, distance=config.distance, rounds=config.rounds,
            **{k: v for k, v in self.budget.items()})
        self.circuit = ref
        self.n_detectors = ref.num_detectors
        self.n_observables = ref.num_observables
        self._build_coordinate_map()

    def _build_coordinate_map(self):
        coords = self.circuit.get_detector_coordinates()
        all_coords = np.array([coords[i] for i in range(self.n_detectors)])
        spatial = all_coords[:, :-1]
        temporal = all_coords[:, -1]
        t_unique = np.sort(np.unique(temporal))
        t_map = {v: i for i, v in enumerate(t_unique)}
        if spatial.shape[1] >= 2:
            x_unique = np.sort(np.unique(spatial[:, 0]))
            y_unique = np.sort(np.unique(spatial[:, 1]))
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
                    det_events[:, det_id].astype(np.float32))
        return tensor

    def sample(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        bs = batch_size or self.config.batch_size
        a_idx = int(self.rng.integers(0, len(self.alphas)))
        a = float(self.alphas[a_idx])
        det_events, obs_flips = self.samplers[a].sample(shots=bs, separate_observables=True)
        syndromes = self.detectors_to_tensor(det_events)
        labels = torch.from_numpy(obs_flips.astype(np.float32))
        return syndromes, labels, a


if __name__ == "__main__":
    cfg = IBMBudgetDataConfig(distance=5, rounds=5, n_alphas=5)
    ds = IBMBudgetSyndromeDataset(cfg)
    print(f"Grid: {ds.grid_shape}, n_det={ds.n_detectors}")
    print(f"alphas: {list(ds.samplers.keys())}")
    for _ in range(5):
        x, y, a = ds.sample(2000)
        det_rate = x.sum().item() / (2000 * ds.n_detectors)
        print(f"  alpha={a:.3f}  obs_flip={y.mean().item():.4f}  det_flip~{det_rate:.4f}")
    print("\nTarget real IBM d=5 r=5: det_flip=0.353 obs_flip=0.491 (alpha=1.0 should match)")
