"""On-the-fly syndrome data generation with Stim.

Uses Stim's detector coordinates for exact spatial layout mapping —
no guessing where detectors sit on the lattice.

Stim generates syndromes at ~1B Clifford gates/sec, so data generation
is never the training bottleneck.
"""
import stim
import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DataConfig:
    distance: int
    rounds: int
    physical_error_rate: float
    batch_size: int = 256
    code_type: str = "surface_code:rotated_memory_z"


class SyndromeDataset:
    """On-the-fly syndrome data generator.

    Each call to sample() generates a fresh batch of syndromes with
    ground-truth observable flips. The detector-to-grid mapping is
    computed once from Stim's detector coordinates.
    """
    def __init__(self, config: DataConfig):
        self.config = config
        self.circuit = stim.Circuit.generated(
            config.code_type,
            distance=config.distance,
            rounds=config.rounds,
            after_clifford_depolarization=config.physical_error_rate,
            before_measure_flip_probability=config.physical_error_rate,
            after_reset_flip_probability=config.physical_error_rate,
        )
        self.sampler = self.circuit.compile_detector_sampler()
        self.n_detectors = self.circuit.num_detectors
        self.n_observables = self.circuit.num_observables
        self._build_coordinate_map()

    def _build_coordinate_map(self):
        """Use Stim's detector coordinates for exact spatial mapping.

        stim.Circuit.get_detector_coordinates() returns a dict:
            detector_id → [x, y, t] or [x, y, z, t]

        We normalize these to integer grid indices for the 3D tensor.
        """
        coords = self.circuit.get_detector_coordinates()

        if not coords or self.n_detectors == 0:
            # Fallback for circuits without coordinate annotations
            d = self.config.distance
            r = self.config.rounds
            dpr = max(1, self.n_detectors // r)
            self.grid_shape = (r, d, d)
            self.det_to_grid = {}
            for i in range(self.n_detectors):
                ri = min(i // dpr, r - 1)
                li = i % dpr
                self.det_to_grid[i] = (ri, min(li // d, d - 1), min(li % d, d - 1))
            return

        # Extract coordinate arrays
        all_coords = np.array([coords[i] for i in range(self.n_detectors)])

        # Spatial coordinates are the first N-1 dims, time is the last
        spatial = all_coords[:, :-1]
        temporal = all_coords[:, -1]

        # Build unique sorted values for each axis
        t_unique = np.sort(np.unique(temporal))
        t_map = {v: i for i, v in enumerate(t_unique)}

        # For spatial dims, compute grid indices
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

        # Build mapping: detector_id → (t_idx, y_idx, x_idx)
        self.det_to_grid = {}
        for det_id in range(self.n_detectors):
            c = coords[det_id]
            t_idx = t_map[c[-1]]
            x_idx = x_map[c[0]]
            y_idx = y_map[c[1]] if len(c) > 2 else 0
            self.det_to_grid[det_id] = (t_idx, y_idx, x_idx)

    def detectors_to_tensor(self, det_events: np.ndarray) -> torch.Tensor:
        """Reshape flat detection events to spatial 3D tensor.

        Args:
            det_events: bool array [B, n_detectors]
        Returns:
            tensor: float tensor [B, 1, T, H, W]
        """
        B = det_events.shape[0]
        T, H, W = self.grid_shape

        tensor = torch.zeros(B, 1, T, H, W, dtype=torch.float32)

        for det_id, (gi, gj, gk) in self.det_to_grid.items():
            if gi < T and gj < H and gk < W and det_id < det_events.shape[1]:
                tensor[:, 0, gi, gj, gk] = torch.from_numpy(
                    det_events[:, det_id].astype(np.float32)
                )

        return tensor

    def sample(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of syndromes and ground-truth labels.

        Returns:
            syndromes: float tensor [B, 1, T, H, W]
            labels: float tensor [B, n_observables]
        """
        bs = batch_size or self.config.batch_size
        det_events, obs_flips = self.sampler.sample(shots=bs, separate_observables=True)

        syndromes = self.detectors_to_tensor(det_events)
        labels = torch.from_numpy(obs_flips.astype(np.float32))

        return syndromes, labels


class CurriculumScheduler:
    """Three-stage noise annealing from Gu et al.

    Stage 1 (0-20%):   constant low noise (p_target × 0.1)
    Stage 2 (20-60%):  linear ramp to medium noise (→ p_target × 0.5)
    Stage 3 (60-100%): linear ramp to target noise (→ p_target)
    """
    def __init__(self, target_rate: float, total_steps: int):
        self.target = target_rate
        self.total = total_steps

    def get_rate(self, step: int) -> float:
        frac = step / max(self.total, 1)
        if frac < 0.2:
            return self.target * 0.1
        elif frac < 0.6:
            t = (frac - 0.2) / 0.4
            return self.target * (0.1 + 0.4 * t)
        else:
            t = (frac - 0.6) / 0.4
            return self.target * (0.5 + 0.5 * t)
