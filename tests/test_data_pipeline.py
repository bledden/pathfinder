"""Tests for Stim data pipeline."""
import sys
import os
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
from data import SyndromeDataset, DataConfig, CurriculumScheduler


class TestSyndromeDataset:
    def test_sample_shape_d3(self):
        config = DataConfig(distance=3, rounds=3, physical_error_rate=0.005, batch_size=16)
        dataset = SyndromeDataset(config)
        syndromes, labels = dataset.sample()
        # Shape: [B, 1, T, H, W]
        assert syndromes.shape[0] == 16
        assert syndromes.shape[1] == 1
        assert labels.shape == (16, 1)
        print(f"d=3 grid shape: {dataset.grid_shape}")

    def test_sample_shape_d5(self):
        config = DataConfig(distance=5, rounds=5, physical_error_rate=0.005, batch_size=8)
        dataset = SyndromeDataset(config)
        syndromes, labels = dataset.sample()
        assert syndromes.shape[0] == 8
        assert syndromes.shape[1] == 1
        print(f"d=5 grid shape: {dataset.grid_shape}")

    def test_binary_values(self):
        config = DataConfig(distance=3, rounds=3, physical_error_rate=0.005, batch_size=100)
        dataset = SyndromeDataset(config)
        syndromes, labels = dataset.sample()
        unique = torch.unique(syndromes)
        assert all(v in [0.0, 1.0] for v in unique.tolist()), f"Non-binary values: {unique}"
        label_unique = torch.unique(labels)
        assert all(v in [0.0, 1.0] for v in label_unique.tolist())

    def test_low_noise_mostly_zeros(self):
        config = DataConfig(distance=3, rounds=3, physical_error_rate=0.0001, batch_size=500)
        dataset = SyndromeDataset(config)
        syndromes, _ = dataset.sample()
        n_zero = torch.sum(syndromes.sum(dim=(1, 2, 3, 4)) == 0).item()
        assert n_zero > 400, f"Expected >400 zero syndromes at low noise, got {n_zero}"

    def test_spatial_mapping_uses_coordinates(self):
        """Grid shape should come from Stim's detector coordinates, not a guess."""
        config = DataConfig(distance=5, rounds=5, physical_error_rate=0.001)
        dataset = SyndromeDataset(config)
        # Stim's rotated surface code should have specific spatial dims
        T, H, W = dataset.grid_shape
        assert T >= 5, f"Expected >=5 time steps, got {T}"
        assert H > 1 and W > 1, f"Spatial dims too small: {H}x{W}"
        print(f"d=5 coordinate-derived grid: {T}x{H}x{W}, "
              f"{len(dataset.det_to_grid)} detectors mapped")

    def test_all_detectors_mapped(self):
        config = DataConfig(distance=3, rounds=3, physical_error_rate=0.001)
        dataset = SyndromeDataset(config)
        assert len(dataset.det_to_grid) == dataset.n_detectors


class TestCurriculum:
    def test_three_stages(self):
        sched = CurriculumScheduler(target_rate=0.007, total_steps=80000)
        early = sched.get_rate(0)
        mid = sched.get_rate(40000)
        late = sched.get_rate(79999)
        assert early < mid < late, f"Curriculum not monotonic: {early}, {mid}, {late}"

    def test_target_reached(self):
        sched = CurriculumScheduler(target_rate=0.007, total_steps=80000)
        final = sched.get_rate(80000)
        assert abs(final - 0.007) < 0.001, f"Final rate {final} not near target 0.007"

    def test_stage1_is_low(self):
        sched = CurriculumScheduler(target_rate=0.007, total_steps=1000)
        rate = sched.get_rate(50)  # 5% through = stage 1
        assert rate < 0.002, f"Stage 1 rate {rate} too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
