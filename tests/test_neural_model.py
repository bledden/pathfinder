"""Tests for neural CNN decoder model."""
import sys
import os
import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
from model import NeuralDecoder, DecoderConfig, DirectionalConv3d, BottleneckBlock


class TestDirectionalConv:
    def test_output_shape(self):
        conv = DirectionalConv3d(32, 64)
        x = torch.randn(4, 32, 3, 3, 3)
        out = conv(x)
        assert out.shape == (4, 64, 3, 3, 3)

    def test_differs_from_standard_conv(self):
        """Direction-specific weights should produce different results than
        a standard Conv3d with shared weights."""
        torch.manual_seed(42)
        dir_conv = DirectionalConv3d(16, 16)
        std_conv = torch.nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=False)

        x = torch.randn(2, 16, 5, 5, 5)
        dir_out = dir_conv(x)
        std_out = std_conv(x)

        # They should NOT be equal (different weight structure)
        assert not torch.allclose(dir_out, std_out, atol=1e-3), \
            "Directional conv should differ from standard conv"

    def test_handles_small_spatial(self):
        """Should work even with 1x1x1 spatial dims."""
        conv = DirectionalConv3d(16, 16)
        x = torch.randn(2, 16, 1, 1, 1)
        out = conv(x)
        assert out.shape == (2, 16, 1, 1, 1)


class TestBottleneckBlock:
    def test_residual_connection(self):
        """Output should differ from input (transformation) but have same shape."""
        block = BottleneckBlock(64)
        x = torch.randn(2, 64, 3, 3, 3)
        out = block(x)
        assert out.shape == x.shape
        assert not torch.allclose(out, x)

    def test_bottleneck_reduces_channels(self):
        """Internal reduced dim should be H//4."""
        block = BottleneckBlock(256)
        assert block.reduce.out_channels == 64
        assert block.restore.in_channels == 64


class TestNeuralDecoder:
    def test_forward_d3(self):
        config = DecoderConfig(distance=3, rounds=3, hidden_dim=64)
        model = NeuralDecoder(config)
        syndrome = torch.randn(4, 1, 3, 3, 3)
        logits = model(syndrome)
        assert logits.shape == (4, 1)

    def test_forward_d5(self):
        config = DecoderConfig(distance=5, rounds=5, hidden_dim=128)
        model = NeuralDecoder(config)
        # Grid shape from Stim might not be exactly 5x5 — test with actual dims
        syndrome = torch.randn(4, 1, 5, 5, 5)
        logits = model(syndrome)
        assert logits.shape == (4, 1)

    def test_predict_returns_bool(self):
        config = DecoderConfig(distance=3, rounds=3, hidden_dim=32)
        model = NeuralDecoder(config)
        syndrome = torch.randn(4, 1, 3, 3, 3)
        preds = model.predict(syndrome)
        assert preds.dtype == torch.bool
        assert preds.shape == (4, 1)

    def test_n_blocks_equals_distance(self):
        config = DecoderConfig(distance=7, rounds=7)
        assert config.n_blocks == 7
        model = NeuralDecoder(config)
        assert len(model.blocks) == 7

    def test_parameter_count_d5_h256(self):
        config = DecoderConfig(distance=5, rounds=5, hidden_dim=256)
        model = NeuralDecoder(config)
        n = NeuralDecoder.count_parameters(model)
        print(f"d=5, H=256, L=5: {n:,} params ({n * 4 / 1e6:.1f} MB FP32, {n * 2 / 1e6:.1f} MB FP16)")
        # Should be in the low millions, not tens of millions
        assert 100_000 < n < 50_000_000

    def test_gradient_flow(self):
        """All parameters should receive gradients."""
        config = DecoderConfig(distance=3, rounds=3, hidden_dim=32)
        model = NeuralDecoder(config)

        syndrome = torch.randn(4, 1, 3, 3, 3)
        target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

        logits = model(syndrome)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
