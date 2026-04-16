"""Neural CNN decoder for surface code QEC — Gu et al. (arXiv:2604.08358, April 2026).

Key architectural elements:
- Direction-specific convolution: separate weight matrices per neighbor direction
  in the 3D syndrome lattice (NOT standard nn.Conv3d which shares weights)
- Bottleneck residual blocks: reduce H→H//4, message pass, restore H//4→H
- L = d blocks (depth scales with code distance)
- Muon optimizer for 2D weights, AdamW for bias/1D params
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DecoderConfig:
    """Configuration matching Gu et al. defaults."""
    distance: int
    rounds: int
    hidden_dim: int = 256     # Paper: 256-512
    n_observables: int = 1

    @property
    def n_blocks(self):
        """L ~ d: number of residual blocks scales with code distance."""
        return self.distance


class DirectionalConv3d(nn.Module):
    """Direction-specific message passing on 3D syndrome lattice.

    Instead of a single 3x3x3 kernel (standard Conv3d), uses 7 separate
    weight matrices — one for each interaction type:
      self, +t, -t, +row, -row, +col, -col

    Paper: "each layer uses direction-specific weight matrices depending on
    relative position between stabilizers, not absolute position"

    This is critical — it preserves the lattice structure that standard
    Conv3d would blur.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.w_self = nn.Linear(in_channels, out_channels, bias=False)
        self.w_tp = nn.Linear(in_channels, out_channels, bias=False)  # +time
        self.w_tm = nn.Linear(in_channels, out_channels, bias=False)  # -time
        self.w_rp = nn.Linear(in_channels, out_channels, bias=False)  # +row
        self.w_rm = nn.Linear(in_channels, out_channels, bias=False)  # -row
        self.w_cp = nn.Linear(in_channels, out_channels, bias=False)  # +col
        self.w_cm = nn.Linear(in_channels, out_channels, bias=False)  # -col

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T, R, Co] — channel-first 3D feature map
        Returns:
            [B, C_out, T, R, Co]
        """
        # Permute to channel-last for nn.Linear: [B, T, R, Co, C]
        xp = x.permute(0, 2, 3, 4, 1)

        # Self connection
        out = self.w_self(xp)

        # Temporal neighbors (shift along T dimension)
        if xp.shape[1] > 1:
            out = out + F.pad(self.w_tp(xp[:, :-1]), (0, 0, 0, 0, 0, 0, 1, 0))
            out = out + F.pad(self.w_tm(xp[:, 1:]),  (0, 0, 0, 0, 0, 0, 0, 1))

        # Row neighbors (shift along R dimension)
        if xp.shape[2] > 1:
            out = out + F.pad(self.w_rp(xp[:, :, :-1]), (0, 0, 0, 0, 1, 0))
            out = out + F.pad(self.w_rm(xp[:, :, 1:]),  (0, 0, 0, 0, 0, 1))

        # Column neighbors (shift along Co dimension)
        if xp.shape[3] > 1:
            out = out + F.pad(self.w_cp(xp[:, :, :, :-1]), (0, 0, 1, 0))
            out = out + F.pad(self.w_cm(xp[:, :, :, 1:]),  (0, 0, 0, 1))

        # Back to channel-first: [B, C_out, T, R, Co]
        return out.permute(0, 4, 1, 2, 3)


class BottleneckBlock(nn.Module):
    """Bottleneck residual block with directional message passing.

    Reduce (1x1x1 conv, H → H//4)
    → Directional message passing (direction-specific weights, H//4 → H//4)
    → Restore (1x1x1 conv, H//4 → H)
    → Residual connection
    → LayerNorm
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        reduced = hidden_dim // 4

        self.reduce = nn.Conv3d(hidden_dim, reduced, kernel_size=1, bias=False)
        self.message = DirectionalConv3d(reduced, reduced)
        self.restore = nn.Conv3d(reduced, hidden_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, T, R, Co] → [B, H, T, R, Co]"""
        residual = x

        out = F.gelu(self.reduce(x))        # [B, H//4, T, R, Co]
        out = F.gelu(self.message(out))      # [B, H//4, T, R, Co]
        out = self.restore(out)              # [B, H, T, R, Co]

        out = out + residual

        # LayerNorm over channel dim: permute to [B, T, R, Co, H], norm, permute back
        out = out.permute(0, 2, 3, 4, 1)
        out = self.norm(out)
        out = out.permute(0, 4, 1, 2, 3)

        return out


class NeuralDecoder(nn.Module):
    """CNN decoder for surface code quantum error correction.

    Architecture (Gu et al. 2026):
        Binary syndrome [B, 1, R, d, d]
        → Embedding (1x1x1 conv, 1 → H)
        → L bottleneck blocks (L = d)
        → Global average pooling
        → MLP head → logit per observable
    """
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        H = config.hidden_dim

        # Embedding: lift binary syndrome to H-dimensional space
        self.embed = nn.Conv3d(1, H, kernel_size=1, bias=True)

        # L = d bottleneck residual blocks
        self.blocks = nn.ModuleList([
            BottleneckBlock(H) for _ in range(config.n_blocks)
        ])

        # Output head: global avg pool → 2-layer MLP → logits
        self.head = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.Linear(H, config.n_observables),
        )

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        """
        Args:
            syndrome: [B, 1, R, d, d] float tensor (binary values as float)
        Returns:
            logits: [B, n_observables] raw logits
        """
        x = self.embed(syndrome)          # [B, H, R, d, d]

        for block in self.blocks:
            x = block(x)                  # [B, H, R, d, d]

        x = x.mean(dim=(2, 3, 4))        # [B, H] — global average pooling

        return self.head(x)               # [B, n_observables]

    def predict(self, syndrome: torch.Tensor) -> torch.Tensor:
        """Predict observable flips (binary)."""
        with torch.no_grad():
            return self.forward(syndrome) > 0

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
