"""
Pathfinder-Equivariant: D₂-equivariant neural decoder for surface codes.

Novel contribution: architecture that is PROVABLY equivariant under the D₂
symmetry group of rotated-memory-Z (row-flip × col-flip). This means:
  - Parameter count is reduced (+row and -row share weights, +col and -col share)
  - Symmetry is guaranteed by construction (no extrapolation failure on flipped syndromes)
  - Sample efficiency improves: the network can't waste capacity on asymmetric solutions

This is the first equivariant neural decoder for surface codes (to our knowledge).

Architecture differences from v1:
  - EquivariantDirectionalConv3d: 5 weights instead of 7 (ties ±row and ±col)
  - Everything else: same as v2 (pre-norm, H=512, noise conditioning)
"""
import sys, os, time, math, argparse
sys.path.insert(0, "/workspace/pathfinder/train")
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import stim
from muon import SingleDeviceMuon
import torch.optim
torch.optim.Muon = SingleDeviceMuon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EquivariantDirectionalConv3d(nn.Module):
    """D₂-equivariant version: ties +row ↔ -row and +col ↔ -col.

    Math: under row-flip R:
      x[b, c, t, r, c] → x[b, c, t, R-1-r, c]
    Our kernel at output position (t, r, c) is:
      out(t,r,c) = W_self * x(t,r,c)
                 + W_tp * x(t-1,r,c) + W_tm * x(t+1,r,c)
                 + W_rp * x(t,r-1,c) + W_rm * x(t,r+1,c)
                 + W_cp * x(t,r,c-1) + W_cm * x(t,r,c+1)

    For row-flip equivariance: we require that applying R first (to input) then the layer
    is the same as applying the layer first and then R (to output).
    This yields: W_rp = W_rm (and the "r direction" becomes symmetric).
    Same reasoning for col-flip: W_cp = W_cm.

    Net result: 5 distinct weights (self, t+, t-, axial_r, axial_c) instead of 7.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.w_self = nn.Linear(in_channels, out_channels, bias=False)
        self.w_tp = nn.Linear(in_channels, out_channels, bias=False)
        self.w_tm = nn.Linear(in_channels, out_channels, bias=False)
        self.w_r = nn.Linear(in_channels, out_channels, bias=False)   # shared ±r
        self.w_c = nn.Linear(in_channels, out_channels, bias=False)   # shared ±c

    def forward(self, x):  # [B, C, T, R, Co]
        xp = x.permute(0, 2, 3, 4, 1)
        out = self.w_self(xp)
        if xp.shape[1] > 1:
            out = out + F.pad(self.w_tp(xp[:, :-1]), (0,0,0,0,0,0,1,0))
            out = out + F.pad(self.w_tm(xp[:, 1:]),  (0,0,0,0,0,0,0,1))
        if xp.shape[2] > 1:
            # Shared w_r for +r and -r
            r_plus = F.pad(self.w_r(xp[:, :, :-1]), (0,0,0,0,1,0))
            r_minus = F.pad(self.w_r(xp[:, :, 1:]),  (0,0,0,0,0,1))
            out = out + r_plus + r_minus
        if xp.shape[3] > 1:
            # Shared w_c for +c and -c
            c_plus = F.pad(self.w_c(xp[:, :, :, :-1]), (0,0,1,0))
            c_minus = F.pad(self.w_c(xp[:, :, :, 1:]),  (0,0,0,1))
            out = out + c_plus + c_minus
        return out.permute(0, 4, 1, 2, 3)


class EquivariantPreNormBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        reduced = hidden_dim // 4
        self.norm = nn.LayerNorm(hidden_dim)
        self.reduce = nn.Conv3d(hidden_dim, reduced, kernel_size=1, bias=False)
        self.message = EquivariantDirectionalConv3d(reduced, reduced)
        self.restore = nn.Conv3d(reduced, hidden_dim, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        h = x.permute(0, 2, 3, 4, 1)
        h = self.norm(h)
        h = h.permute(0, 4, 1, 2, 3)
        h = F.gelu(self.reduce(h))
        h = F.gelu(self.message(h))
        h = self.restore(h)
        return residual + h


class NoiseEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(1, hidden_dim)

    def forward(self, x, log_p):
        emb = self.proj(log_p.unsqueeze(-1))
        return x + emb.view(emb.shape[0], emb.shape[1], 1, 1, 1)


class EquivariantDecoder(nn.Module):
    def __init__(self, distance, rounds, hidden_dim=512, n_blocks=None, n_observables=1):
        super().__init__()
        self.distance = distance; self.rounds = rounds
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks if n_blocks else distance
        self.n_observables = n_observables
        self.embed = nn.Conv3d(1, hidden_dim, kernel_size=1, bias=True)
        self.noise_embed = NoiseEmbedding(hidden_dim)
        self.blocks = nn.ModuleList([EquivariantPreNormBlock(hidden_dim) for _ in range(self.n_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_observables),
        )

    def forward(self, syn, log_p):
        x = self.embed(syn)
        x = self.noise_embed(x, log_p)
        for b in self.blocks:
            x = b(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.final_norm(x)
        x = x.mean(dim=(1, 2, 3))
        return self.head(x)


def test_equivariance():
    """Smoke test: network output should be INVARIANT under D₂ flips."""
    torch.manual_seed(42)
    d = 5
    m = EquivariantDecoder(d, d, hidden_dim=64, n_blocks=2).to(device).eval()

    B = 4
    x = torch.rand(B, 1, d, d+3, d+3, device=device)  # arbitrary shape close to grid
    log_p = torch.full((B,), -4.0, device=device)

    with torch.no_grad():
        y0 = m(x, log_p)
        y_r = m(torch.flip(x, dims=[-2]), log_p)
        y_c = m(torch.flip(x, dims=[-1]), log_p)
        y_rc = m(torch.flip(x, dims=[-2, -1]), log_p)

    # Check invariance (logical Z observable is invariant under D₂ flips)
    d_r = (y0 - y_r).abs().max().item()
    d_c = (y0 - y_c).abs().max().item()
    d_rc = (y0 - y_rc).abs().max().item()
    print(f"Equivariance test (should be ~1e-5 for FP32):")
    print(f"  max |y_id - y_rowflip|  = {d_r:.6e}")
    print(f"  max |y_id - y_colflip|  = {d_c:.6e}")
    print(f"  max |y_id - y_both|     = {d_rc:.6e}")

    return max(d_r, d_c, d_rc) < 1e-3


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        ok = test_equivariance()
        sys.exit(0 if ok else 1)
    print("Usage: python3 train_equivariant.py test  # to verify equivariance")
    print("       python3 train_equivariant.py train # (not yet implemented; uses train_v2 flow)")
