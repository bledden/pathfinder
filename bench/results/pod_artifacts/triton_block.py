"""
Fully-patched BottleneckBlock using both:
  - TritonDirectionalConv3d (fused 7 matmuls + boundary adds)
  - FusedRestoreNormBlock (fused restore + residual add + LayerNorm)

Usage: swap_to_full_triton(decoder) walks model and patches all blocks.
"""
import sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace")
from model import BottleneckBlock, DirectionalConv3d
from triton_directional import TritonDirectionalConv3d
from triton_restore_norm import FusedRestoreNormBlock


class TritonBottleneckBlock(nn.Module):
    """Bottleneck block with both fused Triton kernels."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        reduced = hidden_dim // 4
        self.hidden_dim = hidden_dim
        self.reduced = reduced
        # Reduce stays as nn.Conv3d (Inductor fuses reduce+GELU well already)
        self.reduce = nn.Conv3d(hidden_dim, reduced, kernel_size=1, bias=False)
        # Triton DirectionalConv3d replaces the 7 nn.Linear pattern
        self.message = TritonDirectionalConv3d(reduced, reduced)
        # Fused: restore + residual add + LayerNorm
        self.fused_restore_norm = FusedRestoreNormBlock(hidden_dim, reduced)

    def forward(self, x):
        residual = x
        out = F.gelu(self.reduce(x))
        out = F.gelu(self.message(out))
        out = self.fused_restore_norm(out, residual)
        return out


def swap_to_full_triton(decoder):
    """Replace every BottleneckBlock with TritonBottleneckBlock, copying weights."""
    for name, module in list(decoder.named_modules()):
        if isinstance(module, BottleneckBlock):
            parent_name, _, child_name = name.rpartition(".")
            parent = decoder if parent_name == "" else decoder.get_submodule(parent_name)

            new_block = TritonBottleneckBlock(module.reduce.in_channels)
            # Copy reduce weight
            with torch.no_grad():
                new_block.reduce.weight.copy_(module.reduce.weight)
                # Copy DirectionalConv3d weights (7 linear modules -> packed [7, H_red, H_red])
                sd = {f"w_{d}.weight": getattr(module.message, f"w_{d}").weight.data for d in ["self","tp","tm","rp","rm","cp","cm"]}
                new_block.message.load_from_original(sd)
                # Copy restore + norm into fused kernel
                new_block.fused_restore_norm.load_from_original(
                    module.restore.weight.data,
                    module.norm.weight.data,
                    module.norm.bias.data,
                    module.norm.eps,
                )

            # Move to same device/dtype as original
            new_block = new_block.to(module.reduce.weight.device).to(module.reduce.weight.dtype)
            setattr(parent, child_name, new_block)
    return decoder
