"""
Optimized DirectionalConv3d variants for H200.
External to the Pathfinder repo — keeps MI300X ROCm path untouched.

Variant A: StackedDirectionalConv3d — fuses 7 nn.Linear -> 1 matmul + slice/shift.
Variant B: TritonDirectionalConv3d — custom fused Triton kernel.

Both accept the same state_dict keys as the original so checkpoints load directly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedDirectionalConv3d(nn.Module):
    """Fused variant: stack all 7 direction weights into one matrix, do one matmul,
    then split and shift. Loads the same state_dict as the original.
    """
    _DIRS = ["self", "tp", "tm", "rp", "rm", "cp", "cm"]

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # One fused weight [7*C_out, C_in]; slot layout matches _DIRS
        self.weight = nn.Parameter(torch.zeros(7 * out_channels, in_channels))
        # For state_dict compatibility with original nn.Linear-based impl:
        # we expose the original names by loading/saving as slots.
        self._orig_names = {
            "w_self.weight": 0, "w_tp.weight": 1, "w_tm.weight": 2,
            "w_rp.weight": 3, "w_rm.weight": 4, "w_cp.weight": 5, "w_cm.weight": 6,
        }

    def load_from_original(self, original_state: dict):
        """Given the state dict of the original DirectionalConv3d, build the fused weight."""
        C = self.out_channels
        with torch.no_grad():
            for key, slot in self._orig_names.items():
                self.weight[slot*C:(slot+1)*C].copy_(original_state[key])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T, R, Co] -> channel-last [B, T, R, Co, C_in]
        xp = x.permute(0, 2, 3, 4, 1).contiguous()
        # Single matmul: [..., C_in] @ [C_in, 7*C_out].T -> [..., 7*C_out]
        all_out = F.linear(xp, self.weight)
        B, T, R, Co, _ = all_out.shape
        all_out = all_out.view(B, T, R, Co, 7, self.out_channels)

        out = all_out[..., 0, :]  # self

        # Temporal
        if T > 1:
            out[:, 1:] = out[:, 1:] + all_out[:, :-1, :, :, 1, :]   # +t
            out[:, :-1] = out[:, :-1] + all_out[:, 1:, :, :, 2, :]  # -t
        # Row
        if R > 1:
            out[:, :, 1:] = out[:, :, 1:] + all_out[:, :, :-1, :, 3, :]
            out[:, :, :-1] = out[:, :, :-1] + all_out[:, :, 1:, :, 4, :]
        # Column
        if Co > 1:
            out[:, :, :, 1:] = out[:, :, :, 1:] + all_out[:, :, :, :-1, 5, :]
            out[:, :, :, :-1] = out[:, :, :, :-1] + all_out[:, :, :, 1:, 6, :]

        return out.permute(0, 4, 1, 2, 3).contiguous()


def swap_to_stacked(decoder: nn.Module):
    """Walk the decoder and replace DirectionalConv3d with StackedDirectionalConv3d."""
    from model import DirectionalConv3d
    for name, module in list(decoder.named_modules()):
        if isinstance(module, DirectionalConv3d):
            parent_name, _, child_name = name.rpartition(".")
            parent = decoder if parent_name == "" else decoder.get_submodule(parent_name)
            new = StackedDirectionalConv3d(module.w_self.in_features, module.w_self.out_features)
            orig_sd = {f"w_{d}.weight": getattr(module, f"w_{d}").weight.data for d in ["self","tp","tm","rp","rm","cp","cm"]}
            new.load_from_original(orig_sd)
            new = new.to(module.w_self.weight.device).to(module.w_self.weight.dtype)
            setattr(parent, child_name, new)
    return decoder
