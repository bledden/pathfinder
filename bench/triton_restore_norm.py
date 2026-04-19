"""
Triton kernel fusing: restore conv1x1 + residual add + LayerNorm.
Replaces 3-4 Inductor-emitted kernels with a single launch per bottleneck block.

Block math:
  restored = message_gelu @ W_restore.T      # [B,H_red,T,R,C] @ [H_red,H] -> [B,H,T,R,C]
  added = restored + residual_x              # elementwise
  normalized = LayerNorm(added, dim=channel) * gamma + beta
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def restore_add_norm_kernel(
    MSG_ptr, RESIDUAL_ptr, W_REST_ptr, GAMMA_ptr, BETA_ptr, OUT_ptr,
    # Sizes
    B, T, R, Co, H_RED, H,
    # MSG strides: [B, H_red, T, R, Co] channel-first
    msg_sb, msg_sc, msg_st, msg_sr, msg_sco,
    # Residual strides: [B, H, T, R, Co] channel-first
    res_sb, res_sc, res_st, res_sr, res_sco,
    # W_restore strides: [H, H_red]
    w_sh, w_shr,
    # Out strides: [B, H, T, R, Co] channel-first
    out_sb, out_sc, out_st, out_sr, out_sco,
    EPS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    H_RED_CONST: tl.constexpr,
    H_CONST: tl.constexpr,
):
    """Each program: BLOCK_B batches × 1 spatial position, computes full channel dim H."""
    pid_spatial = tl.program_id(0)
    pid_b = tl.program_id(1)

    t = pid_spatial // (R * Co)
    r = (pid_spatial // Co) % R
    c = pid_spatial % Co

    b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offs < B

    h_red_off = tl.arange(0, H_RED_CONST)
    h_off = tl.arange(0, H_CONST)

    # Load message_gelu [BLOCK_B, H_RED] at this spatial position
    msg_ptrs = MSG_ptr + b_offs[:, None] * msg_sb + h_red_off[None, :] * msg_sc + t * msg_st + r * msg_sr + c * msg_sco
    msg = tl.load(msg_ptrs, mask=b_mask[:, None], other=0.0)

    # Load W_restore [H_RED, H] (transposed layout)
    # W_restore as stored: [H, H_red], so rows index H_out, cols index H_red
    # For matmul msg @ W^T we load W in [H, H_red] order and transpose on load.
    w_ptrs = W_REST_ptr + h_off[None, :] * w_sh + h_red_off[:, None] * w_shr
    w_restore = tl.load(w_ptrs)

    # restored = msg @ W_restore^T = [BLOCK_B, H_RED] @ [H_RED, H] -> [BLOCK_B, H]
    restored = tl.dot(msg.to(tl.float32), w_restore.to(tl.float32))

    # Load residual [BLOCK_B, H] at same position
    res_ptrs = RESIDUAL_ptr + b_offs[:, None] * res_sb + h_off[None, :] * res_sc + t * res_st + r * res_sr + c * res_sco
    residual = tl.load(res_ptrs, mask=b_mask[:, None], other=0.0).to(tl.float32)

    # Residual add
    y = restored + residual

    # LayerNorm per-sample across H channels
    mean = tl.sum(y, axis=1) / H_CONST
    y_centered = y - mean[:, None]
    var = tl.sum(y_centered * y_centered, axis=1) / H_CONST
    rstd = 1.0 / tl.sqrt(var + EPS)
    normalized = y_centered * rstd[:, None]

    # Gamma, beta
    gamma = tl.load(GAMMA_ptr + h_off).to(tl.float32)
    beta = tl.load(BETA_ptr + h_off).to(tl.float32)
    out = normalized * gamma[None, :] + beta[None, :]

    # Store output
    out_ptrs = OUT_ptr + b_offs[:, None] * out_sb + h_off[None, :] * out_sc + t * out_st + r * out_sr + c * out_sco
    tl.store(out_ptrs, out.to(OUT_ptr.dtype.element_ty), mask=b_mask[:, None])


def restore_add_norm(message_gelu: torch.Tensor, residual: torch.Tensor,
                     w_restore: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    """
    Args:
      message_gelu: [B, H_red, T, R, Co]
      residual:     [B, H,    T, R, Co]
      w_restore:    [H, H_red]  (as nn.Conv3d weight: [H, H_red, 1, 1, 1] squeezed)
      gamma, beta:  [H]
    Returns: [B, H, T, R, Co]
    """
    B, H_red, T, R, Co = message_gelu.shape
    _, H, _, _, _ = residual.shape
    assert w_restore.shape == (H, H_red), f"expected ({H},{H_red}), got {w_restore.shape}"

    out = torch.empty_like(residual)

    BLOCK_B = max(16, min(64, triton.next_power_of_2(B)))

    grid = (T * R * Co, triton.cdiv(B, BLOCK_B))
    restore_add_norm_kernel[grid](
        message_gelu, residual, w_restore, gamma, beta, out,
        B, T, R, Co, H_red, H,
        message_gelu.stride(0), message_gelu.stride(1), message_gelu.stride(2), message_gelu.stride(3), message_gelu.stride(4),
        residual.stride(0), residual.stride(1), residual.stride(2), residual.stride(3), residual.stride(4),
        w_restore.stride(0), w_restore.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        EPS=eps,
        BLOCK_B=BLOCK_B,
        H_RED_CONST=H_red,
        H_CONST=H,
    )
    return out


class FusedRestoreNormBlock(nn.Module):
    """Drop-in replacement for the back-half of BottleneckBlock:
       takes (message_gelu, residual) and returns the LayerNormed output.
       Replaces: self.restore(out), +residual, permute, LayerNorm, permute
    """
    def __init__(self, hidden_dim: int, reduced_dim: int):
        super().__init__()
        self.H = hidden_dim
        self.H_red = reduced_dim
        # Weight layout matches the squeezed Conv3d weight (H, H_red) — original module stores as (H, H_red, 1, 1, 1)
        self.w_restore = nn.Parameter(torch.zeros(hidden_dim, reduced_dim))
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = 1e-5

    def load_from_original(self, restore_weight, norm_gamma, norm_beta, eps):
        """restore_weight: [H, H_red, 1, 1, 1] from nn.Conv3d
           norm_gamma, norm_beta: [H] from nn.LayerNorm
           eps: LayerNorm's eps
        """
        with torch.no_grad():
            self.w_restore.copy_(restore_weight.squeeze())
            self.gamma.copy_(norm_gamma)
            self.beta.copy_(norm_beta)
        self.eps = eps

    def forward(self, message_gelu, residual):
        return restore_add_norm(message_gelu, residual, self.w_restore, self.gamma, self.beta, self.eps)
