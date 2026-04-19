"""
Fused Triton kernel for DirectionalConv3d.
7 direction-specific matmuls + boundary-padded adds, one launch.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _accumulate_dir(
    acc, X_ptr, W_ptr,
    src_t, src_r, src_c, valid,
    B, T, R, CO, C_IN, C_OUT,
    d_idx,
    stride_xb, stride_xt, stride_xr, stride_xc, stride_xf,
    stride_wd, stride_wo, stride_wi,
    b_offs, co_offs, ci_offs, b_mask, co_mask, ci_mask,
    BLOCK_B: tl.constexpr, BLOCK_CO: tl.constexpr, BLOCK_C_IN: tl.constexpr,
):
    if valid:
        x_ptrs = X_ptr + (b_offs[:, None] * stride_xb
                          + src_t * stride_xt + src_r * stride_xr + src_c * stride_xc
                          + ci_offs[None, :] * stride_xf)
        x_mask = b_mask[:, None] & ci_mask[None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = W_ptr + d_idx * stride_wd + co_offs[None, :] * stride_wo + ci_offs[:, None] * stride_wi
        w_mask = ci_mask[:, None] & co_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x.to(tl.float32), w.to(tl.float32))
    return acc


@triton.jit
def directional_conv3d_kernel(
    X_ptr, W_ptr, OUT_ptr,
    B, T, R, CO, C_IN, C_OUT,
    stride_xb, stride_xt, stride_xr, stride_xc, stride_xf,
    stride_wd, stride_wo, stride_wi,
    stride_ob, stride_ot, stride_or, stride_oc, stride_of,
    BLOCK_B: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    pid_co = tl.program_id(2)

    t = pid_spatial // (R * CO)
    r = (pid_spatial // CO) % R
    c = pid_spatial % CO

    b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offs < B
    co_offs = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_mask = co_offs < C_OUT
    ci_offs = tl.arange(0, BLOCK_C_IN)
    ci_mask = ci_offs < C_IN

    acc = tl.zeros((BLOCK_B, BLOCK_CO), dtype=tl.float32)

    # Direction 0: self (always valid)
    x_ptrs = X_ptr + (b_offs[:, None] * stride_xb
                      + t * stride_xt + r * stride_xr + c * stride_xc
                      + ci_offs[None, :] * stride_xf)
    x_mask = b_mask[:, None] & ci_mask[None, :]
    x = tl.load(x_ptrs, mask=x_mask, other=0.0)
    w_ptrs = W_ptr + 0 * stride_wd + co_offs[None, :] * stride_wo + ci_offs[:, None] * stride_wi
    w_mask = ci_mask[:, None] & co_mask[None, :]
    w = tl.load(w_ptrs, mask=w_mask, other=0.0)
    acc += tl.dot(x.to(tl.float32), w.to(tl.float32))

    # Direction 1: +t (source t-1)
    if t > 0:
        src_t = t - 1
        x_ptrs = X_ptr + (b_offs[:, None] * stride_xb
                          + src_t * stride_xt + r * stride_xr + c * stride_xc
                          + ci_offs[None, :] * stride_xf)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_ptrs = W_ptr + 1 * stride_wd + co_offs[None, :] * stride_wo + ci_offs[:, None] * stride_wi
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x.to(tl.float32), w.to(tl.float32))

    # Direction 2: -t (source t+1)
    if t < T - 1:
        src_t = t + 1
        x_ptrs = X_ptr + (b_offs[:, None] * stride_xb
                          + src_t * stride_xt + r * stride_xr + c * stride_xc
                          + ci_offs[None, :] * stride_xf)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_ptrs = W_ptr + 2 * stride_wd + co_offs[None, :] * stride_wo + ci_offs[:, None] * stride_wi
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x.to(tl.float32), w.to(tl.float32))

    # Direction 3: +r (source r-1)
    if r > 0:
        src_r = r - 1
        x_ptrs = X_ptr + (b_offs[:, None] * stride_xb
                          + t * stride_xt + src_r * stride_xr + c * stride_xc
                          + ci_offs[None, :] * stride_xf)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_ptrs = W_ptr + 3 * stride_wd + co_offs[None, :] * stride_wo + ci_offs[:, None] * stride_wi
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x.to(tl.float32), w.to(tl.float32))

    # Direction 4: -r (source r+1)
    if r < R - 1:
        src_r = r + 1
        x_ptrs = X_ptr + (b_offs[:, None] * stride_xb
                          + t * stride_xt + src_r * stride_xr + c * stride_xc
                          + ci_offs[None, :] * stride_xf)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_ptrs = W_ptr + 4 * stride_wd + co_offs[None, :] * stride_wo + ci_offs[:, None] * stride_wi
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x.to(tl.float32), w.to(tl.float32))

    # Direction 5: +c (source c-1)
    if c > 0:
        src_c = c - 1
        x_ptrs = X_ptr + (b_offs[:, None] * stride_xb
                          + t * stride_xt + r * stride_xr + src_c * stride_xc
                          + ci_offs[None, :] * stride_xf)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_ptrs = W_ptr + 5 * stride_wd + co_offs[None, :] * stride_wo + ci_offs[:, None] * stride_wi
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x.to(tl.float32), w.to(tl.float32))

    # Direction 6: -c (source c+1)
    if c < CO - 1:
        src_c = c + 1
        x_ptrs = X_ptr + (b_offs[:, None] * stride_xb
                          + t * stride_xt + r * stride_xr + src_c * stride_xc
                          + ci_offs[None, :] * stride_xf)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_ptrs = W_ptr + 6 * stride_wd + co_offs[None, :] * stride_wo + ci_offs[:, None] * stride_wi
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x.to(tl.float32), w.to(tl.float32))

    out_ptrs = OUT_ptr + (b_offs[:, None] * stride_ob
                          + t * stride_ot + r * stride_or + c * stride_oc
                          + co_offs[None, :] * stride_of)
    out_mask = b_mask[:, None] & co_mask[None, :]
    tl.store(out_ptrs, acc.to(OUT_ptr.dtype.element_ty), mask=out_mask)


class TritonDirectionalConv3d(nn.Module):
    _DIRS = ["self", "tp", "tm", "rp", "rm", "cp", "cm"]

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.zeros(7, out_channels, in_channels))

    def load_from_original(self, sd):
        with torch.no_grad():
            for i, d in enumerate(self._DIRS):
                self.weight[i].copy_(sd[f"w_{d}.weight"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp = x.permute(0, 2, 3, 4, 1).contiguous()
        B, T, R, Co, C_in = xp.shape
        C_out = self.out_channels
        out = torch.empty(B, T, R, Co, C_out, device=x.device, dtype=x.dtype)

        BLOCK_B = max(16, min(64, triton.next_power_of_2(max(B, 1))))
        BLOCK_CO = min(64, triton.next_power_of_2(max(C_out, 1)))
        BLOCK_C_IN = max(16, triton.next_power_of_2(max(C_in, 1)))

        grid = (triton.cdiv(B, BLOCK_B), T * R * Co, triton.cdiv(C_out, BLOCK_CO))
        directional_conv3d_kernel[grid](
            xp, self.weight, out,
            B, T, R, Co, C_in, C_out,
            xp.stride(0), xp.stride(1), xp.stride(2), xp.stride(3), xp.stride(4),
            self.weight.stride(0), self.weight.stride(1), self.weight.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            BLOCK_B=BLOCK_B, BLOCK_CO=BLOCK_CO, BLOCK_C_IN=BLOCK_C_IN,
        )
        return out.permute(0, 4, 1, 2, 3).contiguous()


def swap_to_triton(decoder: nn.Module):
    from model import DirectionalConv3d
    for name, module in list(decoder.named_modules()):
        if isinstance(module, DirectionalConv3d):
            parent_name, _, child_name = name.rpartition(".")
            parent = decoder if parent_name == "" else decoder.get_submodule(parent_name)
            new_mod = TritonDirectionalConv3d(module.w_self.in_features, module.w_self.out_features)
            sd = {f"w_{d}.weight": getattr(module, f"w_{d}").weight.data for d in ["self","tp","tm","rp","rm","cp","cm"]}
            new_mod.load_from_original(sd)
            new_mod = new_mod.to(module.w_self.weight.device).to(module.w_self.weight.dtype)
            setattr(parent, child_name, new_mod)
    return decoder
