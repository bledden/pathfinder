"""Verify fused restore+add+layernorm matches reference."""
import sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/workspace")
from triton_restore_norm import FusedRestoreNormBlock, restore_add_norm

torch.manual_seed(0)

for (B, H, H_red, T, R, Co) in [(1, 256, 64, 7, 7, 7), (16, 256, 64, 5, 5, 5), (1024, 256, 64, 7, 7, 7)]:
    # Original ops
    restore = nn.Conv3d(H_red, H, kernel_size=1, bias=False).cuda()
    norm = nn.LayerNorm(H).cuda()

    # Inputs
    msg_gelu = torch.randn(B, H_red, T, R, Co, device="cuda")
    residual = torch.randn(B, H, T, R, Co, device="cuda")

    # Reference
    with torch.no_grad():
        restored = restore(msg_gelu)
        out_ref = restored + residual
        out_ref = out_ref.permute(0, 2, 3, 4, 1)
        out_ref = norm(out_ref)
        out_ref = out_ref.permute(0, 4, 1, 2, 3).contiguous()

    # Fused Triton
    fused = FusedRestoreNormBlock(H, H_red).cuda()
    fused.load_from_original(restore.weight.data, norm.weight.data, norm.bias.data, norm.eps)
    with torch.no_grad():
        out_new = fused(msg_gelu, residual)

    # Compare
    diff = (out_ref - out_new).abs().max().item()
    rel = diff / out_ref.abs().max().item()
    print(f"B={B} T={T} R={R}: max|diff|={diff:.3e}  rel={rel:.3e}  {'PASS' if rel < 1e-3 else 'FAIL'}")

    # FP16
    restore_h = restore.half()
    norm_h = norm.half()
    msg_h = msg_gelu.half()
    res_h = residual.half()
    with torch.no_grad():
        restored_h = restore_h(msg_h)
        out_ref_h = restored_h + res_h
        out_ref_h = out_ref_h.permute(0, 2, 3, 4, 1)
        out_ref_h = norm_h(out_ref_h)
        out_ref_h = out_ref_h.permute(0, 4, 1, 2, 3).contiguous()

    fused_h = FusedRestoreNormBlock(H, H_red).cuda().half()
    fused_h.load_from_original(restore.weight.data.half(), norm.weight.data.half(), norm.bias.data.half(), norm.eps)
    with torch.no_grad():
        out_new_h = fused_h(msg_h, res_h)

    diff_h = (out_ref_h - out_new_h).abs().max().item()
    rel_h = diff_h / out_ref_h.abs().max().item() if out_ref_h.abs().max() > 0 else 0
    print(f"  FP16 diff={diff_h:.3e}  rel={rel_h:.3e}  {'PASS' if rel_h < 5e-3 else 'FAIL'}")
