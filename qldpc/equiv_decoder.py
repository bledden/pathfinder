"""Group-equivariant (Z_l x Z_m) neural decoder for bivariate-bicycle codes.

Design (avoids the equivariance trap from AED-QC-LDPC arXiv:2202.00287):
  - The BB syndrome (N=l*m checks) is reshaped to an (l, m) torus grid (the check group Z_l x Z_m).
  - BODY: a stack of 2D CIRCULAR convolutions on the (l,m) torus. For the abelian group Z_l x Z_m,
    2D cyclic convolution IS group convolution -> EXACTLY Z_l x Z_m-equivariant, weight-shared
    across all 36 group elements. This is where the sample-efficiency / param-sharing benefit lives.
  - HEAD: a POSITION-AWARE linear readout from the full (channels x l x m) feature map to the k
    logicals. NOT global-pool (global pool would make the decoder group-INVARIANT, which is wrong
    because the group permutes logical sectors). The head breaks equivariance deliberately, exactly
    as a CNN image classifier's FC head does.

Matched-capacity NON-equivariant baseline (MLPDecoder): same total parameter budget, but a plain
MLP on the flat syndrome -> no weight sharing across the group. The sample-efficiency comparison
(equiv vs MLP at matched params, swept over training-set size) is the hypothesis.

This module: architecture + a self-test that VERIFIES the body is genuinely group-equivariant
(feed g.s, confirm body feature map = g.(body feature map)) before any training.
"""
import os
import numpy as np
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import torch
import torch.nn as nn


class CircConv2d(nn.Module):
    """2D convolution with circular padding on both axes = group conv on Z_l x Z_m."""
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, padding_mode='circular', bias=True)

    def forward(self, x):
        return self.conv(x)


class EquivBBDecoder(nn.Module):
    def __init__(self, l=6, m=6, k=12, hidden=64, depth=4):
        super().__init__()
        self.l, self.m, self.k = l, m, k
        layers = [CircConv2d(1, hidden)]
        for _ in range(depth - 1):
            layers += [nn.GELU(), CircConv2d(hidden, hidden)]
        self.body = nn.Sequential(*layers)
        # position-aware head: full feature map (hidden*l*m) -> k logicals
        self.head = nn.Linear(hidden * l * m, k)

    def forward(self, s):
        # s: (B, N) binary syndrome -> (B, 1, l, m)
        B = s.shape[0]
        x = s.view(B, 1, self.l, self.m).float()
        f = self.body(x)                       # (B, hidden, l, m), equivariant
        return self.head(f.reshape(B, -1))     # (B, k) logits


class MLPDecoder(nn.Module):
    """Matched-capacity non-equivariant baseline: plain MLP on flat syndrome."""
    def __init__(self, N=36, k=12, width=256, depth=4):
        super().__init__()
        layers = [nn.Linear(N, width), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, k)]
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        return self.net(s.float())


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def verify_body_equivariance(seed=0):
    """The decisive correctness check: the conv BODY must satisfy body(g.s) == g.body(s)
    for all group elements g = cyclic shift (a,b) on the (l,m) torus. Tests all 36."""
    torch.manual_seed(seed)
    l = m = 6
    dec = EquivBBDecoder(l, m)
    dec.eval()
    s = torch.randint(0, 2, (8, l * m)).float()
    x = s.view(8, 1, l, m)
    with torch.no_grad():
        f0 = dec.body(x)                       # (8, H, l, m)
    max_err = 0.0
    for a in range(l):
        for b in range(m):
            xg = torch.roll(x, shifts=(a, b), dims=(2, 3))   # g.s
            with torch.no_grad():
                fg = dec.body(xg)
            f0g = torch.roll(f0, shifts=(a, b), dims=(2, 3))  # g.(body(s))
            err = (fg - f0g).abs().max().item()
            max_err = max(max_err, err)
    return max_err


if __name__ == '__main__':
    import json
    out = {}
    eq = EquivBBDecoder()
    mlp = MLPDecoder(width=256)
    out['equiv_params'] = count_params(eq)
    out['mlp_params_w256'] = count_params(mlp)
    # match MLP width to equiv param count
    target = out['equiv_params']
    best_w, best_diff = 256, 1e18
    for w in range(64, 1200, 8):
        p = count_params(MLPDecoder(width=w))
        if abs(p - target) < best_diff:
            best_diff, best_w = abs(p - target), w
    out['mlp_matched_width'] = best_w
    out['mlp_matched_params'] = count_params(MLPDecoder(width=best_w))
    out['body_equivariance_max_err'] = verify_body_equivariance()
    out['EQUIVARIANT_PASS'] = bool(out['body_equivariance_max_err'] < 1e-4)
    # forward smoke
    s = torch.randint(0, 2, (4, 36)).float()
    out['equiv_out_shape'] = list(eq(s).shape)
    out['mlp_out_shape'] = list(mlp(s).shape)
    json.dump(out, open(os.path.join(_OUT, 'equiv_arch_selftest.json'), 'w'), indent=2)
    print("WROTE equiv_arch_selftest.json")
