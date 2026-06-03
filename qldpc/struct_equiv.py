"""Structure-aware group-algebra convolution for BB-code decoding.

DIAGNOSIS of the negative result: the first equivariant decoder used generic 3x3 circular convs.
But the BB code's checks connect data qubits at the POLYNOMIAL offsets of A = x^3+y+y^2 and
B = y^3+x+x^2, i.e. offsets {(3,0),(0,1),(0,2)} and {(0,3),(1,0),(2,0)} on the (6,6) torus.
A 3x3 kernel only covers (+-1,+-1) -> it cannot even see the qubits each check actually touches.
Right symmetry (Z6xZ6 translation), WRONG kernel support.

FIX (this file): a group-algebra convolution whose kernel support is EXACTLY the union of the
monomial offsets of A and B (and identity + inverses). Output[g] = sum_{h in S} W_h . x[g-h].
This is still Z6xZ6-equivariant (it's a convolution = commutes with translation) but now its
receptive field matches the code's Tanner structure -> the correct inductive bias for one
"message-passing-like" step. Verified equivariant before any training.
"""
import os
import numpy as np
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import torch
import torch.nn as nn


def bb_offsets(l=6, m=6):
    """Support set S on the (l,m) torus: identity + monomial offsets of A,B + their inverses.
    A = x^3 + y + y^2 -> (3,0),(0,1),(0,2);  B = y^3 + x + x^2 -> (0,3),(1,0),(2,0)."""
    base = [(0, 0), (3, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0)]
    S = set()
    for (a, b) in base:
        S.add((a % l, b % m))
        S.add(((-a) % l, (-b) % m))   # inverses for the transpose / both message directions
    return sorted(S)


class StructGroupConv(nn.Module):
    """Group-algebra conv on Z_l x Z_m with kernel support = `offsets` (a list of (di,dj)).
    x: (B, Cin, l, m) -> (B, Cout, l, m). Equivariant to torus translation by construction."""
    def __init__(self, cin, cout, offsets, l=6, m=6):
        super().__init__()
        self.offsets = offsets
        self.l, self.m = l, m
        # one (cout x cin) weight per offset
        self.W = nn.Parameter(torch.empty(len(offsets), cout, cin))
        self.bias = nn.Parameter(torch.zeros(cout))
        nn.init.kaiming_uniform_(self.W, a=5 ** 0.5)

    def forward(self, x):
        out = 0.0
        for idx, (di, dj) in enumerate(self.offsets):
            xs = torch.roll(x, shifts=(di, dj), dims=(2, 3))   # x[g-h]
            # contract cin: (B,Cin,l,m) with W[idx] (Cout,Cin) -> (B,Cout,l,m)
            out = out + torch.einsum('oc,bcij->boij', self.W[idx], xs)
        return out + self.bias.view(1, -1, 1, 1)


class StructEquivBBDecoder(nn.Module):
    def __init__(self, l=6, m=6, k=12, hidden=64, depth=4):
        super().__init__()
        self.l, self.m = l, m
        offs = bb_offsets(l, m)
        self.offsets = offs
        layers = [StructGroupConv(1, hidden, offs, l, m)]
        for _ in range(depth - 1):
            layers += [nn.GELU(), StructGroupConv(hidden, hidden, offs, l, m)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden * l * m, k)

    def forward(self, s):
        B = s.shape[0]
        f = self.body(s.view(B, 1, self.l, self.m).float())
        return self.head(f.reshape(B, -1))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def verify_equivariance(seed=0):
    torch.manual_seed(seed)
    dec = StructEquivBBDecoder()
    dec.eval()
    x = torch.randint(0, 2, (8, 1, 6, 6)).float()
    with torch.no_grad():
        f0 = dec.body(x)
    max_err = 0.0
    for a in range(6):
        for b in range(6):
            xg = torch.roll(x, shifts=(a, b), dims=(2, 3))
            with torch.no_grad():
                fg = dec.body(xg)
            f0g = torch.roll(f0, shifts=(a, b), dims=(2, 3))
            max_err = max(max_err, (fg - f0g).abs().max().item())
    return max_err


if __name__ == '__main__':
    import json
    offs = bb_offsets()
    dec = StructEquivBBDecoder()
    out = dict(offsets=[list(o) for o in offs], n_offsets=len(offs),
               struct_params=count_params(dec),
               body_equivariance_max_err=verify_equivariance(),
               note='generic 3x3 conv support is {(-1..1)x(-1..1)}; this support matches A,B monomials')
    out['EQUIVARIANT_PASS'] = bool(out['body_equivariance_max_err'] < 1e-4)
    json.dump(out, open(os.path.join(_OUT, 'struct_equiv_selftest.json'), 'w'), indent=2)
    print("WROTE struct_equiv_selftest.json")
