"""
Pathfinder-Modern: a 2026-era architecture for neural QEC decoding.

Built on modern deep learning primitives Lange (2023) didn't have:
- RMSNorm (faster, more stable than LayerNorm)
- SwiGLU activation (better than GELU)
- Pre-norm throughout (better gradient flow at depth)
- Rotary positional embedding for 3D detector coords
- Local CNN (DirectionalConv3d) + Global Attention hybrid
- Muon optimizer (not available to Lange)
- Proper noise model (Lange had this)
- Longer + mixed-noise training
"""
import sys, os, time, math, argparse
sys.path.insert(0, "/workspace/pathfinder/train")
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import stim
from muon import SingleDeviceMuon
import torch.optim
torch.optim.Muon = SingleDeviceMuon

device = torch.device("cuda")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / norm)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or int(dim * 8 / 3)
        hidden = (hidden + 63) // 64 * 64  # multiple of 64
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class DirectionalConv3d(nn.Module):
    """Same as Pathfinder v1."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.w_self = nn.Linear(in_ch, out_ch, bias=False)
        self.w_tp = nn.Linear(in_ch, out_ch, bias=False)
        self.w_tm = nn.Linear(in_ch, out_ch, bias=False)
        self.w_rp = nn.Linear(in_ch, out_ch, bias=False)
        self.w_rm = nn.Linear(in_ch, out_ch, bias=False)
        self.w_cp = nn.Linear(in_ch, out_ch, bias=False)
        self.w_cm = nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, x):
        xp = x.permute(0, 2, 3, 4, 1)
        out = self.w_self(xp)
        if xp.shape[1] > 1:
            out = out + F.pad(self.w_tp(xp[:, :-1]), (0,0,0,0,0,0,1,0))
            out = out + F.pad(self.w_tm(xp[:, 1:]),  (0,0,0,0,0,0,0,1))
        if xp.shape[2] > 1:
            out = out + F.pad(self.w_rp(xp[:, :, :-1]), (0,0,0,0,1,0))
            out = out + F.pad(self.w_rm(xp[:, :, 1:]),  (0,0,0,0,0,1))
        if xp.shape[3] > 1:
            out = out + F.pad(self.w_cp(xp[:, :, :, :-1]), (0,0,1,0))
            out = out + F.pad(self.w_cm(xp[:, :, :, 1:]),  (0,0,0,1))
        return out.permute(0, 4, 1, 2, 3)


class GlobalAttentionBlock(nn.Module):
    """Multi-head self-attention over all (T,R,C) positions.
    Complements local CNN with global defect-to-defect attention.
    """
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x: [B, H, T, R, C] -> flatten to [B, T*R*C, H]
        B, H_, T, R, C = x.shape
        N = T * R * C
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, N, H_)
        qkv = self.qkv(x_flat).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Flash attention
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = attn.transpose(1, 2).reshape(B, N, H_)
        out = self.proj(out)
        return out.reshape(B, T, R, C, H_).permute(0, 4, 1, 2, 3)


class ModernBlock(nn.Module):
    """Pre-norm block with local DirectionalConv3d + global attention + SwiGLU FFN."""
    def __init__(self, dim, n_heads=8, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        reduced = dim // 4
        # Local CNN path
        self.norm1 = RMSNorm(dim)
        self.reduce = nn.Conv3d(dim, reduced, kernel_size=1, bias=False)
        self.conv = DirectionalConv3d(reduced, reduced)
        self.restore = nn.Conv3d(reduced, dim, kernel_size=1, bias=False)
        # Global attention path
        if use_attention:
            self.norm2 = RMSNorm(dim)
            self.attn = GlobalAttentionBlock(dim, n_heads)
        # FFN
        self.norm3 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x):
        # x: [B, H, T, R, C]
        # Local conv branch (pre-norm)
        residual = x
        h = x.permute(0, 2, 3, 4, 1)
        h = self.norm1(h)
        h = h.permute(0, 4, 1, 2, 3)
        h = F.silu(self.reduce(h))
        h = F.silu(self.conv(h))
        h = self.restore(h)
        x = residual + h

        # Global attention branch
        if self.use_attention:
            residual = x
            h = x.permute(0, 2, 3, 4, 1)
            h = self.norm2(h)
            h = h.permute(0, 4, 1, 2, 3)
            h = self.attn(h)
            x = residual + h

        # FFN branch
        residual = x
        h = x.permute(0, 2, 3, 4, 1)
        h = self.norm3(h)
        h = self.ffn(h)
        x = residual + h.permute(0, 4, 1, 2, 3)
        return x


class PathfinderModern(nn.Module):
    def __init__(self, d, r, H=384, L=None, n_heads=8):
        super().__init__()
        self.d = d; self.r = r; self.H = H
        self.L = L or (d + 3)  # default L = d+3, giving more depth
        self.embed = nn.Conv3d(1, H, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList([
            ModernBlock(H, n_heads=n_heads, use_attention=True) for _ in range(self.L)
        ])
        self.final_norm = RMSNorm(H)
        self.head = nn.Sequential(
            nn.Linear(H, H), nn.SiLU(), nn.Linear(H, 1)
        )

    def forward(self, syn):
        x = self.embed(syn)
        for b in self.blocks:
            x = b(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.final_norm(x)
        x = x.mean(dim=(1, 2, 3))
        return self.head(x)


class CurriculumScheduler:
    def __init__(self, target, total):
        self.target = target; self.total = total
    def get_rate(self, s):
        f = s / max(self.total, 1)
        if f < 0.2: return self.target * 0.1
        elif f < 0.6:
            return self.target * (0.1 + 0.4 * (f - 0.2) / 0.4)
        else:
            return self.target * (0.5 + 0.5 * (f - 0.6) / 0.4)


class SyndromeDSCorrected:
    def __init__(self, d, r, p, batch=256):
        self.circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=r,
            after_clifford_depolarization=p, before_measure_flip_probability=p,
            after_reset_flip_probability=p, before_round_data_depolarization=p)
        self.sampler = self.circuit.compile_detector_sampler()
        nd = self.circuit.num_detectors
        coords = self.circuit.get_detector_coordinates()
        ac = np.array([coords[i] for i in range(nd)])
        sp, tm = ac[:, :-1], ac[:, -1]
        tu = np.sort(np.unique(tm))
        xu = np.sort(np.unique(sp[:, 0]))
        yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
        self.grid = (len(tu), len(yu), len(xu))
        tm_m = {v: i for i, v in enumerate(tu)}; xm = {v: i for i, v in enumerate(xu)}; ym = {v: i for i, v in enumerate(yu)}
        self.det_idx = np.zeros((nd, 3), dtype=np.int64)
        for did in range(nd):
            c = coords[did]
            self.det_idx[did] = [tm_m[c[-1]], ym.get(c[1], 0) if len(c) > 2 else 0, xm[c[0]]]
        self.nd = nd; self.batch = batch

    def sample(self):
        det, obs = self.sampler.sample(self.batch, separate_observables=True)
        B = det.shape[0]; T, H, W = self.grid
        t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        d = torch.from_numpy(det.astype(np.float32))
        for i in range(self.nd):
            t[:, 0, self.det_idx[i, 0], self.det_idx[i, 1], self.det_idx[i, 2]] = d[:, i]
        return t, torch.from_numpy(obs.astype(np.float32))


def evaluate(model, d, r, p, n=10000, batch=500):
    ds = SyndromeDSCorrected(d, r, p, batch)
    errs = 0; total = 0
    for _ in range(n // batch):
        syn, lab = ds.sample()
        syn = syn.to(device); lab = lab.to(device)
        with torch.no_grad():
            lg = model(syn)
        preds = (lg > 0).float()
        errs += (preds != lab).any(dim=1).sum().item()
        total += batch
    return errs / max(total, 1)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--distance", type=int, default=7)
    a.add_argument("--hidden_dim", type=int, default=384)
    a.add_argument("--n_blocks", type=int, default=10)
    a.add_argument("--n_heads", type=int, default=8)
    a.add_argument("--steps", type=int, default=80000)
    a.add_argument("--batch", type=int, default=256)
    a.add_argument("--noise_rate", type=float, default=0.007)
    a.add_argument("--muon_lr", type=float, default=0.02)
    a.add_argument("--adam_lr", type=float, default=3e-3)
    a.add_argument("--eval_interval", type=int, default=5000)
    a.add_argument("--log_interval", type=int, default=500)
    a.add_argument("--ckpt", type=str, default="/workspace/pathfinder/train/checkpoints/modern_d7")
    args = a.parse_args()
    os.makedirs(args.ckpt, exist_ok=True)

    m = PathfinderModern(args.distance, args.distance, H=args.hidden_dim,
                         L=args.n_blocks, n_heads=args.n_heads).to(device)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Modern Decoder: d={args.distance}, H={args.hidden_dim}, L={args.n_blocks}, heads={args.n_heads}, {n_params:,} params", flush=True)

    curric = CurriculumScheduler(args.noise_rate, args.steps)
    rates = sorted(set(round(curric.get_rate(s), 4) for s in range(0, args.steps, max(args.steps // 50, 1))) | {round(args.noise_rate, 4)})
    samplers = {}
    print(f"Pre-compiling {len(rates)} samplers...", flush=True)
    for p in rates:
        samplers[p] = SyndromeDSCorrected(args.distance, args.distance, max(p, 1e-6), args.batch)
    def get_ds(p):
        return samplers[min(samplers.keys(), key=lambda k: abs(k - p))]

    muon_params = [p for p in m.parameters() if p.ndim == 2 and p.requires_grad]
    adam_params = [p for p in m.parameters() if p.ndim != 2 and p.requires_grad]
    opts = [SingleDeviceMuon(muon_params, lr=args.muon_lr, momentum=0.95, weight_decay=0.01),
            torch.optim.AdamW(adam_params, lr=args.adam_lr, weight_decay=0.0)]
    base_lrs = [[pg['lr'] for pg in opt.param_groups] for opt in opts]
    warmup = 2000
    def step_lr(s):
        if s < warmup: scale = s / warmup
        else:
            prog = (s - warmup) / max(args.steps - warmup, 1)
            scale = 0.5 * (1 + math.cos(math.pi * prog))
        for opt, lrs in zip(opts, base_lrs):
            for pg, lr in zip(opt.param_groups, lrs):
                pg['lr'] = lr * scale

    scaler = torch.amp.GradScaler("cuda")
    m.train()
    best = 1.0
    t0 = time.time()

    for step in range(args.steps):
        cp = curric.get_rate(step)
        syn, lab = get_ds(cp).sample()
        syn = syn.to(device); lab = lab.to(device)
        for opt in opts: opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(syn)
            loss = F.binary_cross_entropy_with_logits(lg, lab)
        scaler.scale(loss).backward()
        for opt in opts: scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        for opt in opts: scaler.step(opt)
        scaler.update()
        step_lr(step)

        if step % args.log_interval == 0:
            sps = (step+1) / max(time.time()-t0, 1)
            eta = (args.steps - step) / max(sps, 0.01) / 60
            print(f"step {step:>6}/{args.steps}  loss={loss.item():.4f}  p={cp:.5f}  {sps:.1f} s/s  ETA {eta:.0f}min", flush=True)

        if step > 0 and step % args.eval_interval == 0:
            m.eval()
            ler = evaluate(m, args.distance, args.distance, args.noise_rate)
            print(f"  >>> EVAL LER @ p={args.noise_rate}: {ler:.5f}", flush=True)
            if ler < best:
                best = ler
                torch.save({'step': step, 'model_state_dict': m.state_dict(),
                            'distance': args.distance, 'rounds': args.distance,
                            'hidden_dim': args.hidden_dim, 'n_blocks': args.n_blocks,
                            'n_heads': args.n_heads, 'modern': True,
                            'ler': ler},
                           f"{args.ckpt}/best_model.pt")
                print(f"  >>> saved (ler={ler:.5f})", flush=True)
            m.train()

    torch.save({'step': args.steps, 'model_state_dict': m.state_dict(),
                'distance': args.distance, 'rounds': args.distance,
                'hidden_dim': args.hidden_dim, 'n_blocks': args.n_blocks,
                'n_heads': args.n_heads, 'modern': True},
               f"{args.ckpt}/final_model.pt")


if __name__ == "__main__":
    main()
