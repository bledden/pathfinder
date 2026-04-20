"""PathfinderHybrid: CNN backbone (DirectionalConv3d) + global attention blocks
with 3D RoPE, SwiGLU FFN, pre-norm RMSNorm. Muon on 2D weights, AdamW on 1D.

Trained at 4-parameter circuit-level noise from scratch. 80K steps.

Usage:
  python3 train_hybrid.py --distance 7 --hidden_dim 192 --n_blocks 7 --attn_every 2 --steps 80000 --ckpt /workspace/persist/checkpoints/hybrid_d7
"""
import sys, os, time, math, argparse
sys.path.insert(0, "/workspace/pathfinder/train")
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import stim
from muon import SingleDeviceMuon
from model import DecoderConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class DirectionalConv3d(nn.Module):
    """7 separate learned projections (self, ±t, ±r, ±c)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.w = nn.Parameter(torch.randn(7, in_ch, out_ch) * (2.0 / in_ch) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x):
        B, C, T, H, W = x.shape
        # x as [B, T, H, W, C_in]
        y = x.permute(0, 2, 3, 4, 1).contiguous()
        out = torch.einsum('bthwi,io->bthwo', y, self.w[0])  # self
        # +t / -t
        out[:, 1:, :, :, :] += torch.einsum('bthwi,io->bthwo', y[:, :-1], self.w[1])
        out[:, :-1, :, :, :] += torch.einsum('bthwi,io->bthwo', y[:, 1:], self.w[2])
        # +r / -r
        out[:, :, 1:, :, :] += torch.einsum('bthwi,io->bthwo', y[:, :, :-1], self.w[3])
        out[:, :, :-1, :, :] += torch.einsum('bthwi,io->bthwo', y[:, :, 1:], self.w[4])
        # +c / -c
        out[:, :, :, 1:, :] += torch.einsum('bthwi,io->bthwo', y[:, :, :, :-1], self.w[5])
        out[:, :, :, :-1, :] += torch.einsum('bthwi,io->bthwo', y[:, :, :, 1:], self.w[6])
        out = out + self.bias
        return out.permute(0, 4, 1, 2, 3).contiguous()


def rope_freqs(dim, max_len):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)  # [max_len, dim/2]
    return torch.cat([freqs, freqs], dim=-1)  # [max_len, dim]


def apply_rope_axis(x, freqs_axis):
    """x: [B, H_heads, N, D]; freqs_axis: [N, D] — apply 1D RoPE along axis."""
    cos = freqs_axis.cos().to(x.dtype)  # [N, D]
    sin = freqs_axis.sin().to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    x_rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + x_rot * sin


class AttentionBlock(nn.Module):
    """Multi-head self-attention with 3D RoPE over (t, row, col)."""
    def __init__(self, dim, n_heads, grid_t, grid_h, grid_w):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        # RoPE splits head dim into 3 (t, h, w); each slice needs to be even
        assert self.head_dim % 6 == 0, f"head_dim {self.head_dim} must be divisible by 6 for 3D RoPE"
        self.rope_slice = self.head_dim // 3
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        # Pre-compute RoPE freqs per axis
        ft = rope_freqs(self.rope_slice, grid_t)
        fh = rope_freqs(self.rope_slice, grid_h)
        fw = rope_freqs(self.rope_slice, grid_w)
        self.register_buffer('freqs_t', ft, persistent=False)
        self.register_buffer('freqs_h', fh, persistent=False)
        self.register_buffer('freqs_w', fw, persistent=False)
        self.grid_t = grid_t; self.grid_h = grid_h; self.grid_w = grid_w

    def forward(self, x):
        # x: [B, C, T, H, W] → flatten to [B, N, C]
        B, C, T, H, W = x.shape
        N = T * H * W
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, N, C)

        qkv = self.qkv(x_flat).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]

        # Build 3D RoPE freqs per position in flat order
        # Flat order: for t in 0..T-1, for h in 0..H-1, for w in 0..W-1
        # Each position's freqs: cat[t_freqs[t], h_freqs[h], w_freqs[w]]
        ts = torch.arange(T, device=x.device).repeat_interleave(H * W)
        hs = torch.arange(H, device=x.device).repeat_interleave(W).repeat(T)
        ws = torch.arange(W, device=x.device).repeat(T * H)
        freqs_full = torch.cat([self.freqs_t[ts.clamp(0, self.freqs_t.shape[0]-1)], self.freqs_h[hs.clamp(0, self.freqs_h.shape[0]-1)], self.freqs_w[ws.clamp(0, self.freqs_w.shape[0]-1)]], dim=-1)  # [N, head_dim]
        freqs_full = freqs_full.unsqueeze(0).unsqueeze(0)  # [1, 1, N, head_dim]

        # Only rotate the first 2*rope_slice per axis (cos/sin part)
        # Simplification: apply RoPE to full head_dim treating freqs_full as the axis
        q_rot = apply_rope_axis(q, freqs_full.squeeze(0).squeeze(0))
        k_rot = apply_rope_axis(k, freqs_full.squeeze(0).squeeze(0))

        out = F.scaled_dot_product_attention(q_rot, k_rot, v)  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out(out)
        # Reshape back to [B, C, T, H, W]
        return out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(dim * 8 / 3)
            hidden_dim = ((hidden_dim + 63) // 64) * 64  # round to multiple of 64
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class HybridBlock(nn.Module):
    def __init__(self, dim, n_heads, use_attention, grid_t, grid_h, grid_w):
        super().__init__()
        self.use_attention = use_attention
        self.norm_conv = RMSNorm(dim)
        self.conv = DirectionalConv3d(dim, dim)
        if use_attention:
            self.norm_attn = RMSNorm(dim)
            self.attn = AttentionBlock(dim, n_heads, grid_t, grid_h, grid_w)
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def _apply_1d_norm(self, norm, x):
        # x: [B, C, T, H, W] → apply norm over C
        return norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()

    def _apply_1d_ffn(self, ffn, x):
        B, C, T, H, W = x.shape
        y = ffn(x.permute(0, 2, 3, 4, 1).reshape(-1, C))
        return y.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x):
        x = x + self.conv(self._apply_1d_norm(self.norm_conv, x))
        if self.use_attention:
            x = x + self.attn(self._apply_1d_norm(self.norm_attn, x))
        x = x + self._apply_1d_ffn(self.ffn, self._apply_1d_norm(self.norm_ffn, x))
        return x


class PathfinderHybrid(nn.Module):
    def __init__(self, d, r, H=192, L=None, n_heads=8, attn_every=2):
        super().__init__()
        self.L = L if L is not None else d
        self.embed = nn.Conv3d(1, H, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList([
            HybridBlock(H, n_heads=n_heads,
                        use_attention=((i + 1) % attn_every == 0),
                        grid_t=r+1, grid_h=d, grid_w=d)
            for i in range(self.L)
        ])
        self.final_norm = RMSNorm(H)
        self.head = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, 1))

    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        B, C, T, H, W = x.shape
        x = x.mean(dim=(2, 3, 4))  # global pool
        x = self.final_norm(x)
        return self.head(x)


class SyndromeDS4Param:
    def __init__(self, distance, rounds, p, batch=256):
        self.circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=distance, rounds=rounds,
            after_clifford_depolarization=p, before_measure_flip_probability=p,
            after_reset_flip_probability=p, before_round_data_depolarization=p,
        )
        self.sampler = self.circuit.compile_detector_sampler()
        nd = self.circuit.num_detectors
        coords = self.circuit.get_detector_coordinates()
        ac = np.array([coords[i] for i in range(nd)])
        sp, tm = ac[:, :-1], ac[:, -1]
        tu = np.sort(np.unique(tm))
        xu = np.sort(np.unique(sp[:, 0]))
        yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
        self.grid = (len(tu), len(yu), len(xu))
        tm_m = {v: i for i, v in enumerate(tu)}
        xm = {v: i for i, v in enumerate(xu)}
        ym = {v: i for i, v in enumerate(yu)}
        self.det_idx = np.zeros((nd, 3), dtype=np.int64)
        for did in range(nd):
            c = coords[did]
            self.det_idx[did] = [tm_m[c[-1]], ym.get(c[1], 0) if len(c) > 2 else 0, xm[c[0]]]
        self.nd = nd
        self.batch = batch

    def sample(self):
        det, obs = self.sampler.sample(self.batch, separate_observables=True)
        B = det.shape[0]; T, H, W = self.grid
        t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        d = torch.from_numpy(det.astype(np.float32))
        for i in range(self.nd):
            t[:, 0, self.det_idx[i, 0], self.det_idx[i, 1], self.det_idx[i, 2]] = d[:, i]
        return t, torch.from_numpy(obs.astype(np.float32))


class Curriculum:
    def __init__(self, target, total):
        self.t = target; self.n = total
    def get(self, s):
        f = s / max(self.n, 1)
        if f < 0.2: return self.t * 0.1
        elif f < 0.6: return self.t * (0.1 + 0.4 * (f - 0.2) / 0.4)
        else: return self.t * (0.5 + 0.5 * (f - 0.6) / 0.4)


def measure_ler(model, d, r, p, n=10000, batch=1000):
    model.eval()
    ds = SyndromeDS4Param(d, r, p, batch)
    errs = 0; total = 0
    with torch.no_grad():
        for _ in range(n // batch):
            syn, lab = ds.sample()
            syn = syn.to(device); lab = lab.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(syn)
            preds = (lg > 0).float()
            errs += int((preds != lab).any(dim=1).sum().item())
            total += batch
    model.train()
    return errs / total


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--distance", type=int, default=7)
    a.add_argument("--hidden_dim", type=int, default=192)
    a.add_argument("--n_blocks", type=int, default=None)
    a.add_argument("--n_heads", type=int, default=8)
    a.add_argument("--attn_every", type=int, default=2)
    a.add_argument("--steps", type=int, default=80000)
    a.add_argument("--batch", type=int, default=256)
    a.add_argument("--noise_rate", type=float, default=0.007)
    a.add_argument("--muon_lr", type=float, default=0.02)
    a.add_argument("--adam_lr", type=float, default=3e-3)
    a.add_argument("--log_interval", type=int, default=200)
    a.add_argument("--measure_interval", type=int, default=4000)
    a.add_argument("--ckpt", type=str, required=True)
    args = a.parse_args()
    os.makedirs(args.ckpt, exist_ok=True)

    config = DecoderConfig(distance=args.distance, rounds=args.distance, hidden_dim=args.hidden_dim)
    n_blocks = args.n_blocks if args.n_blocks else args.distance
    model = PathfinderHybrid(
        d=args.distance, r=args.distance,
        H=args.hidden_dim, L=n_blocks,
        n_heads=args.n_heads, attn_every=args.attn_every,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PathfinderHybrid d={args.distance} H={args.hidden_dim} L={n_blocks} heads={args.n_heads} attn_every={args.attn_every}: {n_params:,} params", flush=True)

    curric = Curriculum(args.noise_rate, args.steps)
    rates = sorted(set(round(curric.get(s), 4) for s in range(0, args.steps, max(args.steps // 50, 1))) | {round(args.noise_rate, 4)})
    samplers = {p: SyndromeDS4Param(args.distance, args.distance, max(p, 1e-6), args.batch) for p in rates}
    print(f"Pre-compiled {len(rates)} samplers (4-parameter noise).", flush=True)

    def get_ds(p): return samplers[min(samplers.keys(), key=lambda k: abs(k - p))]

    muon_params = [p for p in model.parameters() if p.ndim == 2 and p.requires_grad]
    adam_params = [p for p in model.parameters() if p.ndim != 2 and p.requires_grad]
    print(f"Optimizer split: {sum(p.numel() for p in muon_params):,} Muon params, {sum(p.numel() for p in adam_params):,} AdamW params", flush=True)
    opts = [SingleDeviceMuon(muon_params, lr=args.muon_lr, momentum=0.95, weight_decay=0.01),
            torch.optim.AdamW(adam_params, lr=args.adam_lr, weight_decay=0.0)]
    base_lrs = [[pg['lr'] for pg in opt.param_groups] for opt in opts]
    warmup = 1000

    def step_lr(s):
        if s < warmup: scale = s / warmup
        else:
            prog = (s - warmup) / max(args.steps - warmup, 1)
            scale = 0.5 * (1 + math.cos(math.pi * prog))
        for opt, lrs in zip(opts, base_lrs):
            for pg, lr in zip(opt.param_groups, lrs):
                pg['lr'] = lr * scale

    scaler = torch.amp.GradScaler("cuda")
    model.train()
    best = 1.0
    t0 = time.time()

    for step in range(args.steps):
        cp = curric.get(step)
        ds = get_ds(cp)
        syn, lab = ds.sample()
        syn = syn.to(device); lab = lab.to(device)
        for opt in opts: opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(syn)
            loss = F.binary_cross_entropy_with_logits(lg, lab)
        scaler.scale(loss).backward()
        for opt in opts: scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts: scaler.step(opt)
        scaler.update()
        step_lr(step)

        if step % args.log_interval == 0:
            sps = (step + 1) / max(time.time() - t0, 1)
            eta = (args.steps - step) / max(sps, 0.01) / 60
            print(f"step {step:>6}/{args.steps}  loss={loss.item():.4f}  p={cp:.5f}  {sps:.1f} s/s  ETA {eta:.0f}min", flush=True)

        if step > 0 and step % args.measure_interval == 0:
            ler = measure_ler(model, args.distance, args.distance, args.noise_rate)
            print(f"  >>> LER @ p={args.noise_rate}: {ler:.5f}", flush=True)
            if ler < best:
                best = ler
                torch.save({'step': step, 'model_state_dict': model.state_dict(), 'config': config, 'ler': ler,
                            'hybrid_hparams': {'H': args.hidden_dim, 'L': n_blocks, 'n_heads': args.n_heads, 'attn_every': args.attn_every}},
                           f"{args.ckpt}/best_model.pt")
                print(f"  >>> saved (ler={ler:.5f})", flush=True)

    final_ler = measure_ler(model, args.distance, args.distance, args.noise_rate, n=50000)
    print(f"\nFinal LER @ p={args.noise_rate} (50K shots): {final_ler:.5f}", flush=True)
    torch.save({'step': args.steps, 'model_state_dict': model.state_dict(), 'config': config, 'final_ler': final_ler,
                'hybrid_hparams': {'H': args.hidden_dim, 'L': n_blocks, 'n_heads': args.n_heads, 'attn_every': args.attn_every}},
               f"{args.ckpt}/final_model.pt")


if __name__ == "__main__":
    main()
