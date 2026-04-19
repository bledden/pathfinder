"""
Pathfinder-Transformer: fully modern pure transformer decoder.

Inspired by: Llama 3, DiT, AlphaQubit.
No CNN — treats syndrome positions as tokens with 3D rotary position embeddings.

Modern ingredients:
  - Pure transformer (12 layers)
  - Multi-head attention with 3D rotary position embeddings (RoPE-3D)
  - RMSNorm (pre-norm)
  - SwiGLU FFN
  - Muon optimizer
  - Flash attention via torch.scaled_dot_product_attention
  - Attention pooling (CLS token)
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
        hidden = hidden or int(dim * 8 / 3 / 64) * 64
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def precompute_rope_3d(T, H, W, head_dim, device, base=10000.0):
    """3D rotary positional embedding: split head_dim into 3 parts (t, r, c)."""
    assert head_dim % 6 == 0, "head_dim must be divisible by 6 for 3D RoPE"
    per_axis = head_dim // 3  # dims per spatial axis

    def rope_freqs(seq_len, dims):
        theta = 1.0 / (base ** (torch.arange(0, dims, 2, device=device).float() / dims))
        pos = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(pos, theta)
        return torch.cos(freqs), torch.sin(freqs)

    ct, st = rope_freqs(T, per_axis)
    cr, sr = rope_freqs(H, per_axis)
    cc, sc = rope_freqs(W, per_axis)

    # Build [T, H, W, head_dim] cos and sin tensors
    # Each position (t, r, c) has rotation freqs: [t_cos, r_cos, c_cos] concatenated with matching sins
    cos = torch.zeros(T, H, W, head_dim, device=device)
    sin = torch.zeros(T, H, W, head_dim, device=device)
    for t in range(T):
        for r in range(H):
            for c in range(W):
                # Interleave: first per_axis dims for t, next per_axis for r, next per_axis for c
                # Each pair (2k, 2k+1) has cos, sin
                cos[t, r, c, 0:per_axis:2] = ct[t]
                sin[t, r, c, 0:per_axis:2] = st[t]
                cos[t, r, c, per_axis:2*per_axis:2] = cr[r]
                sin[t, r, c, per_axis:2*per_axis:2] = sr[r]
                cos[t, r, c, 2*per_axis::2] = cc[c]
                sin[t, r, c, 2*per_axis::2] = sc[c]
                cos[t, r, c, 1:per_axis:2] = ct[t]
                sin[t, r, c, 1:per_axis:2] = st[t]
                cos[t, r, c, per_axis+1:2*per_axis:2] = cr[r]
                sin[t, r, c, per_axis+1:2*per_axis:2] = sr[r]
                cos[t, r, c, 2*per_axis+1::2] = cc[c]
                sin[t, r, c, 2*per_axis+1::2] = sc[c]
    # Flatten to [T*H*W, head_dim]
    return cos.reshape(-1, head_dim), sin.reshape(-1, head_dim)


def apply_rope(q, k, cos, sin):
    """Apply rotary position embedding.
    q, k: [B, n_heads, N, head_dim]
    cos, sin: [N, head_dim]
    """
    # Rotate pairs of dimensions
    def rotate(x):
        # x: [..., head_dim], interpret as pairs
        d = x.shape[-1]
        x1 = x[..., :d//2]
        x2 = x[..., d//2:]
        return torch.cat([-x2, x1], dim=-1)
    # Simple rotation: x * cos + rotate(x) * sin
    q = q * cos.unsqueeze(0).unsqueeze(0) + rotate(q) * sin.unsqueeze(0).unsqueeze(0)
    k = k * cos.unsqueeze(0).unsqueeze(0) + rotate(k) * sin.unsqueeze(0).unsqueeze(0)
    return q, k


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos=None, sin=None):
        # x: [B, N, dim]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if cos is not None:
            q, k = apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x, cos=None, sin=None):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class PathfinderTransformer(nn.Module):
    """Pure transformer decoder for 3D surface-code syndromes."""
    def __init__(self, d, r, H=256, L=12, n_heads=8, T=None, spatial=None):
        super().__init__()
        self.d = d; self.r = r; self.H = H; self.L = L
        self.n_heads = n_heads
        self.head_dim = H // n_heads
        assert self.head_dim % 6 == 0, f"head_dim={self.head_dim} must be divisible by 6"
        # Embed: [B, 1, T, R, C] -> [B, N, H] with learned token embedding
        self.token_embed = nn.Linear(1, H)  # 1D binary -> H dims (applied after flatten)
        # CLS token for pooling
        self.cls = nn.Parameter(torch.randn(1, 1, H) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(H, n_heads) for _ in range(L)])
        self.norm = RMSNorm(H)
        self.head = nn.Linear(H, 1)
        # Precompute RoPE if grid size known
        self._rope_cache = {}

    def get_rope(self, T, H_, W):
        key = (T, H_, W)
        if key not in self._rope_cache:
            cos, sin = precompute_rope_3d(T, H_, W, self.head_dim, next(self.parameters()).device)
            self._rope_cache[key] = (cos, sin)
        return self._rope_cache[key]

    def forward(self, syn):
        # syn: [B, 1, T, H, W]
        B, _, T, H_, W = syn.shape
        N = T * H_ * W
        # Flatten spatial: [B, N, 1] -> [B, N, H]
        tokens = syn.permute(0, 2, 3, 4, 1).reshape(B, N, 1)
        x = self.token_embed(tokens)
        # Prepend CLS token (no rope on CLS, but simple: just concat)
        cls = self.cls.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        # RoPE for actual positions (skip CLS with 0 rotation)
        cos_pos, sin_pos = self.get_rope(T, H_, W)
        cos = torch.cat([torch.ones(1, self.head_dim, device=cos_pos.device), cos_pos], dim=0)
        sin = torch.cat([torch.zeros(1, self.head_dim, device=sin_pos.device), sin_pos], dim=0)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)
        return self.head(x[:, 0])  # CLS token output


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


def evaluate(m, d, r, p, n=10000, batch=500):
    ds = SyndromeDSCorrected(d, r, p, batch)
    errs = 0; total = 0
    for _ in range(n // batch):
        syn, lab = ds.sample()
        syn = syn.to(device); lab = lab.to(device)
        with torch.no_grad():
            lg = m(syn)
        preds = (lg > 0).float()
        errs += (preds != lab).any(dim=1).sum().item()
        total += batch
    return errs / max(total, 1)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--distance", type=int, default=7)
    a.add_argument("--hidden_dim", type=int, default=192)   # 192/8=24, divisible by 6 for 3D RoPE
    a.add_argument("--n_blocks", type=int, default=10)
    a.add_argument("--n_heads", type=int, default=8)
    a.add_argument("--steps", type=int, default=80000)
    a.add_argument("--batch", type=int, default=512)
    a.add_argument("--noise_rate", type=float, default=0.007)
    a.add_argument("--muon_lr", type=float, default=0.01)  # slightly lower for transformer stability
    a.add_argument("--adam_lr", type=float, default=2e-3)
    a.add_argument("--eval_interval", type=int, default=5000)
    a.add_argument("--log_interval", type=int, default=500)
    a.add_argument("--ckpt", type=str, default="/workspace/pathfinder/train/checkpoints/transformer_d7")
    args = a.parse_args()
    os.makedirs(args.ckpt, exist_ok=True)

    m = PathfinderTransformer(args.distance, args.distance, H=args.hidden_dim,
                              L=args.n_blocks, n_heads=args.n_heads).to(device)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Transformer Decoder: d={args.distance}, H={args.hidden_dim}, L={args.n_blocks}, heads={args.n_heads}, {n_params:,} params ({n_params*2/1e6:.2f}MB)", flush=True)

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
                            'n_heads': args.n_heads, 'transformer': True,
                            'ler': ler}, f"{args.ckpt}/best_model.pt")
                print(f"  >>> saved (ler={ler:.5f})", flush=True)
            m.train()


if __name__ == "__main__":
    main()
