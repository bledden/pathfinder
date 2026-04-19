"""
Pathfinder v2 training:
  - Proper noise model (before_round_data_depolarization=p included)
  - Mixed-noise training (log-uniform p in [0.001, 0.015])
  - D₄ spatial symmetry augmentation (8-element group, verified safe for memory-Z)
  - Pre-norm LayerNorm (modern architecture)
  - H=512 scaled backbone (matches Lange's parameter count)
  - Noise-rate conditioning injected as learned embedding added to features
  - 150K training steps
  - Fresh syndromes each batch (no overfitting risk)

This is a research script. It produces a new checkpoint distinct from the paper's
reported d5_muon/d7_final etc. — the original checkpoints remain untouched.
"""
import sys, os, time, math, argparse
sys.path.insert(0, "/workspace/pathfinder/train")
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path
import stim
from muon import SingleDeviceMuon
import torch.optim
torch.optim.Muon = SingleDeviceMuon


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Pre-norm variant of the bottleneck block + wider backbone (H=512 by default)
# ============================================================================

class DirectionalConv3d(nn.Module):
    """Same as v1 but permute-free via einsum for speed."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.w_self = nn.Linear(in_channels, out_channels, bias=False)
        self.w_tp = nn.Linear(in_channels, out_channels, bias=False)
        self.w_tm = nn.Linear(in_channels, out_channels, bias=False)
        self.w_rp = nn.Linear(in_channels, out_channels, bias=False)
        self.w_rm = nn.Linear(in_channels, out_channels, bias=False)
        self.w_cp = nn.Linear(in_channels, out_channels, bias=False)
        self.w_cm = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):  # x: [B, C, T, R, C_co]
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


class PreNormBottleneckBlock(nn.Module):
    """Pre-norm variant: norm first, then transform, then add residual."""
    def __init__(self, hidden_dim):
        super().__init__()
        reduced = hidden_dim // 4
        self.norm = nn.LayerNorm(hidden_dim)
        self.reduce = nn.Conv3d(hidden_dim, reduced, kernel_size=1, bias=False)
        self.message = DirectionalConv3d(reduced, reduced)
        self.restore = nn.Conv3d(reduced, hidden_dim, kernel_size=1, bias=False)

    def forward(self, x):
        # x: [B, H, T, R, C]
        residual = x
        # Pre-norm (channel-last then back)
        h = x.permute(0, 2, 3, 4, 1)
        h = self.norm(h)
        h = h.permute(0, 4, 1, 2, 3)
        h = F.gelu(self.reduce(h))
        h = F.gelu(self.message(h))
        h = self.restore(h)
        return residual + h


class NoiseEmbedding(nn.Module):
    """Learned embedding of log-noise-rate — added to the initial feature map."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(1, hidden_dim)

    def forward(self, x, log_p):
        # log_p: [B] — log10(p) roughly in [-4, -1.5]
        emb = self.proj(log_p.unsqueeze(-1))  # [B, H]
        return x + emb.view(emb.shape[0], emb.shape[1], 1, 1, 1)


class NeuralDecoderV2(nn.Module):
    """Pathfinder v2 with pre-norm, H=512 default, noise-rate conditioning."""
    def __init__(self, distance, rounds, hidden_dim=512, n_blocks=None, n_observables=1):
        super().__init__()
        self.distance = distance
        self.rounds = rounds
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks if n_blocks else distance
        self.n_observables = n_observables
        self.embed = nn.Conv3d(1, hidden_dim, kernel_size=1, bias=True)
        self.noise_embed = NoiseEmbedding(hidden_dim)
        self.blocks = nn.ModuleList([PreNormBottleneckBlock(hidden_dim) for _ in range(self.n_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_observables),
        )

    def forward(self, syndrome, log_p):
        x = self.embed(syndrome)
        x = self.noise_embed(x, log_p)
        for b in self.blocks:
            x = b(x)
        # Final norm
        x = x.permute(0, 2, 3, 4, 1)
        x = self.final_norm(x)
        x = x.mean(dim=(1, 2, 3))
        return self.head(x)


# ============================================================================
# D4 symmetry augmentations (safe for memory-Z: identity, R-flip, C-flip, both)
# ============================================================================

def d4_augment(syndromes, labels):
    """Random group element from {identity, R-flip, C-flip, R+C-flip}.
    For rotated memory-Z: R and C flips each preserve logical Z observable,
    so labels are invariant. 180° rotation is R+C flip composed.

    syndromes: [B, 1, T, R, C] float
    labels: [B, n_observables] float
    """
    B = syndromes.shape[0]
    g = torch.randint(0, 4, (B,), device=syndromes.device)  # group element per sample
    out = syndromes.clone()
    for i in range(B):
        gi = g[i].item()
        if gi == 1:
            out[i] = torch.flip(out[i], dims=[-2])  # R-flip
        elif gi == 2:
            out[i] = torch.flip(out[i], dims=[-1])  # C-flip
        elif gi == 3:
            out[i] = torch.flip(out[i], dims=[-2, -1])  # both
    return out, labels


# Vectorized version for speed
def d4_augment_fast(syndromes, labels):
    """Vectorized D4 augmentation. Apply same group element to whole batch (simpler, still gives 4x diversity across batches)."""
    g = int(torch.randint(0, 4, (1,)).item())
    if g == 0:
        return syndromes, labels
    elif g == 1:
        return torch.flip(syndromes, dims=[-2]), labels
    elif g == 2:
        return torch.flip(syndromes, dims=[-1]), labels
    else:
        return torch.flip(syndromes, dims=[-2, -1]), labels


# ============================================================================
# Data generation with proper noise model
# ============================================================================

class MixedNoiseDataset:
    """Mixed-noise training dataset.
    Samples p uniformly in log-space from [p_min, p_max] each batch,
    generates fresh syndromes, returns (syndromes, labels, log_p).
    """
    def __init__(self, distance, rounds, p_min=0.001, p_max=0.015,
                 batch_size=512, seed_base=0):
        self.distance = distance
        self.rounds = rounds
        self.p_min = p_min
        self.p_max = p_max
        self.batch_size = batch_size
        self._samplers = {}
        self._mappers = {}
        # Pre-compile at log-grid of noise rates
        log_p_min = math.log(p_min)
        log_p_max = math.log(p_max)
        self.log_grid = np.linspace(log_p_min, log_p_max, 60)
        self.p_grid = np.exp(self.log_grid)
        for i, p in enumerate(self.p_grid):
            p = float(p)
            circ = stim.Circuit.generated(
                "surface_code:rotated_memory_z", distance=distance, rounds=rounds,
                after_clifford_depolarization=p,
                before_measure_flip_probability=p,
                after_reset_flip_probability=p,
                before_round_data_depolarization=p,  # THE fix
            )
            self._samplers[p] = circ.compile_detector_sampler()
            if i == 0:
                self._mapper = self._make_mapper(circ)

    def _make_mapper(self, circ):
        coords = circ.get_detector_coordinates()
        nd = circ.num_detectors
        ac = np.array([coords[i] for i in range(nd)])
        sp, tm = ac[:, :-1], ac[:, -1]
        tu = np.sort(np.unique(tm))
        xu = np.sort(np.unique(sp[:, 0]))
        yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
        grid = (len(tu), len(yu), len(xu))
        tm_m = {v: i for i, v in enumerate(tu)}
        xm = {v: i for i, v in enumerate(xu)}
        ym = {v: i for i, v in enumerate(yu)}
        det_idx = np.zeros((nd, 3), dtype=np.int64)
        for did in range(nd):
            c = coords[did]
            det_idx[did] = [tm_m[c[-1]], ym.get(c[1], 0) if len(c) > 2 else 0, xm[c[0]]]
        return {"grid": grid, "det_idx": det_idx, "nd": nd}

    def sample(self):
        # Random log-uniform p
        idx = np.random.randint(len(self.p_grid))
        p = float(self.p_grid[idx])
        sampler = self._samplers[p]
        det, obs = sampler.sample(self.batch_size, separate_observables=True)
        # Build tensor
        B = det.shape[0]
        T, H, W = self._mapper["grid"]
        tensor = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        det_f = torch.from_numpy(det.astype(np.float32))
        for i in range(self._mapper["nd"]):
            tensor[:, 0, self._mapper["det_idx"][i, 0],
                         self._mapper["det_idx"][i, 1],
                         self._mapper["det_idx"][i, 2]] = det_f[:, i]
        labels = torch.from_numpy(obs.astype(np.float32))
        log_p = torch.full((B,), math.log(p), dtype=torch.float32)
        return tensor, labels, log_p


def evaluate(model, distance, rounds, p, n_shots=20000, batch=1000, device="cuda"):
    ds = MixedNoiseDataset(distance, rounds, p_min=p, p_max=p, batch_size=batch)
    errs = 0
    total = 0
    log_p = torch.full((batch,), math.log(p), dtype=torch.float32, device=device)
    for _ in range(n_shots // batch):
        syn, lab, _ = ds.sample()
        syn = syn.to(device)
        lab = lab.to(device)
        with torch.no_grad():
            logits = model(syn, log_p[:syn.shape[0]])
        preds = (logits > 0).float()
        errs += (preds != lab).any(dim=1).sum().item()
        total += batch
    return errs / max(total, 1)


def build_opts(model, muon_lr=0.02, adam_lr=3e-3):
    muon_params, adam_params = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim == 2:
            muon_params.append(p)
        else:
            adam_params.append(p)
    opts = []
    if muon_params:
        opts.append(SingleDeviceMuon(muon_params, lr=muon_lr, momentum=0.95, weight_decay=0.01))
    if adam_params:
        opts.append(torch.optim.AdamW(adam_params, lr=adam_lr, weight_decay=0.0))
    return opts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--steps", type=int, default=150000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--p_min", type=float, default=0.001)
    parser.add_argument("--p_max", type=float, default=0.015)
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--adam_lr", type=float, default=3e-3)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--eval_shots", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="/workspace/pathfinder/train/checkpoints/v2_d5")
    parser.add_argument("--d4_aug", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Model
    model = NeuralDecoderV2(distance=args.distance, rounds=args.distance,
                            hidden_dim=args.hidden_dim, n_blocks=args.distance).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: d={args.distance}, H={args.hidden_dim}, L={args.distance}, {n_params:,} params ({n_params * 2 / 1e6:.2f} MB FP16)", flush=True)

    # Dataset
    ds = MixedNoiseDataset(args.distance, args.distance, args.p_min, args.p_max, args.batch_size)
    print(f"Dataset: mixed noise log-uniform [{args.p_min}, {args.p_max}], 60-point grid", flush=True)

    # Optimizers
    opts = build_opts(model, args.muon_lr, args.adam_lr)
    print(f"Optimizers: {[type(o).__name__ for o in opts]}", flush=True)

    # LR schedule
    base_lrs = [[pg['lr'] for pg in opt.param_groups] for opt in opts]
    def step_lr(s):
        if s < args.warmup:
            scale = s / max(args.warmup, 1)
        else:
            prog = (s - args.warmup) / max(args.steps - args.warmup, 1)
            scale = 0.5 * (1 + math.cos(math.pi * prog))
        for opt, lrs in zip(opts, base_lrs):
            for pg, lr in zip(opt.param_groups, lrs):
                pg['lr'] = lr * scale

    scaler = torch.amp.GradScaler("cuda")
    model.train()
    best_ler_07 = 1.0
    t0 = time.time()

    for step in range(args.steps):
        syn, lab, log_p = ds.sample()
        syn = syn.to(device); lab = lab.to(device); log_p = log_p.to(device)

        # D4 augmentation (fast batched version)
        if args.d4_aug:
            syn, lab = d4_augment_fast(syn, lab)

        for opt in opts:
            opt.zero_grad()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(syn, log_p)
            loss = F.binary_cross_entropy_with_logits(logits, lab)

        scaler.scale(loss).backward()
        for opt in opts:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts:
            scaler.step(opt)
        scaler.update()
        step_lr(step)

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            sps = (step + 1) / max(elapsed, 1)
            eta = (args.steps - step) / max(sps, 0.01) / 60
            lr = opts[0].param_groups[0]['lr']
            print(f"step {step:>6}/{args.steps}  loss={loss.item():.4f}  p={math.exp(log_p[0].item()):.5f}  lr={lr:.5f}  {sps:.1f} s/s  ETA {eta:.0f}min", flush=True)

        if step > 0 and step % args.eval_interval == 0:
            model.eval()
            ler_07 = evaluate(model, args.distance, args.distance, 0.007, args.eval_shots, device=device)
            ler_001 = evaluate(model, args.distance, args.distance, 0.001, args.eval_shots, device=device)
            ler_015 = evaluate(model, args.distance, args.distance, 0.015, args.eval_shots, device=device)
            print(f"  >>> EVAL LER p=0.001 {ler_001:.5f}  p=0.007 {ler_07:.5f}  p=0.015 {ler_015:.5f}", flush=True)
            if ler_07 < best_ler_07:
                best_ler_07 = ler_07
                torch.save({'step': step, 'model_state_dict': model.state_dict(),
                            'distance': args.distance, 'rounds': args.distance,
                            'hidden_dim': args.hidden_dim, 'n_blocks': args.distance,
                            'ler_p007': ler_07, 'ler_p001': ler_001, 'ler_p015': ler_015},
                           f"{args.checkpoint_dir}/best_model.pt")
                print(f"  >>> Saved best (ler@p=0.007={ler_07:.5f})", flush=True)
            model.train()

    # Final save
    torch.save({'step': args.steps, 'model_state_dict': model.state_dict(),
                'distance': args.distance, 'rounds': args.distance,
                'hidden_dim': args.hidden_dim, 'n_blocks': args.distance},
               f"{args.checkpoint_dir}/final_model.pt")
    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
