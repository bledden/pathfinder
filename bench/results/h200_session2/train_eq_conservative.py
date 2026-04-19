"""
Conservative equivariant training: D₂-equivariant architecture at H=256,
mixed p ∈ [0.001, 0.007] (overlap with Lange's range), no noise embedding,
standard 80K steps. Test if architectural equivariance alone improves over v1.
"""
import sys, os, time, math, argparse
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace")
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import stim
from muon import SingleDeviceMuon
import torch.optim
torch.optim.Muon = SingleDeviceMuon


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EquivariantDirectionalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.w_self = nn.Linear(in_channels, out_channels, bias=False)
        self.w_tp = nn.Linear(in_channels, out_channels, bias=False)
        self.w_tm = nn.Linear(in_channels, out_channels, bias=False)
        self.w_r = nn.Linear(in_channels, out_channels, bias=False)
        self.w_c = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        xp = x.permute(0, 2, 3, 4, 1)
        out = self.w_self(xp)
        if xp.shape[1] > 1:
            out = out + F.pad(self.w_tp(xp[:, :-1]), (0,0,0,0,0,0,1,0))
            out = out + F.pad(self.w_tm(xp[:, 1:]),  (0,0,0,0,0,0,0,1))
        if xp.shape[2] > 1:
            out = out + F.pad(self.w_r(xp[:, :, :-1]), (0,0,0,0,1,0))
            out = out + F.pad(self.w_r(xp[:, :, 1:]),  (0,0,0,0,0,1))
        if xp.shape[3] > 1:
            out = out + F.pad(self.w_c(xp[:, :, :, :-1]), (0,0,1,0))
            out = out + F.pad(self.w_c(xp[:, :, :, 1:]),  (0,0,0,1))
        return out.permute(0, 4, 1, 2, 3)


class PreNormBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        reduced = hidden_dim // 4
        self.norm = nn.LayerNorm(hidden_dim)
        self.reduce = nn.Conv3d(hidden_dim, reduced, kernel_size=1, bias=False)
        self.message = EquivariantDirectionalConv3d(reduced, reduced)
        self.restore = nn.Conv3d(reduced, hidden_dim, kernel_size=1, bias=False)

    def forward(self, x):
        res = x
        h = x.permute(0, 2, 3, 4, 1)
        h = self.norm(h)
        h = h.permute(0, 4, 1, 2, 3)
        h = F.gelu(self.reduce(h))
        h = F.gelu(self.message(h))
        h = self.restore(h)
        return res + h


class EqDecoder(nn.Module):
    def __init__(self, d, r, H=256, L=None):
        super().__init__()
        self.d = d; self.r = r; self.H = H; self.L = L or d
        self.embed = nn.Conv3d(1, H, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList([PreNormBlock(H) for _ in range(self.L)])
        self.final_norm = nn.LayerNorm(H)
        self.head = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, 1))

    def forward(self, syn):
        x = self.embed(syn)
        for b in self.blocks:
            x = b(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.final_norm(x)
        x = x.mean(dim=(1, 2, 3))
        return self.head(x)


class MixedDS:
    def __init__(self, d, r, p_min=0.001, p_max=0.007, batch=512):
        self.d = d; self.r = r; self.batch = batch
        self.log_min = math.log(p_min); self.log_max = math.log(p_max)
        self.grid = np.exp(np.linspace(self.log_min, self.log_max, 50))
        self.samplers = {}
        first_circ = None
        for p in self.grid:
            p = float(p)
            c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=r,
                after_clifford_depolarization=p, before_measure_flip_probability=p,
                after_reset_flip_probability=p, before_round_data_depolarization=p)
            self.samplers[p] = c.compile_detector_sampler()
            if first_circ is None: first_circ = c
        self.mapper = self._mk(first_circ)

    def _mk(self, circ):
        nd = circ.num_detectors
        coords = circ.get_detector_coordinates()
        ac = np.array([coords[i] for i in range(nd)])
        sp, tm = ac[:, :-1], ac[:, -1]
        tu = np.sort(np.unique(tm))
        xu = np.sort(np.unique(sp[:, 0]))
        yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
        grid = (len(tu), len(yu), len(xu))
        tm_m = {v: i for i, v in enumerate(tu)}; xm = {v: i for i, v in enumerate(xu)}; ym = {v: i for i, v in enumerate(yu)}
        di = np.zeros((nd, 3), dtype=np.int64)
        for did in range(nd):
            c = coords[did]; di[did] = [tm_m[c[-1]], ym.get(c[1], 0) if len(c) > 2 else 0, xm[c[0]]]
        return {"grid": grid, "di": di, "nd": nd}

    def sample(self):
        p = float(self.grid[np.random.randint(len(self.grid))])
        det, obs = self.samplers[p].sample(self.batch, separate_observables=True)
        B = self.batch; T, H, W = self.mapper["grid"]
        t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        d = torch.from_numpy(det.astype(np.float32))
        for i in range(self.mapper["nd"]):
            t[:, 0, self.mapper["di"][i, 0], self.mapper["di"][i, 1], self.mapper["di"][i, 2]] = d[:, i]
        return t, torch.from_numpy(obs.astype(np.float32))


def evaluate(m, d, r, p, n=10000, batch=1000):
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=r,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)
    ds = MixedDS(d, r, p, p, batch)
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
    import argparse
    p_args = argparse.ArgumentParser()
    p_args.add_argument("--distance", type=int, default=5)
    p_args.add_argument("--hidden_dim", type=int, default=256)
    p_args.add_argument("--steps", type=int, default=80000)
    p_args.add_argument("--batch", type=int, default=512)
    p_args.add_argument("--muon_lr", type=float, default=0.02)
    p_args.add_argument("--adam_lr", type=float, default=3e-3)
    p_args.add_argument("--p_min", type=float, default=0.001)
    p_args.add_argument("--p_max", type=float, default=0.007)
    p_args.add_argument("--eval_interval", type=int, default=5000)
    p_args.add_argument("--log_interval", type=int, default=500)
    p_args.add_argument("--ckpt", type=str, default="/workspace/pathfinder/train/checkpoints/eq_d5")
    args = p_args.parse_args()

    os.makedirs(args.ckpt, exist_ok=True)

    m = EqDecoder(args.distance, args.distance, H=args.hidden_dim).to(device)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Equivariant Decoder: d={args.distance}, H={args.hidden_dim}, L={args.distance}, {n_params:,} params", flush=True)

    ds = MixedDS(args.distance, args.distance, args.p_min, args.p_max, args.batch)
    muon_params = [p for p in m.parameters() if p.ndim == 2 and p.requires_grad]
    adam_params = [p for p in m.parameters() if p.ndim != 2 and p.requires_grad]
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
    m.train()
    best = 1.0
    t0 = time.time()

    for step in range(args.steps):
        syn, lab = ds.sample()
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
            print(f"step {step:>6}/{args.steps}  loss={loss.item():.4f}  {sps:.1f} s/s  ETA {eta:.0f}min", flush=True)
        if step > 0 and step % args.eval_interval == 0:
            m.eval()
            ler = evaluate(m, args.distance, args.distance, 0.005)
            ler_03 = evaluate(m, args.distance, args.distance, 0.003)
            print(f"  >>> EVAL LER p=0.003 {ler_03:.5f}  p=0.005 {ler:.5f}", flush=True)
            if ler < best:
                best = ler
                torch.save({'step': step, 'model_state_dict': m.state_dict(),
                            'distance': args.distance, 'rounds': args.distance,
                            'hidden_dim': args.hidden_dim, 'n_blocks': args.distance,
                            'equivariant': True, 'ler_p005': ler, 'ler_p003': ler_03},
                           f"{args.ckpt}/best_model.pt")
                print(f"  >>> saved (ler@0.005={ler:.5f})", flush=True)
            m.train()
    torch.save({'step': args.steps, 'model_state_dict': m.state_dict(),
                'distance': args.distance, 'rounds': args.distance,
                'hidden_dim': args.hidden_dim, 'n_blocks': args.distance,
                'equivariant': True},
               f"{args.ckpt}/final_model.pt")
    print(f"Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
