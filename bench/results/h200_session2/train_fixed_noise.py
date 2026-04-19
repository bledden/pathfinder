"""
RETRAIN v1 architecture with CORRECT noise model (includes before_round_data_depolarization=p).
No other changes to architecture or training recipe — isolates whether the noise-model fix alone
is enough to match/beat Lange under matched conditions.
"""
import sys, os, time, math, argparse
sys.path.insert(0, "/workspace/pathfinder/train")
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import stim
from muon import SingleDeviceMuon
import torch.optim
torch.optim.Muon = SingleDeviceMuon

from model import NeuralDecoder, DecoderConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SyndromeDatasetCorrected:
    """Same as SyndromeDataset but WITH before_round_data_depolarization=p."""
    def __init__(self, distance, rounds, p, batch_size=512):
        self.circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=distance, rounds=rounds,
            after_clifford_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p,
            before_round_data_depolarization=p,  # THE fix
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
        tm_m = {v: i for i, v in enumerate(tu)}; xm = {v: i for i, v in enumerate(xu)}; ym = {v: i for i, v in enumerate(yu)}
        self.det_idx = np.zeros((nd, 3), dtype=np.int64)
        for did in range(nd):
            c = coords[did]
            self.det_idx[did] = [tm_m[c[-1]], ym.get(c[1], 0) if len(c) > 2 else 0, xm[c[0]]]
        self.nd = nd
        self.batch_size = batch_size

    def sample(self):
        det, obs = self.sampler.sample(self.batch_size, separate_observables=True)
        B = det.shape[0]; T, H, W = self.grid
        t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        d = torch.from_numpy(det.astype(np.float32))
        for i in range(self.nd):
            t[:, 0, self.det_idx[i, 0], self.det_idx[i, 1], self.det_idx[i, 2]] = d[:, i]
        return t, torch.from_numpy(obs.astype(np.float32))


class CurriculumScheduler:
    """Same as v1."""
    def __init__(self, target, total_steps):
        self.target = target; self.total = total_steps
    def get_rate(self, step):
        f = step / max(self.total, 1)
        if f < 0.2: return self.target * 0.1
        elif f < 0.6:
            t = (f - 0.2) / 0.4
            return self.target * (0.1 + 0.4 * t)
        else:
            t = (f - 0.6) / 0.4
            return self.target * (0.5 + 0.5 * t)


def evaluate(model, d, r, p, n=10000, batch=1000, device="cuda"):
    ds = SyndromeDatasetCorrected(d, r, p, batch)
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
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--distance", type=int, default=5)
    a.add_argument("--hidden_dim", type=int, default=256)
    a.add_argument("--steps", type=int, default=80000)
    a.add_argument("--batch", type=int, default=512)
    a.add_argument("--noise_rate", type=float, default=0.007)
    a.add_argument("--muon_lr", type=float, default=0.02)
    a.add_argument("--adam_lr", type=float, default=3e-3)
    a.add_argument("--eval_interval", type=int, default=5000)
    a.add_argument("--log_interval", type=int, default=500)
    a.add_argument("--ckpt", type=str, default="/workspace/pathfinder/train/checkpoints/fixed_d5")
    args = a.parse_args()
    os.makedirs(args.ckpt, exist_ok=True)

    config = DecoderConfig(distance=args.distance, rounds=args.distance, hidden_dim=args.hidden_dim, n_observables=1)
    model = NeuralDecoder(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: d={args.distance}, H={args.hidden_dim}, L={args.distance}, {n_params:,} params", flush=True)

    # Curriculum noise annealing (like v1)
    curric = CurriculumScheduler(args.noise_rate, args.steps)
    # Pre-compile samplers at curriculum rates
    rates = sorted(set(round(curric.get_rate(s), 4) for s in range(0, args.steps, max(args.steps // 50, 1))) | {round(args.noise_rate, 4)})
    samplers = {}
    print(f"Pre-compiling {len(rates)} samplers at curriculum rates (with proper noise model)...", flush=True)
    for p in rates:
        samplers[p] = SyndromeDatasetCorrected(args.distance, args.distance, max(p, 1e-6), args.batch)

    def get_ds(p):
        return samplers[min(samplers.keys(), key=lambda k: abs(k - p))]

    muon_params = [p for p in model.parameters() if p.ndim == 2 and p.requires_grad]
    adam_params = [p for p in model.parameters() if p.ndim != 2 and p.requires_grad]
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
        cp = curric.get_rate(step)
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
            sps = (step+1) / max(time.time()-t0, 1)
            eta = (args.steps - step) / max(sps, 0.01) / 60
            print(f"step {step:>6}/{args.steps}  loss={loss.item():.4f}  p={cp:.5f}  {sps:.1f} s/s  ETA {eta:.0f}min", flush=True)

        if step > 0 and step % args.eval_interval == 0:
            model.eval()
            ler = evaluate(model, args.distance, args.distance, args.noise_rate)
            print(f"  >>> EVAL LER @ p={args.noise_rate}: {ler:.5f}", flush=True)
            if ler < best:
                best = ler
                torch.save({'step': step, 'model_state_dict': model.state_dict(),
                            'config': config, 'ler': ler},
                           f"{args.ckpt}/best_model.pt")
                print(f"  >>> saved (ler={ler:.5f})", flush=True)
            model.train()

    torch.save({'step': args.steps, 'model_state_dict': model.state_dict(), 'config': config},
               f"{args.ckpt}/final_model.pt")


if __name__ == "__main__":
    main()
