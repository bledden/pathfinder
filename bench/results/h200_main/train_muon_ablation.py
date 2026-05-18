"""Muon ablation: train Pathfinder with AdamW on ALL parameters (both 2D and 1D).
Matches the 3-parameter noise model and recipe used in Table 1 / Table 4.
Output fills in the d=3 and d=7 rows of the Muon ablation section (5.4).

Usage:
  python3 train_muon_ablation.py --distance 3 --steps 20000 --ckpt /workspace/pathfinder/train/checkpoints/ablation_adamw_d3
  python3 train_muon_ablation.py --distance 7 --steps 80000 --batch 256 --ckpt /workspace/pathfinder/train/checkpoints/ablation_adamw_d7
"""
import sys, os, time, math, argparse
sys.path.insert(0, "/workspace/pathfinder/train")
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from model import NeuralDecoder, DecoderConfig
from data import SyndromeDataset, DataConfig, CurriculumScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def measure_ler(model, config, noise_rate, n_shots=10000):
    model.eval()
    ds = SyndromeDataset(DataConfig(distance=config.distance, rounds=config.rounds, physical_error_rate=noise_rate))
    errors = 0
    total = 0
    batch = min(1000, n_shots)
    with torch.no_grad():
        for _ in range(n_shots // batch):
            syn, lab = ds.sample(batch)
            syn = syn.to(device); lab = lab.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(syn)
            preds = (lg > 0).float()
            errors += int((preds != lab).any(dim=1).sum().item())
            total += batch
    model.train()
    return errors / total


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--distance", type=int, default=3)
    a.add_argument("--hidden_dim", type=int, default=256)
    a.add_argument("--steps", type=int, default=20000)
    a.add_argument("--batch", type=int, default=512)
    a.add_argument("--lr", type=float, default=3e-3)
    a.add_argument("--weight_decay", type=float, default=0.01)
    a.add_argument("--noise_rate", type=float, default=0.007)
    a.add_argument("--log_interval", type=int, default=100)
    a.add_argument("--measure_interval", type=int, default=2000)
    a.add_argument("--ckpt", type=str, default="/workspace/pathfinder/train/checkpoints/ablation_adamw")
    args = a.parse_args()
    os.makedirs(args.ckpt, exist_ok=True)

    config = DecoderConfig(distance=args.distance, rounds=args.distance, hidden_dim=args.hidden_dim, n_observables=1)
    model = NeuralDecoder(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Ablation: AdamW on ALL params, d={args.distance}, H={args.hidden_dim}, L={args.distance}, {n_params:,} params", flush=True)

    curric = CurriculumScheduler(args.noise_rate, args.steps)
    rates = sorted(set(round(curric.get_rate(s), 4) for s in range(0, args.steps, max(args.steps // 50, 1))) | {round(args.noise_rate, 4)})
    samplers = {p: SyndromeDataset(DataConfig(distance=args.distance, rounds=args.distance, physical_error_rate=max(p, 1e-6))) for p in rates}
    print(f"Pre-compiled {len(rates)} samplers at 3-parameter noise (Table 4 methodology).", flush=True)

    def get_ds(p):
        return samplers[min(samplers.keys(), key=lambda k: abs(k - p))]

    # AdamW on ALL parameters — this is the ablation
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    base_lr = args.lr
    warmup = 1000
    def step_lr(s):
        if s < warmup:
            scale = s / warmup
        else:
            prog = (s - warmup) / max(args.steps - warmup, 1)
            scale = 0.5 * (1 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg['lr'] = base_lr * scale

    scaler = torch.amp.GradScaler("cuda")
    model.train()
    best = 1.0
    t0 = time.time()

    for step in range(args.steps):
        cp = curric.get_rate(step)
        ds = get_ds(cp)
        syn, lab = ds.sample(args.batch)
        syn = syn.to(device); lab = lab.to(device)
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(syn)
            loss = F.binary_cross_entropy_with_logits(lg, lab)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step_lr(step)

        if step % args.log_interval == 0:
            sps = (step + 1) / max(time.time() - t0, 1)
            eta = (args.steps - step) / max(sps, 0.01) / 60
            print(f"step {step:>6}/{args.steps}  loss={loss.item():.4f}  p={cp:.5f}  {sps:.1f} s/s  ETA {eta:.0f}min", flush=True)

        if step > 0 and step % args.measure_interval == 0:
            ler = measure_ler(model, config, args.noise_rate)
            print(f"  >>> LER @ p={args.noise_rate}: {ler:.5f}", flush=True)
            if ler < best:
                best = ler
                torch.save({'step': step, 'model_state_dict': model.state_dict(), 'config': config, 'ler': ler},
                           f"{args.ckpt}/best_model.pt")
                print(f"  >>> saved (ler={ler:.5f})", flush=True)

    final_ler = measure_ler(model, config, args.noise_rate, n_shots=50000)
    print(f"\nFinal LER @ p={args.noise_rate} (50K shots): {final_ler:.5f}", flush=True)
    torch.save({'step': args.steps, 'model_state_dict': model.state_dict(), 'config': config, 'final_ler': final_ler},
               f"{args.ckpt}/final_model.pt")


if __name__ == "__main__":
    main()
