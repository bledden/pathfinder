"""Full trainer for Pathfinder-Equivariant: D₂-equivariant + mixed noise + pre-norm + H=512.
Uses train_v2's data pipeline but swaps in the equivariant architecture.
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

# Re-use equivariant architecture
from train_equivariant import EquivariantDecoder
# Re-use data pipeline
from train_v2 import MixedNoiseDataset, d4_augment_fast, build_opts, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    parser.add_argument("--checkpoint_dir", type=str, default="/workspace/pathfinder/train/checkpoints/eq_d5")
    # Equivariant networks are already D2-invariant, so augmentation is redundant.
    # We still apply it as noise-robustness, which may help with the float-rounding of training.
    parser.add_argument("--d4_aug", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model = EquivariantDecoder(distance=args.distance, rounds=args.distance,
                               hidden_dim=args.hidden_dim, n_blocks=args.distance).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Equivariant Model: d={args.distance}, H={args.hidden_dim}, L={args.distance}, {n_params:,} params ({n_params*2/1e6:.2f} MB FP16)", flush=True)
    print(f"Architecture enforces D₂ equivariance (row-flip, col-flip invariance) by weight sharing.", flush=True)

    ds = MixedNoiseDataset(args.distance, args.distance, args.p_min, args.p_max, args.batch_size)
    opts = build_opts(model, args.muon_lr, args.adam_lr)

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
    best = 1.0
    t0 = time.time()

    for step in range(args.steps):
        syn, lab, log_p = ds.sample()
        syn = syn.to(device); lab = lab.to(device); log_p = log_p.to(device)
        if args.d4_aug:
            syn, lab = d4_augment_fast(syn, lab)

        for opt in opts: opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(syn, log_p)
            loss = F.binary_cross_entropy_with_logits(logits, lab)
        scaler.scale(loss).backward()
        for opt in opts: scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts: scaler.step(opt)
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
            ler_7 = evaluate(model, args.distance, args.distance, 0.007, args.eval_shots, device=device)
            ler_1 = evaluate(model, args.distance, args.distance, 0.001, args.eval_shots, device=device)
            ler_15 = evaluate(model, args.distance, args.distance, 0.015, args.eval_shots, device=device)
            print(f"  >>> EVAL LER p=0.001 {ler_1:.5f}  p=0.007 {ler_7:.5f}  p=0.015 {ler_15:.5f}", flush=True)
            if ler_7 < best:
                best = ler_7
                torch.save({'step': step, 'model_state_dict': model.state_dict(),
                            'distance': args.distance, 'rounds': args.distance,
                            'hidden_dim': args.hidden_dim, 'n_blocks': args.distance,
                            'equivariant': True,
                            'ler_p007': ler_7, 'ler_p001': ler_1, 'ler_p015': ler_15},
                           f"{args.checkpoint_dir}/best_model.pt")
                print(f"  >>> Saved best (ler@p=0.007={ler_7:.5f})", flush=True)
            model.train()

    torch.save({'step': args.steps, 'model_state_dict': model.state_dict(),
                'distance': args.distance, 'rounds': args.distance,
                'hidden_dim': args.hidden_dim, 'n_blocks': args.distance,
                'equivariant': True},
               f"{args.checkpoint_dir}/final_model.pt")
    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
