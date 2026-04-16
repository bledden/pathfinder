"""Training script for neural CNN decoder — Gu et al. (arXiv:2604.08358).

Usage:
    python train/train.py --distance 3 --steps 20000
    python train/train.py --distance 5 --hidden_dim 256 --steps 80000
"""
import argparse
import os
import sys
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.optim import Muon, AdamW

sys.path.insert(0, os.path.dirname(__file__))
from model import NeuralDecoder, DecoderConfig
from data import SyndromeDataset, DataConfig, CurriculumScheduler


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS has slow fallback for 5D padding in DirectionalConv3d — use CPU instead
    # TODO: enable MPS when PyTorch adds native 5D padding support
    return torch.device("cpu")


def build_optimizers(model, muon_lr=0.02, adam_lr=1e-3, weight_decay=0.01):
    """Muon for 2D weight matrices, AdamW for bias/1D params.

    Gu et al.: "Muon (Newton-Schulz orthogonalization) + Lion"
    PyTorch's Muon is for 2D params only. 1D params (bias, LayerNorm)
    go to AdamW.
    """
    muon_params = []
    adam_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Muon ONLY supports exactly 2D params (nn.Linear weights)
        # Conv3d weights are 5D, bias/LayerNorm are 1D — both go to AdamW
        if param.ndim == 2:
            muon_params.append(param)
        else:
            adam_params.append(param)

    optimizers = []

    if muon_params:
        optimizers.append(Muon(muon_params, lr=muon_lr, momentum=0.95,
                               weight_decay=weight_decay))

    if adam_params:
        optimizers.append(AdamW(adam_params, lr=adam_lr, weight_decay=0.0))

    return optimizers


class WarmupCosineScheduler:
    """Linear warmup → cosine decay. From Gu et al.: 1000-step warmup."""
    def __init__(self, optimizers, warmup_steps, total_steps):
        self.optimizers = optimizers
        self.warmup = warmup_steps
        self.total = total_steps
        self.base_lrs = [[pg['lr'] for pg in opt.param_groups] for opt in optimizers]

    def step(self, current_step):
        if current_step < self.warmup:
            scale = current_step / max(self.warmup, 1)
        else:
            progress = (current_step - self.warmup) / max(self.total - self.warmup, 1)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for opt, base_lr_list in zip(self.optimizers, self.base_lrs):
            for pg, base_lr in zip(opt.param_groups, base_lr_list):
                pg['lr'] = base_lr * scale


def evaluate(model, config, device, noise_rate, n_shots=10000):
    """Evaluate model LER on fresh syndromes at given noise rate."""
    model.eval()
    data_config = DataConfig(
        distance=config.distance,
        rounds=config.rounds,
        physical_error_rate=noise_rate,
    )
    dataset = SyndromeDataset(data_config)

    total_errors = 0
    total = 0
    batch_size = min(1000, n_shots)

    for _ in range(n_shots // batch_size):
        syndromes, labels = dataset.sample(batch_size)
        syndromes = syndromes.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(syndromes)
            preds = (logits > 0).float()
            total_errors += (preds != labels).any(dim=1).sum().item()
            total += batch_size

    model.train()
    return total_errors / max(total, 1)


def train(args):
    device = get_device()
    print(f"Device: {device}")

    # Model config — full paper spec
    config = DecoderConfig(
        distance=args.distance,
        rounds=args.distance,
        hidden_dim=args.hidden_dim,
        n_observables=1,
    )
    model = NeuralDecoder(config).to(device)
    n_params = NeuralDecoder.count_parameters(model)
    print(f"Model: d={config.distance}, H={config.hidden_dim}, L={config.n_blocks}, "
          f"{n_params:,} params ({n_params * 2 / 1e6:.1f} MB at FP16)")

    # Optimizers: Muon for 2D weights, AdamW for 1D params
    optimizers = build_optimizers(model, muon_lr=args.muon_lr, adam_lr=args.adam_lr)
    print(f"Optimizers: {len(optimizers)} ({', '.join(type(o).__name__ for o in optimizers)})")

    # LR scheduler: linear warmup → cosine decay
    scheduler = WarmupCosineScheduler(optimizers, warmup_steps=1000, total_steps=args.steps)

    # Curriculum noise annealing
    curriculum = CurriculumScheduler(args.noise_rate, args.steps)

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Mixed precision
    use_amp = device.type in ('cuda',)
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    autocast_dtype = torch.bfloat16 if use_amp else None

    # Pre-compile Stim samplers for all curriculum noise rates
    # This avoids expensive circuit compilation in the training loop
    print("Pre-compiling Stim samplers for curriculum...", end=" ", flush=True)
    curriculum_rates = sorted(set(
        round(curriculum.get_rate(s), 4) for s in range(0, args.steps, max(args.steps // 50, 1))
    ))
    # Ensure target rate is included
    curriculum_rates = sorted(set(curriculum_rates) | {round(args.noise_rate, 4)})
    dataset_cache = {}
    for p in curriculum_rates:
        dataset_cache[p] = SyndromeDataset(DataConfig(
            distance=args.distance,
            rounds=args.distance,
            physical_error_rate=max(p, 1e-6),
            batch_size=args.batch_size,
        ))
    print(f"{len(dataset_cache)} samplers ready.", flush=True)

    def get_dataset(noise_rate):
        """Find the closest pre-compiled sampler."""
        key = min(dataset_cache.keys(), key=lambda k: abs(k - noise_rate))
        return dataset_cache[key]

    # Training loop
    model.train()
    best_ler = 1.0
    start_time = time.time()

    for step in range(args.steps):
        # Current noise rate from curriculum
        current_p = curriculum.get_rate(step)
        dataset = get_dataset(current_p)
        syndromes, labels = dataset.sample()
        syndromes = syndromes.to(device)
        labels = labels.to(device)

        # Forward + backward
        for opt in optimizers:
            opt.zero_grad()

        if use_amp and autocast_dtype:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                logits = model(syndromes)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            scaler.scale(loss).backward()
            for opt in optimizers:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in optimizers:
                scaler.step(opt)
            scaler.update()
        else:
            logits = model(syndromes)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in optimizers:
                opt.step()

        scheduler.step(step)

        # Log
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            lr = optimizers[0].param_groups[0]['lr']
            steps_per_sec = (step + 1) / max(elapsed, 1)
            eta = (args.steps - step) / max(steps_per_sec, 0.01)
            print(f"step {step:>6}/{args.steps}  loss={loss.item():.4f}  "
                  f"p={current_p:.5f}  lr={lr:.6f}  "
                  f"{steps_per_sec:.1f} steps/s  ETA {eta/60:.0f}min")

        # Evaluate
        if step > 0 and step % args.eval_interval == 0:
            ler = evaluate(model, config, device, args.noise_rate, n_shots=args.eval_shots)
            print(f"  >>> EVAL LER at p={args.noise_rate}: {ler:.6f}")

            if ler < best_ler:
                best_ler = ler
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'ler': ler,
                    'grid_shape': dataset.grid_shape,
                }, ckpt_dir / "best_model.pt")
                print(f"  >>> Saved best model (LER={ler:.6f})")

    # Final evaluation
    final_ler = evaluate(model, config, device, args.noise_rate, n_shots=50000)
    print(f"\nFinal LER at p={args.noise_rate}: {final_ler:.6f}")
    print(f"Best LER: {best_ler:.6f}")
    print(f"Total time: {time.time() - start_time:.0f}s")

    torch.save({
        'step': args.steps,
        'model_state_dict': model.state_dict(),
        'config': config,
        'ler': final_ler,
        'grid_shape': dataset.grid_shape,
    }, ckpt_dir / "final_model.pt")


def main():
    parser = argparse.ArgumentParser(description="Train neural QEC decoder (Gu et al. 2026)")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--adam_lr", type=float, default=1e-3)
    parser.add_argument("--noise_rate", type=float, default=0.007)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--eval_shots", type=int, default=10000)
    parser.add_argument("--checkpoint_dir", type=str, default="train/checkpoints")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
