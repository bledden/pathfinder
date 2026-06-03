"""Train PFWL3S with IBM-calibrated multi-noise distribution.

Same architecture as train.py (NeuralDecoder + Muon + AdamW + WarmupCosine)
but uses CalibratedSyndromeDataset that samples p in [p_low, p_high] per batch
to produce a model robust to the noise-rate range that real IBM hardware
operates in.

Usage:
    python train/train_calibrated.py --distance 5 --rounds 5 --hidden_dim 384 \
        --steps 150000 --p_low 0.003 --p_high 0.025 --readout_scale 1.5 \
        --checkpoint_dir bench/results/h200_main/calibrated/d5r5_v1
"""
import argparse, os, sys, time, math, json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

try:
    from torch.optim import Muon, AdamW
except ImportError:
    from muon import SingleDeviceMuon
    import torch.optim
    torch.optim.Muon = SingleDeviceMuon
    from torch.optim import Muon, AdamW

sys.path.insert(0, os.path.dirname(__file__))
from model import NeuralDecoder, DecoderConfig
from data import SyndromeDataset, DataConfig
from data_calibrated import CalibratedSyndromeDataset, CalibratedDataConfig, CurriculumMultiNoise


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizers(model, muon_lr=0.02, adam_lr=1e-3, weight_decay=0.01):
    muon_params = []
    adam_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2:
            muon_params.append(param)
        else:
            adam_params.append(param)
    opts = []
    if muon_params:
        opts.append(Muon(muon_params, lr=muon_lr, momentum=0.95, weight_decay=weight_decay))
    if adam_params:
        opts.append(AdamW(adam_params, lr=adam_lr, weight_decay=0.0))
    return opts


class WarmupCosineScheduler:
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


def measure_per_rate(model, ds, device, n_shots=10000):
    """Compute per-rate LER over the calibrated noise sweep."""
    model.train(False)
    bs = min(1000, n_shots)
    per_rate = {}
    for p, sampler in ds.samplers.items():
        errs = 0
        total = 0
        for _ in range(n_shots // bs):
            det, obs = sampler.sample(shots=bs, separate_observables=True)
            syndromes = ds.detectors_to_tensor(det).to(device)
            labels = torch.from_numpy(obs.astype(np.float32)).to(device)
            with torch.no_grad():
                logits = model(syndromes)
                preds = (logits > 0).float()
                errs += (preds != labels).any(dim=1).sum().item()
                total += bs
        per_rate[p] = errs / max(total, 1)
    model.train(True)
    return per_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=None,
                        help="Default: rounds=distance")
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--steps", type=int, default=150000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--adam_lr", type=float, default=1e-3)
    parser.add_argument("--p_low", type=float, default=0.003)
    parser.add_argument("--p_high", type=float, default=0.025)
    parser.add_argument("--n_rates", type=int, default=15)
    parser.add_argument("--readout_scale", type=float, default=1.5)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--eval_shots", type=int, default=5000)
    parser.add_argument("--curriculum", action="store_true",
                        help="Use curriculum noise-band expansion")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rounds = args.rounds if args.rounds is not None else args.distance
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")
    print(f"Config: d={args.distance} r={rounds} H={args.hidden_dim} "
          f"p in [{args.p_low}, {args.p_high}] readout x{args.readout_scale} "
          f"steps={args.steps} bs={args.batch_size} seed={args.seed}")

    config = DecoderConfig(
        distance=args.distance, rounds=rounds,
        hidden_dim=args.hidden_dim, n_observables=1,
    )
    model = NeuralDecoder(config).to(device)
    n_params = NeuralDecoder.count_parameters(model)
    print(f"Model: {n_params:,} params ({n_params * 2 / 1e6:.1f} MB at FP16)")

    optimizers = build_optimizers(model, muon_lr=args.muon_lr, adam_lr=args.adam_lr)
    scheduler = WarmupCosineScheduler(optimizers, warmup_steps=1000, total_steps=args.steps)

    base_data_cfg = CalibratedDataConfig(
        distance=args.distance, rounds=rounds,
        p_low=args.p_low, p_high=args.p_high,
        n_rates=args.n_rates, readout_scale=args.readout_scale,
        batch_size=args.batch_size,
    )

    if args.curriculum:
        curriculum = CurriculumMultiNoise(base_data_cfg, args.steps)
        ds_cache = {}
        def get_ds(step):
            band_cfg = curriculum.get_config(step)
            key = round(band_cfg.p_high, 4)
            if key not in ds_cache:
                ds_cache[key] = CalibratedSyndromeDataset(
                    band_cfg, rng=np.random.default_rng(args.seed)
                )
            return ds_cache[key]
        ds_for_eval = CalibratedSyndromeDataset(base_data_cfg, rng=np.random.default_rng(args.seed))
    else:
        ds = CalibratedSyndromeDataset(base_data_cfg, rng=np.random.default_rng(args.seed))
        def get_ds(step):
            return ds
        ds_for_eval = ds

    print(f"Pre-compiled {args.n_rates} samplers in [{args.p_low}, {args.p_high}]")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    autocast_dtype = torch.bfloat16 if use_amp else None

    model.train(True)
    best_avg_ler = 1.0
    start_time = time.time()
    train_log = []

    for step in range(args.steps):
        ds_cur = get_ds(step)
        syndromes, labels, p_used = ds_cur.sample()
        syndromes = syndromes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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

        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            lr = optimizers[0].param_groups[0]['lr']
            sps = (step + 1) / max(elapsed, 1)
            eta = (args.steps - step) / max(sps, 0.01)
            print(f"step {step:>6}/{args.steps}  loss={loss.item():.4f}  "
                  f"p={p_used:.4f}  lr={lr:.6f}  "
                  f"{sps:.1f} sps  ETA {eta/60:.0f}min", flush=True)

        if step > 0 and step % args.eval_interval == 0:
            per_rate = measure_per_rate(model, ds_for_eval, device, n_shots=args.eval_shots)
            avg_ler = float(np.mean(list(per_rate.values())))
            high_p = sorted(per_rate.keys())[-1]
            high_ler = per_rate[high_p]
            print(f"  >>> avg LER={avg_ler:.4f}  "
                  f"high-p (p={high_p:.4f}) LER={high_ler:.4f}", flush=True)
            train_log.append({
                "step": step, "avg_ler": avg_ler, "per_rate": per_rate,
                "elapsed_s": time.time() - start_time,
            })
            if avg_ler < best_avg_ler:
                best_avg_ler = avg_ler
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'ler': avg_ler,
                    'per_rate': per_rate,
                    'grid_shape': ds_for_eval.grid_shape,
                    'train_log': train_log,
                    'args': vars(args),
                }, ckpt_dir / "best_model.pt")
                print(f"  >>> Saved best (avg_LER={avg_ler:.4f})", flush=True)

    per_rate_final = measure_per_rate(model, ds_for_eval, device, n_shots=20000)
    avg_final = float(np.mean(list(per_rate_final.values())))
    print(f"\nFinal avg LER: {avg_final:.4f}")
    print(f"Per-rate: {per_rate_final}")
    print(f"Best: {best_avg_ler:.4f}, total time: {(time.time()-start_time)/60:.1f}min")

    torch.save({
        'step': args.steps,
        'model_state_dict': model.state_dict(),
        'config': config,
        'ler': avg_final,
        'per_rate': per_rate_final,
        'grid_shape': ds_for_eval.grid_shape,
        'train_log': train_log,
        'args': vars(args),
    }, ckpt_dir / "final_model.pt")

    with open(ckpt_dir / "train_log.json", "w") as f:
        json.dump({"args": vars(args), "train_log": train_log,
                   "final_per_rate": per_rate_final,
                   "best_avg_ler": best_avg_ler}, f, indent=2)


if __name__ == "__main__":
    main()
