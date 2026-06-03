"""Train PFWL3S on IBM-budget asymmetric noise (real-hardware-faithful).

Uses IBMBudgetSyndromeDataset (Stim asymmetric per-component noise matching
ibm_fez's measurement-dominated error budget) instead of uniform depolarizing.
At alpha=1.0 the simulated det_flip=0.354 matches real IBM d=5 r=5 (0.353).

Can train from scratch or fine-tune from an existing checkpoint (--init-ckpt).

Usage:
    python train/train_ibm_budget.py --distance 5 --rounds 5 --hidden_dim 384 \
        --steps 120000 --batch_size 512 --checkpoint_dir <dir> --seed 0 \
        [--init-ckpt <pfwl3s.pt>]
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
from data_ibm_budget import IBMBudgetSyndromeDataset, IBMBudgetDataConfig, IBM_BUDGET


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_optimizers(model, muon_lr=0.02, adam_lr=1e-3, weight_decay=0.01):
    muon_params, adam_params = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (muon_params if p.ndim == 2 else adam_params).append(p)
    opts = []
    if muon_params:
        opts.append(Muon(muon_params, lr=muon_lr, momentum=0.95, weight_decay=weight_decay))
    if adam_params:
        opts.append(AdamW(adam_params, lr=adam_lr, weight_decay=0.0))
    return opts


class WarmupCosineScheduler:
    def __init__(self, optimizers, warmup_steps, total_steps):
        self.opts = optimizers
        self.warmup = warmup_steps
        self.total = total_steps
        self.base_lrs = [[pg['lr'] for pg in opt.param_groups] for opt in optimizers]

    def step(self, s):
        if s < self.warmup:
            scale = s / max(self.warmup, 1)
        else:
            prog = (s - self.warmup) / max(self.total - self.warmup, 1)
            scale = 0.5 * (1.0 + math.cos(math.pi * prog))
        for opt, base in zip(self.opts, self.base_lrs):
            for pg, b in zip(opt.param_groups, base):
                pg['lr'] = b * scale


def measure_per_alpha(model, ds, device, n_shots=5000):
    """LER at each alpha in the IBM-budget sweep; alpha=1.0 is the IBM op-point."""
    model.train(False)
    bs = min(1000, n_shots)
    per_alpha = {}
    for a, sampler in ds.samplers.items():
        errs = tot = 0
        for _ in range(n_shots // bs):
            det, obs = sampler.sample(shots=bs, separate_observables=True)
            syn = ds.detectors_to_tensor(det).to(device)
            lab = torch.from_numpy(obs.astype(np.float32)).to(device)
            with torch.no_grad():
                preds = (model(syn) > 0).float()
                errs += (preds != lab).any(dim=1).sum().item()
                tot += bs
        per_alpha[a] = errs / max(tot, 1)
    model.train(True)
    return per_alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--hidden_dim", type=int, default=384)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--adam_lr", type=float, default=1e-3)
    ap.add_argument("--alpha_low", type=float, default=0.7)
    ap.add_argument("--alpha_high", type=float, default=1.3)
    ap.add_argument("--n_alphas", type=int, default=13)
    ap.add_argument("--log_interval", type=int, default=500)
    ap.add_argument("--eval_interval", type=int, default=5000)
    ap.add_argument("--eval_shots", type=int, default=5000)
    ap.add_argument("--init-ckpt", default=None, help="optional warm-start checkpoint")
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rounds = args.rounds if args.rounds is not None else args.distance
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Config: d={args.distance} r={rounds} H={args.hidden_dim} "
          f"alpha in [{args.alpha_low}, {args.alpha_high}] steps={args.steps} "
          f"bs={args.batch_size} seed={args.seed}")

    cfg = DecoderConfig(distance=args.distance, rounds=rounds,
                        hidden_dim=args.hidden_dim, n_observables=1)
    model = NeuralDecoder(cfg).to(device)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, weights_only=False, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        print(f"Warm-started from {args.init_ckpt} (its LER={ck.get('ler','N/A')})")
    n_params = NeuralDecoder.count_parameters(model)
    print(f"Model: {n_params:,} params")

    opts = build_optimizers(model, muon_lr=args.muon_lr, adam_lr=args.adam_lr)
    sched = WarmupCosineScheduler(opts, warmup_steps=1000, total_steps=args.steps)

    data_cfg = IBMBudgetDataConfig(distance=args.distance, rounds=rounds,
                                   alpha_low=args.alpha_low, alpha_high=args.alpha_high,
                                   n_alphas=args.n_alphas, batch_size=args.batch_size)
    ds = IBMBudgetSyndromeDataset(data_cfg, rng=np.random.default_rng(args.seed))
    print(f"IBM budget (alpha=1.0, d={args.distance}): {ds.budget}")
    print(f"Pre-compiled {args.n_alphas} IBM-budget samplers, alpha in [{args.alpha_low}, {args.alpha_high}]")

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    model.train(True)
    best_a1_ler = 1.0   # track LER at alpha=1.0 (the IBM op-point) specifically
    t0 = time.time()
    train_log = []
    alpha_one = min(ds.samplers.keys(), key=lambda a: abs(a - 1.0))

    for step in range(args.steps):
        syn, lab, a = ds.sample()
        syn, lab = syn.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        for opt in opts:
            opt.zero_grad()
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(syn)
                loss = F.binary_cross_entropy_with_logits(logits, lab)
            scaler.scale(loss).backward()
            for opt in opts:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in opts:
                scaler.step(opt)
            scaler.update()
        else:
            logits = model(syn)
            loss = F.binary_cross_entropy_with_logits(logits, lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in opts:
                opt.step()
        sched.step(step)

        if step % args.log_interval == 0:
            dt = time.time() - t0
            sps = (step + 1) / max(dt, 0.001)
            eta = (args.steps - step) / max(sps, 0.001) / 60
            print(f"step {step:>6}/{args.steps} loss={loss.item():.4f} alpha={a:.3f} "
                  f"lr={opts[0].param_groups[0]['lr']:.6f} {sps:.1f}sps ETA {eta:.0f}min", flush=True)

        if step > 0 and step % args.eval_interval == 0:
            per_a = measure_per_alpha(model, ds, device, n_shots=args.eval_shots)
            a1 = per_a[alpha_one]
            avg = float(np.mean(list(per_a.values())))
            print(f"  >>> alpha=1.0 (IBM op-point) LER={a1:.4f}  avg LER={avg:.4f}", flush=True)
            train_log.append({"step": step, "alpha1_ler": a1, "avg_ler": avg, "per_alpha": per_a,
                              "elapsed_s": time.time() - t0})
            if a1 < best_a1_ler:
                best_a1_ler = a1
                torch.save({'step': step, 'model_state_dict': model.state_dict(),
                            'config': cfg, 'ler': a1, 'per_alpha': per_a,
                            'grid_shape': ds.grid_shape, 'train_log': train_log,
                            'args': vars(args), 'ibm_budget': ds.budget},
                           out_dir / 'best_model.pt')
                print(f"  >>> saved best (alpha=1.0 LER={a1:.4f})", flush=True)

    per_a = measure_per_alpha(model, ds, device, n_shots=20000)
    a1_final = per_a[alpha_one]
    print(f"\nFinal alpha=1.0 LER: {a1_final:.4f}, best: {best_a1_ler:.4f}")
    print(f"Per-alpha: {per_a}")
    torch.save({'step': args.steps, 'model_state_dict': model.state_dict(),
                'config': cfg, 'ler': a1_final, 'per_alpha': per_a,
                'grid_shape': ds.grid_shape, 'train_log': train_log,
                'args': vars(args), 'ibm_budget': ds.budget},
               out_dir / 'final_model.pt')
    with open(out_dir / 'train_log.json', 'w') as f:
        json.dump({'args': vars(args), 'best_alpha1_ler': best_a1_ler,
                   'final_per_alpha': per_a, 'log': train_log}, f, indent=2)


if __name__ == "__main__":
    main()
