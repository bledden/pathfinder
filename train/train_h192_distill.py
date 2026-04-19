"""
Knowledge distillation: narrow H=128 student learns from full H=256 teacher.
Teacher is d7_final (best full model). Student trains on BCE(label) + alpha*KL(logits,teacher_logits).
Matches full-model training setup otherwise (curriculum, Muon, mixed precision).
"""
import sys, os, time, math, torch, torch.nn.functional as F
from pathlib import Path
from muon import SingleDeviceMuon
import torch.optim
torch.optim.Muon = SingleDeviceMuon

sys.path.insert(0, "/workspace/pathfinder/train")
from model import NeuralDecoder, DecoderConfig
from data import SyndromeDataset, DataConfig, CurriculumScheduler


# --- Hyperparams ---
class Args:
    distance = 7
    hidden_dim = 192
    steps = 80000           # longer than narrow (60K) to give KL time to work
    batch_size = 512
    muon_lr = 0.02
    adam_lr = 1e-3
    noise_rate = 0.007
    log_interval = 500
    eval_interval = 5000
    eval_shots = 10000
    temperature = 2.0
    alpha_kl = 0.7           # soft targets weight
    alpha_bce = 0.3          # hard labels weight
    teacher_ckpt = "/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt"
    checkpoint_dir = "/workspace/pathfinder/train/checkpoints/d7_h192_distill"


def build_opts(model, muon_lr, adam_lr):
    muon_params, adam_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim == 2: muon_params.append(p)
        else: adam_params.append(p)
    opts = []
    if muon_params: opts.append(SingleDeviceMuon(muon_params, lr=muon_lr, momentum=0.95, weight_decay=0.01))
    if adam_params: opts.append(torch.optim.AdamW(adam_params, lr=adam_lr, weight_decay=0.0))
    return opts


def evaluate(model, d, noise, n_shots):
    model.eval()
    ds = SyndromeDataset(DataConfig(distance=d, rounds=d, physical_error_rate=noise))
    errors = 0; total = 0; batch = 1000
    for _ in range(n_shots // batch):
        synd, lab = ds.sample(batch)
        synd, lab = synd.cuda(), lab.cuda()
        with torch.no_grad():
            logits = model(synd)
        preds = (logits > 0).float()
        errors += (preds != lab).any(dim=1).sum().item()
        total += batch
    model.train()
    return errors / max(total, 1)


def main():
    args = Args()
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # --- Load teacher ---
    teacher_ck = torch.load(args.teacher_ckpt, weights_only=False, map_location="cuda")
    teacher = NeuralDecoder(teacher_ck["config"]).cuda().eval()
    teacher.load_state_dict(teacher_ck["model_state_dict"])
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher: H={teacher_ck['config'].hidden_dim}, {sum(p.numel() for p in teacher.parameters()):,} params")

    # --- Build student ---
    config = DecoderConfig(distance=args.distance, rounds=args.distance, hidden_dim=args.hidden_dim, n_observables=1)
    student = NeuralDecoder(config).cuda()
    n_student = sum(p.numel() for p in student.parameters())
    print(f"Student: H={config.hidden_dim}, {n_student:,} params ({n_student * 2 / 1e6:.2f} MB FP16)")

    # --- Optimizers, curriculum ---
    opts = build_opts(student, args.muon_lr, args.adam_lr)
    curriculum = CurriculumScheduler(args.noise_rate, args.steps)

    # Pre-compile samplers
    rates = sorted(set(round(curriculum.get_rate(s), 4) for s in range(0, args.steps, max(args.steps // 50, 1))) | {round(args.noise_rate, 4)})
    cache = {p: SyndromeDataset(DataConfig(distance=args.distance, rounds=args.distance, physical_error_rate=max(p, 1e-6), batch_size=args.batch_size)) for p in rates}
    print(f"Samplers: {len(cache)}")
    def get_ds(p): return cache[min(cache.keys(), key=lambda k: abs(k-p))]

    # --- LR schedule ---
    warmup = 1000
    base_lrs = [[pg['lr'] for pg in opt.param_groups] for opt in opts]
    def step_lr(s):
        if s < warmup:
            scale = s / max(warmup, 1)
        else:
            prog = (s - warmup) / max(args.steps - warmup, 1)
            scale = 0.5 * (1 + math.cos(math.pi * prog))
        for opt, lrs in zip(opts, base_lrs):
            for pg, lr in zip(opt.param_groups, lrs):
                pg['lr'] = lr * scale

    scaler = torch.amp.GradScaler("cuda")
    T = args.temperature
    student.train()
    best_ler = 1.0
    t0 = time.time()

    for step in range(args.steps):
        p = curriculum.get_rate(step)
        ds = get_ds(p)
        synd, lab = ds.sample()
        synd, lab = synd.cuda(), lab.cuda()

        for opt in opts: opt.zero_grad()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Teacher (no grad, on same syndromes)
            with torch.no_grad():
                t_logits = teacher(synd)
            s_logits = student(synd)

            # Hard-label loss
            loss_bce = F.binary_cross_entropy_with_logits(s_logits, lab)

            # Soft-label distillation loss — KL between tempered sigmoids
            # Using BCE with teacher probabilities as a stand-in for KL on binary outputs
            t_prob = torch.sigmoid(t_logits / T)
            loss_kl = F.binary_cross_entropy_with_logits(s_logits / T, t_prob) * (T * T)

            loss = args.alpha_bce * loss_bce + args.alpha_kl * loss_kl

        scaler.scale(loss).backward()
        for opt in opts:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        for opt in opts:
            scaler.step(opt)
        scaler.update()
        step_lr(step)

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            sps = (step + 1) / max(elapsed, 1)
            eta = (args.steps - step) / max(sps, 0.01)
            print(f"step {step:>6}/{args.steps}  bce={loss_bce.item():.4f}  kl={loss_kl.item():.4f}  p={p:.5f}  {sps:.1f} s/s  ETA {eta/60:.0f}min", flush=True)

        if step > 0 and step % args.eval_interval == 0:
            ler = evaluate(student, args.distance, args.noise_rate, args.eval_shots)
            print(f"  >>> EVAL LER @ p={args.noise_rate}: {ler:.6f}", flush=True)
            if ler < best_ler:
                best_ler = ler
                torch.save({
                    'step': step, 'model_state_dict': student.state_dict(),
                    'config': config, 'ler': ler,
                }, f"{args.checkpoint_dir}/best_model.pt")
                print(f"  >>> saved (LER={ler:.6f})", flush=True)

    final_ler = evaluate(student, args.distance, args.noise_rate, 50000)
    print(f"\nFinal LER @ p={args.noise_rate}: {final_ler:.6f}")
    print(f"Best during training: {best_ler:.6f}")
    print(f"Total time: {time.time() - t0:.0f}s")
    torch.save({'step': args.steps, 'model_state_dict': student.state_dict(), 'config': config, 'ler': final_ler},
               f"{args.checkpoint_dir}/final_model.pt")


if __name__ == "__main__":
    main()
