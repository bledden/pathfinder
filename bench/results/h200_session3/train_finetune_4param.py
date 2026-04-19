"""Fine-tune an existing Pathfinder checkpoint at 4-parameter circuit-level noise.

Motivation: fixed_d7 from-scratch at 4-parameter noise failed catastrophically
in session 2 (stuck at ~40% LER through 80K steps). Hypothesis: the optimizer
never found a good basin from random init on the harder noise model. Starting
from the Table-1 3-parameter checkpoint gives a known-good initialization, and
a short fine-tune at lower LR should adapt the model to the 4-parameter noise.

Usage:
  python3 train_finetune_4param.py --distance 7 --init /workspace/pathfinder/train/checkpoints/d7_final/best_model.pt --steps 40000 --ckpt /workspace/persist/checkpoints/finetune_d7
"""
import sys, os, time, math, argparse
sys.path.insert(0, "/workspace/pathfinder/train")
import numpy as np, torch
import torch.nn.functional as F
import stim
from muon import SingleDeviceMuon
from model import NeuralDecoder, DecoderConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SyndromeDataset4Param:
    def __init__(self, distance, rounds, p, batch_size=256):
        self.circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=distance, rounds=rounds,
            after_clifford_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p,
            before_round_data_depolarization=p,
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
        tm_m = {v: i for i, v in enumerate(tu)}
        xm = {v: i for i, v in enumerate(xu)}
        ym = {v: i for i, v in enumerate(yu)}
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


def measure_ler(model, distance, rounds, p, n_shots=10000):
    model.eval()
    ds = SyndromeDataset4Param(distance, rounds, p, batch_size=1000)
    errors = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_shots // 1000):
            syn, lab = ds.sample()
            syn = syn.to(device); lab = lab.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(syn)
            preds = (lg > 0).float()
            errors += int((preds != lab).any(dim=1).sum().item())
            total += 1000
    model.train()
    return errors / total


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--distance", type=int, required=True)
    a.add_argument("--init", type=str, required=True, help="Path to init checkpoint (.pt)")
    a.add_argument("--steps", type=int, default=40000)
    a.add_argument("--batch", type=int, default=256)
    a.add_argument("--noise_rate", type=float, default=0.007)
    a.add_argument("--muon_lr", type=float, default=0.005)
    a.add_argument("--adam_lr", type=float, default=1e-3)
    a.add_argument("--log_interval", type=int, default=100)
    a.add_argument("--measure_interval", type=int, default=2000)
    a.add_argument("--ckpt", type=str, required=True)
    args = a.parse_args()
    os.makedirs(args.ckpt, exist_ok=True)

    print(f"Loading init from {args.init}", flush=True)
    init = torch.load(args.init, weights_only=False, map_location=device)
    config = init["config"]
    assert config.distance == args.distance, f"init distance {config.distance} != requested {args.distance}"
    model = NeuralDecoder(config).to(device)
    model.load_state_dict(init["model_state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded d={config.distance} H={config.hidden_dim} L={config.distance} {n_params:,} params", flush=True)

    # Measure init LER at 4-param noise (baseline we're trying to improve)
    init_ler = measure_ler(model, args.distance, args.distance, args.noise_rate, n_shots=10000)
    print(f"Init LER @ 4-param p={args.noise_rate}: {init_ler:.5f}", flush=True)

    # Pre-compile sampler at target noise only (no curriculum — we start from a good init)
    ds = SyndromeDataset4Param(args.distance, args.distance, args.noise_rate, batch_size=args.batch)

    muon_params = [p for p in model.parameters() if p.ndim == 2 and p.requires_grad]
    adam_params = [p for p in model.parameters() if p.ndim != 2 and p.requires_grad]
    opts = [SingleDeviceMuon(muon_params, lr=args.muon_lr, momentum=0.95, weight_decay=0.01),
            torch.optim.AdamW(adam_params, lr=args.adam_lr, weight_decay=0.0)]
    base_lrs = [[pg['lr'] for pg in opt.param_groups] for opt in opts]
    warmup = 500  # short warmup because we start from a good init

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
    best = init_ler
    torch.save({'step': 0, 'model_state_dict': model.state_dict(), 'config': config, 'ler': init_ler},
               f"{args.ckpt}/best_model.pt")
    print(f"  >>> saved init checkpoint (ler={init_ler:.5f})", flush=True)
    t0 = time.time()

    for step in range(args.steps):
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
            print(f"step {step:>6}/{args.steps}  loss={loss.item():.4f}  {sps:.1f} s/s  ETA {eta:.0f}min", flush=True)

        if step > 0 and step % args.measure_interval == 0:
            ler = measure_ler(model, args.distance, args.distance, args.noise_rate)
            print(f"  >>> LER @ p={args.noise_rate}: {ler:.5f}", flush=True)
            if ler < best:
                best = ler
                torch.save({'step': step, 'model_state_dict': model.state_dict(), 'config': config, 'ler': ler},
                           f"{args.ckpt}/best_model.pt")
                print(f"  >>> saved (ler={ler:.5f})", flush=True)

    final_ler = measure_ler(model, args.distance, args.distance, args.noise_rate, n_shots=50000)
    print(f"\nFinal LER @ p={args.noise_rate} (50K shots): {final_ler:.5f}", flush=True)
    torch.save({'step': args.steps, 'model_state_dict': model.state_dict(), 'config': config, 'final_ler': final_ler},
               f"{args.ckpt}/final_model.pt")


if __name__ == "__main__":
    main()
