"""Fine-tune PFWL3S on IBM-calibrated syndromes (from Aer NoiseModel).

Loads a pre-generated .npz of (detectors, observables) sampled from
AerSimulator with NoiseModel.from_backend(ibm_fez), and fine-tunes from
the existing calibrated PFWL3S checkpoint.

Compared to train_calibrated.py (on-the-fly uniform Stim noise), this
trains on structured per-qubit IBM noise — closes the noise-OOD bias
that left calibrated PFWL3S tied with PM on real IBM data.
"""
import argparse, os, sys, time, math, json
import numpy as np
import torch
import torch.nn.functional as F
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


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_optimizers(model, muon_lr=0.005, adam_lr=2e-4, weight_decay=0.01):
    """Lower LR than train_calibrated since fine-tuning."""
    muon_params, adam_params = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad: continue
        (muon_params if p.ndim == 2 else adam_params).append(p)
    opts = []
    if muon_params:
        opts.append(Muon(muon_params, lr=muon_lr, momentum=0.95, weight_decay=weight_decay))
    if adam_params:
        opts.append(AdamW(adam_params, lr=adam_lr, weight_decay=0.0))
    return opts


class IBMCalibratedDataset:
    """Loads pre-generated (detectors, observables) from .npz and yields batches."""
    def __init__(self, npz_path, distance, rounds, batch_size):
        data = np.load(npz_path)
        self.detectors = data['detectors'].astype(np.uint8)
        self.observables = data['observables'].astype(np.uint8)
        self.distance = distance
        self.rounds = rounds
        self.batch_size = batch_size
        self.n_total = self.detectors.shape[0]
        print(f"Loaded IBM-calibrated dataset: {self.n_total} shots, "
              f"det_flip={self.detectors.mean():.4f}, obs_flip={self.observables.mean():.4f}")
        # Build coordinate map from a reference Stim circuit
        import stim
        circ = stim.Circuit.generated('surface_code:rotated_memory_z',
                                       distance=distance, rounds=rounds)
        coords = circ.get_detector_coordinates()
        ac = np.array([coords[i] for i in range(circ.num_detectors)])
        sp, tm = ac[:, :-1], ac[:, -1]
        tu = np.sort(np.unique(tm))
        xu = np.sort(np.unique(sp[:, 0]))
        yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
        self.grid = (len(tu), len(yu), len(xu))
        tm_m = {v: i for i, v in enumerate(tu)}
        xm = {v: i for i, v in enumerate(xu)}
        ym = {v: i for i, v in enumerate(yu)}
        di = np.zeros((circ.num_detectors, 3), dtype=np.int64)
        for did in range(circ.num_detectors):
            c = coords[did]
            di[did] = [tm_m[c[-1]], ym.get(c[1], 0) if len(c) > 2 else 0, xm[c[0]]]
        self.det_idx = di
        self.n_det = circ.num_detectors
        self.rng = np.random.default_rng(0)

    def to_tensor(self, det):
        B = det.shape[0]
        T, H, W = self.grid
        t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        d = torch.from_numpy(det.astype(np.float32))
        for i in range(self.n_det):
            t[:, 0, self.det_idx[i, 0], self.det_idx[i, 1], self.det_idx[i, 2]] = d[:, i]
        return t

    def sample(self):
        idx = self.rng.integers(0, self.n_total, size=self.batch_size)
        det = self.detectors[idx]
        obs = self.observables[idx].astype(np.float32)
        return self.to_tensor(det), torch.from_numpy(obs)


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
            progress = (s - self.warmup) / max(self.total - self.warmup, 1)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for opt, base in zip(self.opts, self.base_lrs):
            for pg, b in zip(opt.param_groups, base):
                pg['lr'] = b * scale


def evaluate_on_held_out(model, ds, device, n=2000):
    model.train(False)
    held_idx = np.arange(min(n, ds.n_total))
    det = ds.detectors[held_idx]
    obs = ds.observables[held_idx]
    syndromes = ds.to_tensor(det).to(device)
    with torch.no_grad():
        logits = model(syndromes)
        preds = (logits.cpu().numpy() > 0).astype(np.uint8)
    err = np.any(preds != obs, axis=1).sum()
    model.train(True)
    return err / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", required=True, help="path to PFWL3S checkpoint to resume from")
    ap.add_argument("--npz", required=True, help="IBM-calibrated syndromes .npz")
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--hidden_dim", type=int, default=384)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--muon_lr", type=float, default=0.005)
    ap.add_argument("--adam_lr", type=float, default=2e-4)
    ap.add_argument("--log_interval", type=int, default=200)
    ap.add_argument("--eval_interval", type=int, default=2000)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    cfg = DecoderConfig(distance=args.distance, rounds=args.rounds,
                        hidden_dim=args.hidden_dim, n_observables=1)
    model = NeuralDecoder(cfg).to(device)

    # Load init ckpt
    print(f"Loading init: {args.init_ckpt}")
    ck = torch.load(args.init_ckpt, weights_only=False, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    print(f"  init LER (from ckpt training): {ck.get('ler', 'N/A')}")

    opts = build_optimizers(model, muon_lr=args.muon_lr, adam_lr=args.adam_lr)
    sched = WarmupCosineScheduler(opts, warmup_steps=500, total_steps=args.steps)

    ds = IBMCalibratedDataset(args.npz, args.distance, args.rounds, args.batch_size)

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    init_held_ler = evaluate_on_held_out(model, ds, device, n=2000)
    print(f"Init held-out LER on IBM-calibrated data: {init_held_ler:.4f}")
    best_held_ler = init_held_ler
    train_log = []
    t0 = time.time()
    model.train(True)
    for step in range(args.steps):
        syn, lab = ds.sample()
        syn, lab = syn.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        for opt in opts: opt.zero_grad()
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(syn)
                loss = F.binary_cross_entropy_with_logits(logits, lab)
            scaler.scale(loss).backward()
            for opt in opts: scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in opts: scaler.step(opt)
            scaler.update()
        else:
            logits = model(syn)
            loss = F.binary_cross_entropy_with_logits(logits, lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in opts: opt.step()
        sched.step(step)
        if step % args.log_interval == 0:
            dt = time.time() - t0
            sps = (step + 1) / max(dt, 0.001)
            eta = (args.steps - step) / max(sps, 0.001) / 60
            print(f"step {step:>5}/{args.steps} loss={loss.item():.4f} "
                  f"lr={opts[0].param_groups[0]['lr']:.6f} {sps:.1f}sps ETA {eta:.1f}min", flush=True)
        if step > 0 and step % args.eval_interval == 0:
            held = evaluate_on_held_out(model, ds, device, n=3000)
            print(f"  >>> held-out LER on IBM-calib: {held:.4f}", flush=True)
            train_log.append({'step': step, 'held_ler': held, 'elapsed_s': time.time() - t0})
            if held < best_held_ler:
                best_held_ler = held
                torch.save({'step': step, 'model_state_dict': model.state_dict(),
                            'config': cfg, 'ler': held, 'train_log': train_log,
                            'args': vars(args), 'init_ckpt': args.init_ckpt,
                            'npz': args.npz},
                           out_dir / 'best_model.pt')
                print(f"  >>> saved best (held LER={held:.4f})", flush=True)

    final = evaluate_on_held_out(model, ds, device, n=5000)
    print(f"\nFinal held-out LER: {final:.4f}, best: {best_held_ler:.4f}")
    print(f"Total time: {(time.time()-t0)/60:.1f}min")
    torch.save({'step': args.steps, 'model_state_dict': model.state_dict(),
                'config': cfg, 'ler': final, 'train_log': train_log,
                'args': vars(args), 'init_ckpt': args.init_ckpt, 'npz': args.npz},
               out_dir / 'final_model.pt')
    with open(out_dir / 'train_log.json', 'w') as f:
        json.dump({'args': vars(args), 'init_held_ler': init_held_ler,
                   'best_held_ler': best_held_ler, 'final_held_ler': final,
                   'log': train_log}, f, indent=2)


if __name__ == "__main__":
    main()
