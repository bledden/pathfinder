"""Multi-noise-mixture distillation: sample noise rate p uniformly from a set each batch.
Based on train_distill_lange_lowlr.py but with per-step p selection."""
import sys, os, time, math, argparse
sys.path.insert(0, '/workspace/pathfinder/train')
sys.path.insert(0, '/workspace/GNN_decoder')
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import stim
from torch_geometric.nn import knn_graph
from muon import SingleDeviceMuon
import torch.optim
torch.optim.Muon = SingleDeviceMuon

from model import NeuralDecoder, DecoderConfig
from src.gnn_models import GNN_7

device = torch.device('cuda')


class LangeTeacher:
    def __init__(self, d, d_t):
        self.d = d; self.d_t = d_t
        self.model = GNN_7(
            hidden_channels_GCN=[32, 128, 256, 512, 512, 512, 512] if d == 9 else [32, 128, 256, 512, 512, 256, 256],
            hidden_channels_MLP=[512, 256, 128, 64, 32, 16] if d == 9 else [256, 128, 64],
            num_classes=1).to(device)
        ck = torch.load(f'/workspace/GNN_decoder/models/circuit_level_noise/d{d}/d{d}_d_t_{d_t}.pt', weights_only=False, map_location=device)
        self.model.load_state_dict(ck['model']); self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False
        self.m = 10; self.power = 2
        self._cache = {}

    def init_for_circuit(self, circuit, p):
        if p in self._cache: return self._cache[p]
        coords = circuit.get_detector_coordinates()
        dc = np.array(list(coords.values())); dc[:, :2] = dc[:, :2] / 2
        dc = dc.astype(np.uint8)
        sz = self.d + 1
        sx = np.zeros((sz, sz), dtype=np.uint8); sx[::2, 1:sz-1:2] = 1; sx[1::2, 2::2] = 1
        smz = np.rot90(sx) * 3
        syn_mask = np.dstack([sx + smz] * (self.d_t + 1))
        self._cache[p] = (dc, syn_mask)
        return self._cache[p]

    def teacher_logits(self, det, dc, syn_mask):
        B = det.shape[0]
        mask = np.repeat(syn_mask[None, ...], B, 0)
        s3d = np.zeros_like(mask)
        s3d[:, dc[:, 1], dc[:, 0], dc[:, 2]] = det
        s3d[np.nonzero(s3d)] = mask[np.nonzero(s3d)]
        s3d = s3d.astype(np.float32)
        inds = np.nonzero(s3d); defs = s3d[inds]
        if defs.shape[0] == 0:
            return torch.zeros(B, 1, device=device)
        inds_t = np.transpose(np.array(inds))
        xd = defs == 1; zd = defs == 3
        nf = np.zeros((defs.shape[0], 6), dtype=np.float32)
        nf[xd, 0] = 1; nf[xd, 2:] = inds_t[xd]
        nf[zd, 1] = 1; nf[zd, 2:] = inds_t[zd]
        x_cols = [0, 1, 3, 4, 5]
        x = torch.tensor(nf[:, x_cols]).to(device)
        batch = torch.tensor(nf[:, 2]).long().to(device)
        pos = x[:, 2:]
        ei = knn_graph(pos, self.m, batch=batch)
        dist = torch.sqrt(((pos[ei[0],:] - pos[ei[1],:])**2).sum(dim=1, keepdim=True))
        ea = 1.0 / (dist ** self.power)
        logits_nt = self.model(x, ei, batch, ea)
        out = torch.zeros(B, 1, device=device)
        any_defects = np.sum(det, axis=1) > 0
        nz_idx = np.where(any_defects)[0]
        if len(nz_idx) == logits_nt.shape[0]:
            out[nz_idx] = logits_nt
        return out


class SyndromeDS:
    def __init__(self, d, r, p, batch):
        self.circuit = stim.Circuit.generated('surface_code:rotated_memory_z', distance=d, rounds=r,
            after_clifford_depolarization=p, before_measure_flip_probability=p,
            after_reset_flip_probability=p, before_round_data_depolarization=p)
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
        self.nd = nd; self.batch = batch

    def sample(self):
        det, obs = self.sampler.sample(self.batch, separate_observables=True)
        B = det.shape[0]; T, H, W = self.grid
        t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        d = torch.from_numpy(det.astype(np.float32))
        for i in range(self.nd):
            t[:, 0, self.det_idx[i, 0], self.det_idx[i, 1], self.det_idx[i, 2]] = d[:, i]
        return t, torch.from_numpy(obs.astype(np.float32)), det


def measure_ler(m, d, r, p, n=10000, batch=500):
    m.eval()
    ds = SyndromeDS(d, r, p, batch)
    errs = tot = 0
    with torch.no_grad():
        for _ in range(n // batch):
            syn, lab, _ = ds.sample()
            syn = syn.to(device); lab = lab.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = m(syn)
            preds = (lg > 0).float()
            errs += int((preds != lab).any(dim=1).sum().item())
            tot += batch
    m.train()
    return errs / tot


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--distance', type=int, default=7)
    a.add_argument('--hidden_dim', type=int, default=384)
    a.add_argument('--steps', type=int, default=80000)
    a.add_argument('--batch', type=int, default=128)
    a.add_argument('--noise_rates', type=float, nargs='+', default=[0.003, 0.005, 0.007, 0.010])
    a.add_argument('--eval_p', type=float, default=0.007)
    a.add_argument('--alpha_kl', type=float, default=0.7)
    a.add_argument('--alpha_bce', type=float, default=0.3)
    a.add_argument('--temperature', type=float, default=2.0)
    a.add_argument('--ckpt', type=str, required=True)
    a.add_argument('--eval_interval', type=int, default=4000)
    a.add_argument('--log_interval', type=int, default=500)
    args = a.parse_args()
    os.makedirs(args.ckpt, exist_ok=True)

    config = DecoderConfig(distance=args.distance, rounds=args.distance, hidden_dim=args.hidden_dim)
    student = NeuralDecoder(config).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f'Student: d={args.distance}, H={args.hidden_dim}, {n_params:,} params', flush=True)

    teacher = LangeTeacher(args.distance, args.distance)
    print(f'Teacher loaded and frozen. Noise rates: {args.noise_rates}', flush=True)

    # Pre-build one dataset per noise rate + teacher cache
    ds_map = {}; tc_map = {}
    for p in args.noise_rates:
        ds_map[p] = SyndromeDS(args.distance, args.distance, p, args.batch)
        tc_map[p] = teacher.init_for_circuit(ds_map[p].circuit, p)
    print(f'Pre-built {len(args.noise_rates)} samplers', flush=True)

    muon_params = [p for p in student.parameters() if p.ndim == 2 and p.requires_grad]
    adam_params = [p for p in student.parameters() if p.ndim != 2 and p.requires_grad]
    opts = [SingleDeviceMuon(muon_params, lr=0.005, momentum=0.95, weight_decay=0.01),
            torch.optim.AdamW(adam_params, lr=1e-3, weight_decay=0.0)]
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

    scaler = torch.amp.GradScaler('cuda')
    student.train()
    best = 1.0; t0 = time.time()
    T = args.temperature
    rng = np.random.default_rng(42)

    for step in range(args.steps):
        p = args.noise_rates[rng.integers(len(args.noise_rates))]
        syn, lab, det = ds_map[p].sample()
        dc, smask = tc_map[p]
        syn = syn.to(device); lab = lab.to(device)
        with torch.no_grad():
            tl = teacher.teacher_logits(det, dc, smask)
        for opt in opts: opt.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            sl = student(syn)
            l_bce = F.binary_cross_entropy_with_logits(sl, lab)
            t_prob = torch.sigmoid(tl / T)
            l_kl = F.binary_cross_entropy_with_logits(sl / T, t_prob) * (T * T)
            loss = args.alpha_bce * l_bce + args.alpha_kl * l_kl
        scaler.scale(loss).backward()
        for opt in opts: scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        for opt in opts: scaler.step(opt)
        scaler.update()
        step_lr(step)

        if step % args.log_interval == 0:
            sps = (step+1) / max(time.time()-t0, 1)
            eta = (args.steps - step) / max(sps, 0.01) / 60
            print(f'step {step:>6}/{args.steps}  p={p:.4f}  bce={l_bce.item():.4f}  kl={l_kl.item():.4f}  {sps:.1f}s/s  ETA {eta:.0f}min', flush=True)
        if step > 0 and step % args.eval_interval == 0:
            ler = measure_ler(student, args.distance, args.distance, args.eval_p)
            print(f'  >>> EVAL LER @ p={args.eval_p}: {ler:.5f}', flush=True)
            if ler < best:
                best = ler
                torch.save({'step': step, 'model_state_dict': student.state_dict(), 'config': config, 'ler': ler},
                           f'{args.ckpt}/best_model.pt')
                print(f'  >>> saved (ler={ler:.5f})', flush=True)

    final = measure_ler(student, args.distance, args.distance, args.eval_p, n=50000)
    print(f'\nFinal LER @ p={args.eval_p} (50K shots): {final:.5f}', flush=True)
    torch.save({'step': args.steps, 'model_state_dict': student.state_dict(), 'config': config, 'final_ler': final},
               f'{args.ckpt}/final_model.pt')


if __name__ == '__main__':
    main()
