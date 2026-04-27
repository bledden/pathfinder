"""Triad-distillation: train Pathfinder student to match the 3-way majority
vote of (existing PFWL3S 3-seed avg) + Lange + PyMatching.

The Triad teacher is much stronger than Lange alone (e.g. d=7 p=0.007:
Triad 2.38% vs Lange 2.94% LER). If the student can absorb this signal,
3-seed averaging of these new students may strictly beat the original Triad
without needing PM/Lange at inference time.

Loss = α_bce * BCE(student, true_labels) + α_kl * T² * KL(σ(student/T), soft_triad)
where soft_triad = (σ(PF_avg_logit) + σ(Lange_logit) + PM_binary) / 3
"""
import sys, os, time, math, argparse
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace/GNN_decoder")
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import stim, pymatching
from torch_geometric.nn import knn_graph
from muon import SingleDeviceMuon
import torch.optim
torch.optim.Muon = SingleDeviceMuon

from model import NeuralDecoder, DecoderConfig
from src.gnn_models import GNN_7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LangeTeacher:
    def __init__(self, d, d_t):
        self.d = d; self.d_t = d_t
        self.model = GNN_7(
            hidden_channels_GCN=([32,128,256,512,512,512,512] if self.d==9 else [32,128,256,512,512,256,256]),
            hidden_channels_MLP=([512,256,128,64,32,16] if self.d==9 else [256,128,64]), num_classes=1).to(device)
        ck = torch.load(f"/workspace/GNN_decoder/models/circuit_level_noise/d{d}/d{d}_d_t_{d_t}.pt",
                        weights_only=False, map_location=device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False
        self.m_nearest = 10; self.power = 2
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

    def teacher_logits(self, det_events, det_coords, syn_mask):
        B = det_events.shape[0]
        any_flip = np.sum(det_events, axis=1) != 0
        out_logits = torch.zeros(B, 1, device=device)
        if not np.any(any_flip): return out_logits
        det_nt = det_events[any_flip]
        mask = np.repeat(syn_mask[None, ...], det_nt.shape[0], 0)
        s3d = np.zeros_like(mask)
        s3d[:, det_coords[:,1], det_coords[:,0], det_coords[:,2]] = det_nt
        s3d[np.nonzero(s3d)] = mask[np.nonzero(s3d)]
        s3d = s3d.astype(np.float32)
        inds = np.nonzero(s3d); defs = s3d[inds]; inds_t = np.transpose(np.array(inds))
        xd = defs == 1; zd = defs == 3
        nf = np.zeros((defs.shape[0], 6), dtype=np.float32)
        nf[xd, 0] = 1; nf[xd, 2:] = inds_t[xd]
        nf[zd, 1] = 1; nf[zd, 2:] = inds_t[zd]
        x = torch.tensor(nf[:, [0,1,3,4,5]]).to(device)
        batch = torch.tensor(nf[:,2]).long().to(device)
        pos = x[:,2:]
        ei = knn_graph(pos, self.m_nearest, batch=batch)
        dist = torch.sqrt(((pos[ei[0]] - pos[ei[1]])**2).sum(dim=1, keepdim=True))
        ea = 1.0 / (dist ** self.power)
        with torch.no_grad():
            out = self.model(x, ei, batch, ea)
        # Map back to per-batch indices
        nz_idx = np.nonzero(any_flip)[0]
        unique_batch_in_g = torch.unique(batch).cpu().numpy()
        for i, b in enumerate(unique_batch_in_g):
            out_logits[nz_idx[b], 0] = out[i, 0]
        return out_logits


class PFWL3STeacher:
    """Frozen ensemble of existing PFWL3S ckpts at this distance."""
    def __init__(self, ckpt_paths):
        self.models = []
        for p in ckpt_paths:
            ck = torch.load(p, weights_only=False, map_location=device)
            m = NeuralDecoder(ck["config"]).to(device); m.load_state_dict(ck["model_state_dict"]); m.eval()
            for pp in m.parameters(): pp.requires_grad = False
            self.models.append(m)

    def teacher_logits(self, syn):
        with torch.no_grad():
            avg = None
            for m in self.models:
                lg = m(syn)
                avg = lg if avg is None else avg + lg
            return avg / len(self.models)


class PMTeacher:
    """PyMatching teacher (CPU). Returns binary predictions per batch."""
    def __init__(self, circuit):
        dem = circuit.detector_error_model(decompose_errors=True)
        self.pm = pymatching.Matching.from_detector_error_model(dem)

    def teacher_binary(self, det_events):
        # det_events: numpy array [B, num_detectors] uint8
        return self.pm.decode_batch(det_events).astype(np.float32)  # [B, n_obs]


class SyndromeDSCorrected:
    def __init__(self, d, r, p, batch=512):
        self.d = d; self.r = r; self.p = p
        self.circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=d, rounds=r,
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
        self.nd = nd
        self.batch = batch

    def sample(self):
        det, obs = self.sampler.sample(self.batch, separate_observables=True)
        det = det.astype(np.uint8); obs = obs.astype(np.uint8)
        T, H, W = self.grid
        t = torch.zeros(self.batch, 1, T, H, W, dtype=torch.float32)
        d = torch.from_numpy(det.astype(np.float32))
        for i in range(self.nd):
            t[:, 0, self.det_idx[i, 0], self.det_idx[i, 1], self.det_idx[i, 2]] = d[:, i]
        return t, torch.from_numpy(obs.astype(np.float32)), det


def evaluate(model, d, r, p, n=10000, batch=1000):
    ds = SyndromeDSCorrected(d, r, p, batch)
    errs = total = 0
    model.eval()
    for _ in range(n // batch):
        syn, lab, _ = ds.sample()
        syn = syn.to(device); lab = lab.to(device)
        with torch.no_grad():
            pred = (model(syn) > 0).float()
        errs += int(((pred != lab).any(dim=1)).sum().item())
        total += batch
    model.train()
    return errs / total


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--distance", type=int, default=7)
    a.add_argument("--hidden_dim", type=int, default=384)
    a.add_argument("--steps", type=int, default=160000)
    a.add_argument("--batch", type=int, default=128)
    a.add_argument("--noise_rate", type=float, default=0.007)
    a.add_argument("--muon_lr", type=float, default=0.005)
    a.add_argument("--alpha_kl", type=float, default=0.7)
    a.add_argument("--alpha_bce", type=float, default=0.3)
    a.add_argument("--temperature", type=float, default=2.0)
    a.add_argument("--ckpt", type=str, required=True)
    a.add_argument("--init", type=str, default=None)
    a.add_argument("--pf_teacher_ckpts", type=str, nargs="+", required=True,
                   help="PFWL3S teacher ckpts (3-seed avg)")
    a.add_argument("--eval_interval", type=int, default=10000)
    a.add_argument("--log_interval", type=int, default=1000)
    args = a.parse_args()
    os.makedirs(args.ckpt, exist_ok=True)

    config = DecoderConfig(distance=args.distance, rounds=args.distance, hidden_dim=args.hidden_dim)
    student = NeuralDecoder(config).to(device)
    if args.init:
        ck = torch.load(args.init, weights_only=False, map_location=device)
        student.load_state_dict(ck["model_state_dict"])
        print(f"Loaded init from {args.init}", flush=True)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student (Pathfinder): d={args.distance}, H={args.hidden_dim}, {n_params:,} params", flush=True)

    # Three teachers
    lange = LangeTeacher(args.distance, args.distance)
    print(f"LangeTeacher loaded", flush=True)
    pf_teacher = PFWL3STeacher(args.pf_teacher_ckpts)
    print(f"PFWL3STeacher loaded ({len(pf_teacher.models)} ckpts)", flush=True)

    ds = SyndromeDSCorrected(args.distance, args.distance, args.noise_rate, args.batch)
    print(f"Dataset: p={args.noise_rate}, batch={args.batch}", flush=True)
    pm_teacher = PMTeacher(ds.circuit)
    print(f"PMTeacher loaded", flush=True)

    dc, syn_mask = lange.init_for_circuit(ds.circuit, args.noise_rate)

    muon_params = [p for p in student.parameters() if p.ndim == 2 and p.requires_grad]
    adam_params = [p for p in student.parameters() if p.ndim != 2 and p.requires_grad]
    opts = [SingleDeviceMuon(muon_params, lr=args.muon_lr, momentum=0.95, weight_decay=0.01),
            torch.optim.AdamW(adam_params, lr=1e-3, weight_decay=0.0)]
    base_lrs = [[pg["lr"] for pg in opt.param_groups] for opt in opts]
    warmup = 1000
    def step_lr(s):
        if s < warmup: scale = s / warmup
        else:
            prog = (s - warmup) / max(args.steps - warmup, 1)
            scale = 0.5 * (1 + math.cos(math.pi * prog))
        for opt, lrs in zip(opts, base_lrs):
            for pg, lr in zip(opt.param_groups, lrs):
                pg["lr"] = lr * scale

    scaler = torch.amp.GradScaler("cuda")
    student.train()
    best = 1.0
    t0 = time.time()
    T = args.temperature

    for step in range(args.steps):
        syn, lab, det = ds.sample()
        syn = syn.to(device); lab = lab.to(device)

        # All three teachers (frozen, no grad)
        with torch.no_grad():
            lange_logits = lange.teacher_logits(det, dc, syn_mask)  # [B, 1]
            pf_logits = pf_teacher.teacher_logits(syn)               # [B, 1]
            pm_binary = torch.from_numpy(pm_teacher.teacher_binary(det)).to(device)  # [B, n_obs]
            # Soft Triad target: average of probabilities
            soft_triad = (torch.sigmoid(lange_logits) + torch.sigmoid(pf_logits) + pm_binary) / 3.0
            # clamp to avoid log(0)
            soft_triad = soft_triad.clamp(1e-6, 1 - 1e-6)

        for opt in opts: opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_logits = student(syn)
            loss_bce = F.binary_cross_entropy_with_logits(student_logits, lab)
            # KL via soft target match (binary case = BCE with soft labels)
            loss_kl = F.binary_cross_entropy_with_logits(student_logits / T, soft_triad) * (T * T)
            loss = args.alpha_bce * loss_bce + args.alpha_kl * loss_kl

        scaler.scale(loss).backward()
        for opt in opts: scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        for opt in opts: scaler.step(opt)
        scaler.update()
        step_lr(step)

        if step % args.log_interval == 0:
            sps = (step+1) / max(time.time()-t0, 1)
            eta = (args.steps - step) / max(sps, 0.01) / 60
            print(f"step {step:>6}/{args.steps}  bce={loss_bce.item():.4f}  kl={loss_kl.item():.4f}  {sps:.1f} s/s  ETA {eta:.0f}min", flush=True)

        if step > 0 and step % args.eval_interval == 0:
            ler = evaluate(student, args.distance, args.distance, args.noise_rate)
            print(f"  >>> EVAL LER @ p={args.noise_rate}: {ler:.5f}", flush=True)
            if ler < best:
                best = ler
                torch.save({"step": step, "model_state_dict": student.state_dict(),
                            "config": config, "ler": ler, "distilled_from_triad": True},
                           f"{args.ckpt}/best_model.pt")
                print(f"  >>> saved (ler={ler:.5f})", flush=True)

    torch.save({"step": args.steps, "model_state_dict": student.state_dict(),
                "config": config, "distilled_from_triad": True},
               f"{args.ckpt}/final_model.pt")
    print(f"Training done. Best LER: {best:.5f}", flush=True)


if __name__ == "__main__":
    main()
