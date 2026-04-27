"""Hard-label Triad distillation: use Triad's binary majority vote as the
training LABEL (replacing true labels). Pure BCE loss, no soft KL term —
the student treats the Triad as ground truth.

Theory: if Triad's 2.38% LER is much better than true-label noise (which is
0% wrong by definition but the model can't learn perfectly), then training
to match Triad's output may give the student a smoother label distribution
that captures "what the Triad does well" more directly than KL distillation.
"""
import sys, os, time, math, argparse
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace/GNN_decoder")
import numpy as np, torch, torch.nn.functional as F
import stim, pymatching
from torch_geometric.nn import knn_graph
from muon import SingleDeviceMuon
import torch.optim
torch.optim.Muon = SingleDeviceMuon
from model import NeuralDecoder, DecoderConfig
from src.gnn_models import GNN_7

device = torch.device("cuda")

# Re-use teachers from the soft Triad script
sys.path.insert(0, "/workspace")
exec(open("/workspace/train_distill_triad.py").read().split("def main():")[0])

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--distance", type=int, default=7)
    a.add_argument("--hidden_dim", type=int, default=384)
    a.add_argument("--steps", type=int, default=160000)
    a.add_argument("--batch", type=int, default=128)
    a.add_argument("--noise_rate", type=float, default=0.007)
    a.add_argument("--alpha_true", type=float, default=0.3, help="weight on true labels")
    a.add_argument("--alpha_triad", type=float, default=0.7, help="weight on hard Triad labels")
    a.add_argument("--ckpt", type=str, required=True)
    a.add_argument("--init", type=str, default=None)
    a.add_argument("--pf_teacher_ckpts", type=str, nargs="+", required=True)
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
    print(f"Student: d={args.distance}, H={args.hidden_dim}, {n_params:,} params", flush=True)

    lange = LangeTeacher(args.distance, args.distance)
    pf_teacher = PFWL3STeacher(args.pf_teacher_ckpts)
    print(f"Teachers loaded: Lange + PFWL3S({len(pf_teacher.models)} ckpts)", flush=True)

    ds = SyndromeDSCorrected(args.distance, args.distance, args.noise_rate, args.batch)
    pm_teacher = PMTeacher(ds.circuit)
    dc, syn_mask = lange.init_for_circuit(ds.circuit, args.noise_rate)
    print(f"Dataset+PM ready, p={args.noise_rate}, batch={args.batch}", flush=True)

    muon_params = [p for p in student.parameters() if p.ndim == 2 and p.requires_grad]
    adam_params = [p for p in student.parameters() if p.ndim != 2 and p.requires_grad]
    opts = [SingleDeviceMuon(muon_params, lr=0.005, momentum=0.95, weight_decay=0.01),
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

    for step in range(args.steps):
        syn, lab, det = ds.sample()
        syn = syn.to(device); lab = lab.to(device)

        with torch.no_grad():
            lange_logits = lange.teacher_logits(det, dc, syn_mask)
            pf_logits = pf_teacher.teacher_logits(syn)
            pm_binary = torch.from_numpy(pm_teacher.teacher_binary(det)).to(device)
            # Hard Triad: majority vote
            lange_binary = (torch.sigmoid(lange_logits) > 0.5).float()
            pf_binary = (torch.sigmoid(pf_logits) > 0.5).float()
            triad_binary = ((lange_binary + pf_binary + pm_binary) >= 2).float()

        for opt in opts: opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_logits = student(syn)
            loss_true = F.binary_cross_entropy_with_logits(student_logits, lab)
            loss_triad = F.binary_cross_entropy_with_logits(student_logits, triad_binary)
            loss = args.alpha_true * loss_true + args.alpha_triad * loss_triad

        scaler.scale(loss).backward()
        for opt in opts: scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        for opt in opts: scaler.step(opt)
        scaler.update()
        step_lr(step)

        if step % args.log_interval == 0:
            sps = (step+1) / max(time.time()-t0, 1)
            eta = (args.steps - step) / max(sps, 0.01) / 60
            print(f"step {step:>6}/{args.steps}  true={loss_true.item():.4f}  triad={loss_triad.item():.4f}  {sps:.1f} s/s  ETA {eta:.0f}min", flush=True)

        if step > 0 and step % args.eval_interval == 0:
            ler = evaluate(student, args.distance, args.distance, args.noise_rate)
            print(f"  >>> EVAL LER @ p={args.noise_rate}: {ler:.5f}", flush=True)
            if ler < best:
                best = ler
                torch.save({"step": step, "model_state_dict": student.state_dict(),
                            "config": config, "ler": ler, "distilled_from_triad_hardlabel": True},
                           f"{args.ckpt}/best_model.pt")
                print(f"  >>> saved (ler={ler:.5f})", flush=True)

    torch.save({"step": args.steps, "model_state_dict": student.state_dict(),
                "config": config, "distilled_from_triad_hardlabel": True},
               f"{args.ckpt}/final_model.pt")
    print(f"Training done. Best LER: {best:.5f}", flush=True)

if __name__ == "__main__":
    main()
