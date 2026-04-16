"""Mixed-noise training: sample from uniform distribution of noise rates.

Instead of curriculum annealing to a single target, each batch trains at
a randomly sampled noise rate from [p_min, p_max]. This produces a single
model robust across all noise rates — no tradeoffs between low and high noise.
"""
import sys, os, time, math, torch, random
import torch.nn.functional as F
sys.path.insert(0, "train")
from model import NeuralDecoder, DecoderConfig
from data import SyndromeDataset, DataConfig
from muon import SingleDeviceMuonWithAuxAdam

device = torch.device("cuda")
d = 7
config = DecoderConfig(distance=d, rounds=d, hidden_dim=256)
model = NeuralDecoder(config).to(device)
print(f"d={d} H=256 L={config.n_blocks} {sum(p.numel() for p in model.parameters()):,} params", flush=True)

w2d = [p for p in model.parameters() if p.ndim == 2 and p.requires_grad]
other = [p for p in model.parameters() if p.ndim != 2 and p.requires_grad]
opt = SingleDeviceMuonWithAuxAdam([
    dict(params=w2d, lr=0.02, momentum=0.95, use_muon=True),
    dict(params=other, lr=3e-3, betas=(0.9, 0.999), eps=1e-8, use_muon=False),
])

steps = 80000
batch_size = 1024

# Pre-compile samplers for a grid of noise rates
# Log-uniform from 0.001 to 0.02 (20 points)
p_min, p_max = 0.001, 0.02
noise_grid = [p_min * (p_max / p_min) ** (i / 19) for i in range(20)]
noise_grid = sorted(set(round(p, 5) for p in noise_grid))
samplers = {p: SyndromeDataset(DataConfig(d, d, p, batch_size=batch_size)) for p in noise_grid}
print(f"{len(samplers)} samplers (p={noise_grid[0]:.4f} to {noise_grid[-1]:.4f})", flush=True)

# Warmup phase: first 10% at moderate noise only (p=0.003-0.007)
# Then full range for remaining 90%
def get_noise_rate(step):
    if step < steps * 0.1:
        # Warmup: moderate noise
        return random.uniform(0.003, 0.007)
    else:
        # Full range: log-uniform from p_min to p_max
        log_p = random.uniform(math.log(p_min), math.log(p_max))
        return math.exp(log_p)

os.makedirs("train/checkpoints/d7_mixed", exist_ok=True)
model.train()
best_ler = {0.007: 1.0, 0.01: 1.0, 0.015: 1.0}
t0 = time.time()

for step in range(steps):
    p = get_noise_rate(step)
    # Find closest pre-compiled sampler
    ds = samplers[min(noise_grid, key=lambda k: abs(k - p))]
    syn, lab = ds.sample()
    syn, lab = syn.to(device), lab.to(device)

    logits = model(syn)
    loss = F.binary_cross_entropy_with_logits(logits, lab)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    # Cosine LR with warmup
    if step < 1000:
        sc = step / 1000
    else:
        sc = 0.5 * (1 + math.cos(math.pi * (step - 1000) / (steps - 1000)))
    for pg in opt.param_groups:
        pg["lr"] = (0.02 if pg["use_muon"] else 3e-3) * sc

    if step % 5000 == 0:
        el = time.time() - t0
        sps = (step + 1) / max(el, 1)
        print(f"step {step:>6}/{steps} loss={loss.item():.4f} p={p:.5f} {sps:.1f}s/s ETA={int((steps-step)/max(sps,.01)/60)}min", flush=True)

    if step > 0 and step % 5000 == 0:
        model.eval()
        # Eval at multiple noise rates
        for eval_p in [0.005, 0.007, 0.01, 0.015]:
            evds = SyndromeDataset(DataConfig(d, d, eval_p, batch_size=512))
            te, tt = 0, 0
            for _ in range(20):
                se, le = evds.sample(500)
                se, le = se.to(device), le.to(device)
                with torch.no_grad():
                    te += ((model(se) > 0).float() != le).any(dim=1).sum().item()
                    tt += 500
            ler = te / tt
            marker = " <--" if eval_p == 0.015 else ""
            print(f"  p={eval_p}: LER={ler:.6f}{marker}", flush=True)
            if ler < best_ler.get(eval_p, 1.0):
                best_ler[eval_p] = ler
                if eval_p == 0.007:  # Save on primary metric
                    torch.save({"step": step, "model_state_dict": model.state_dict(),
                                "config": config, "ler": ler, "training": "mixed_noise"},
                               "train/checkpoints/d7_mixed/best_model.pt")
                    print(f"  SAVED best p=0.007 (LER={ler:.6f})", flush=True)
        model.train()

# Save final
torch.save({"step": steps, "model_state_dict": model.state_dict(),
            "config": config, "ler": best_ler, "training": "mixed_noise"},
           "train/checkpoints/d7_mixed/final_model.pt")
print(f"\nDONE best_ler={best_ler} time={time.time()-t0:.0f}s", flush=True)
