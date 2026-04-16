"""d=7 targeted at p=0.015 — the last holdout."""
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
print("d=7 H=256 target p=0.015 — " + str(sum(p.numel() for p in model.parameters())) + " params", flush=True)

w2d = [p for p in model.parameters() if p.ndim == 2 and p.requires_grad]
other = [p for p in model.parameters() if p.ndim != 2 and p.requires_grad]
opt = SingleDeviceMuonWithAuxAdam([
    dict(params=w2d, lr=0.02, momentum=0.95, use_muon=True),
    dict(params=other, lr=3e-3, betas=(0.9, 0.999), eps=1e-8, use_muon=False),
])

steps = 80000
batch_size = 1024
target_p = 0.015

# Compressed curriculum ramping to p=0.015
class Curriculum:
    def __init__(s, t, n):
        s.t, s.n = t, n
    def get_rate(s, step):
        f = step / max(s.n, 1)
        if f < 0.1: return s.t * 0.3
        elif f < 0.4: return s.t * (0.3 + 0.4 * (f - 0.1) / 0.3)
        else: return s.t * (0.7 + 0.3 * (f - 0.4) / 0.6)

cur = Curriculum(target_p, steps)
rates = sorted(set(round(cur.get_rate(s), 4) for s in range(0, steps, steps//50)) | {target_p})
samplers = {p: SyndromeDataset(DataConfig(d, d, max(p, 1e-6), batch_size=batch_size)) for p in rates}
print(str(len(samplers)) + " samplers ready", flush=True)

os.makedirs("train/checkpoints/d7_p015", exist_ok=True)
model.train()
best_ler = 1.0
t0 = time.time()

for step in range(steps):
    p = cur.get_rate(step)
    ds = samplers[min(samplers.keys(), key=lambda k: abs(k - p))]
    syn, lab = ds.sample()
    syn, lab = syn.to(device), lab.to(device)
    logits = model(syn)
    loss = F.binary_cross_entropy_with_logits(logits, lab)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    sc = (step / 1000 if step < 1000 else 0.5 * (1 + math.cos(math.pi * (step - 1000) / (steps - 1000))))
    for pg in opt.param_groups:
        pg["lr"] = (0.02 if pg["use_muon"] else 3e-3) * sc

    if step % 5000 == 0:
        el = time.time() - t0
        sps = (step + 1) / max(el, 1)
        print("step {:>6}/{} loss={:.4f} p={:.5f} {:.1f}s/s ETA={}min".format(
            step, steps, loss.item(), p, sps, int((steps-step)/max(sps,.01)/60)), flush=True)

    if step > 0 and step % 5000 == 0:
        model.eval()
        evds = SyndromeDataset(DataConfig(d, d, target_p, batch_size=512))
        te, tt = 0, 0
        for _ in range(20):
            se, le = evds.sample(500)
            se, le = se.to(device), le.to(device)
            with torch.no_grad():
                te += ((model(se) > 0).float() != le).any(dim=1).sum().item()
                tt += 500
        ler = te / tt
        print("  EVAL p=0.015: LER={:.6f}".format(ler), flush=True)
        if ler < best_ler:
            best_ler = ler
            torch.save({"step": step, "model_state_dict": model.state_dict(),
                        "config": config, "ler": ler},
                       "train/checkpoints/d7_p015/best_model.pt")
            print("  SAVED (LER={:.6f})".format(ler), flush=True)
        model.train()

print("DONE best={:.6f} time={:.0f}s".format(best_ler, time.time()-t0), flush=True)
