"""Train PFWL3S on SOFT detectors (calibrated to real ibm_fez kerneled-IQ uncertain-frac).

Same recipe as train_ibm_budget.py but uses SoftSyndromeDataset (float soft detectors from
simulated per-shot IQ at the per-(d,r) calibrated SNR). Makes train==test for the soft-readout
decode of real IBM kerneled data.

Usage:
  python train/train_soft.py --distance 3 --rounds 3 --snr 8.5 --checkpoint_dir <dir> --seed 0
  python train/train_soft.py --distance 5 --rounds 5 --snr 6.5 --checkpoint_dir <dir> --seed 0
"""
import argparse, os, sys, time, math, json
import torch, numpy as np
import torch.nn.functional as F
from pathlib import Path
try:
    from torch.optim import Muon, AdamW
except ImportError:
    from muon import SingleDeviceMuon
    import torch.optim; torch.optim.Muon = SingleDeviceMuon
    from torch.optim import Muon, AdamW
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, 'coda_experiments')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'coda_experiments'))
from model import NeuralDecoder, DecoderConfig
from data_soft import SoftSyndromeDataset, SoftDataConfig


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def build_optimizers(model, muon_lr=0.02, adam_lr=1e-3, weight_decay=0.01):
    mu, ad = [], []
    for _, p in model.named_parameters():
        if p.requires_grad: (mu if p.ndim == 2 else ad).append(p)
    opts = []
    if mu: opts.append(Muon(mu, lr=muon_lr, momentum=0.95, weight_decay=weight_decay))
    if ad: opts.append(AdamW(ad, lr=adam_lr, weight_decay=0.0))
    return opts

class Sched:
    def __init__(s, opts, warm, tot):
        s.o = opts; s.w = warm; s.t = tot
        s.b = [[g['lr'] for g in o.param_groups] for o in opts]
    def step(s, i):
        sc = i/max(s.w,1) if i < s.w else 0.5*(1+math.cos(math.pi*(i-s.w)/max(s.t-s.w,1)))
        for o, bb in zip(s.o, s.b):
            for g, b in zip(o.param_groups, bb): g['lr'] = b*sc

def eval_soft(model, ds, device, n=5000):
    model.train(False); bs = 1000; errs = tot = 0
    for _ in range(n//bs):
        x, y, a = ds.sample(bs); x = x.to(device); y = y.to(device)
        with torch.no_grad():
            errs += ((model(x) > 0).float() != y).any(1).sum().item(); tot += bs
    model.train(True); return errs/max(tot, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--hidden_dim", type=int, default=384)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--snr", type=float, required=True)
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--adam_lr", type=float, default=1e-3)
    ap.add_argument("--eval_interval", type=int, default=4000)
    ap.add_argument("--log_interval", type=int, default=500)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = get_device()
    print(f"Device {dev}  d={a.distance} r={a.rounds} snr={a.snr} seed={a.seed}", flush=True)
    cfg = DecoderConfig(distance=a.distance, rounds=a.rounds, hidden_dim=a.hidden_dim, n_observables=1)
    model = NeuralDecoder(cfg).to(dev)
    opts = build_optimizers(model, a.muon_lr, a.adam_lr); sch = Sched(opts, 1000, a.steps)
    dcfg = SoftDataConfig(distance=a.distance, rounds=a.rounds, batch_size=a.batch_size,
                          snr_median=a.snr, snr_min=a.snr*0.35, snr_max=a.snr*1.4)
    ds = SoftSyndromeDataset(dcfg, rng=np.random.default_rng(a.seed))
    print(f"soft dataset ready grid={ds.grid_shape}", flush=True)
    out = Path(a.checkpoint_dir); out.mkdir(parents=True, exist_ok=True)
    amp = dev.type == 'cuda'; scaler = torch.amp.GradScaler(dev.type) if amp else None
    model.train(True); best = 1.0; t0 = time.time(); log = []
    for step in range(a.steps):
        x, y, al = ds.sample(); x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
        for o in opts: o.zero_grad()
        if amp:
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                loss = F.binary_cross_entropy_with_logits(model(x), y)
            scaler.scale(loss).backward()
            for o in opts: scaler.unscale_(o)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for o in opts: scaler.step(o)
            scaler.update()
        else:
            loss = F.binary_cross_entropy_with_logits(model(x), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for o in opts: o.step()
        sch.step(step)
        if step % a.log_interval == 0:
            sps = (step+1)/max(time.time()-t0, 1e-3)
            print(f"step {step}/{a.steps} loss={loss.item():.4f} {sps:.1f}sps ETA {(a.steps-step)/max(sps,1e-3)/60:.0f}min", flush=True)
        if step > 0 and step % a.eval_interval == 0:
            ler = eval_soft(model, ds, dev); print(f"  >>> soft eval LER={ler:.4f}", flush=True)
            log.append({'step': step, 'ler': ler})
            if ler < best:
                best = ler
                torch.save({'step': step, 'model_state_dict': model.state_dict(), 'config': cfg,
                            'ler': ler, 'snr': a.snr, 'soft': True, 'grid_shape': ds.grid_shape,
                            'train_log': log, 'args': vars(a)}, out/'best_model.pt')
                print(f"  >>> saved best {ler:.4f}", flush=True)
    fin = eval_soft(model, ds, dev, 20000)
    torch.save({'step': a.steps, 'model_state_dict': model.state_dict(), 'config': cfg, 'ler': fin,
                'snr': a.snr, 'soft': True, 'grid_shape': ds.grid_shape, 'train_log': log, 'args': vars(a)},
               out/'final_model.pt')
    json.dump({'best': best, 'final': fin, 'log': log, 'args': vars(a)}, open(out/'train_log.json','w'), indent=2)
    print(f"DONE best={best:.4f} final={fin:.4f} time={(time.time()-t0)/60:.0f}min", flush=True)

if __name__ == "__main__": main()
