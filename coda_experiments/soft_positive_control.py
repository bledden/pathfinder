"""Soft-readout POSITIVE CONTROL (§5.15.2 strengthener).

CODA round-2: the ibm_fez soft-readout null is confounded with "the pipeline is inert"
unless we show the pipeline CAN extract soft gain where it provably exists. This is that
control. We sweep the readout SNR and show that the soft decoder's advantage over the hard
decoder GROWS as readout gets noisier (more graded information) — and is ~0 at ibm_fez's
effective SNR (~6.4), which is exactly why §5.15.2 found a null there (the chip is too clean),
not because the pipeline cannot use soft information.

Design (maximally matched): at each SNR, train two identical-architecture NeuralDecoders in
LOCKSTEP on the SAME simulated shots — the only difference is the input representation:
  - soft model: graded soft detectors P(detector fired) in [0,1] (data_soft pipeline)
  - hard model: the same detectors thresholded at 0.5 (the standard hard pipeline)
Both see identical underlying errors + identical IQ draws every step; both use AdamW at the
same LR/budget. Then decode the SAME fresh test shots with each and compare LER (Wilson CI)
+ the paired McNemar test (soft vs hard on identical shots).

Expected: LER_hard - LER_soft increases as SNR decreases; ~0 at SNR~6.4 (ibm_fez anchor).
"""
import sys, os, json, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import torch.nn.functional as F
from model import NeuralDecoder, DecoderConfig
from data_soft import SoftSyndromeDataset, SoftDataConfig

D, R, H = 3, 3, 128
SNRS = [1.5, 2.5, 4.0, 6.4]      # 6.4 = ibm_fez d3 effective SNR (anchor); lower = noisier readout
STEPS = int(os.environ.get('PC_STEPS', 12000))
BS = 256
EVAL_SHOTS = int(os.environ.get('PC_EVAL', 40000))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0, 0.0
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return p, c-h, c+h

def mcnemar(b, c):
    chi = (abs(b-c)-1)**2/(b+c) if (b+c) > 0 else 0.0
    return chi, (math.erfc(math.sqrt(chi/2)) if chi > 0 else 1.0)

def run_snr(snr, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = SoftDataConfig(distance=D, rounds=R, batch_size=BS,
                         snr_median=snr, snr_min=snr*0.97, snr_max=snr*1.03)  # ~fixed SNR
    ds = SoftSyndromeDataset(cfg, rng=np.random.default_rng(seed))
    soft_m = NeuralDecoder(DecoderConfig(distance=D, rounds=R, hidden_dim=H, n_observables=1)).to(dev)
    hard_m = NeuralDecoder(DecoderConfig(distance=D, rounds=R, hidden_dim=H, n_observables=1)).to(dev)
    o_s = torch.optim.AdamW(soft_m.parameters(), lr=1e-3, weight_decay=0.01)
    o_h = torch.optim.AdamW(hard_m.parameters(), lr=1e-3, weight_decay=0.01)
    soft_m.train(); hard_m.train(); t0 = time.time()
    for step in range(STEPS):
        x_soft, y, _ = ds.sample()           # identical batch for both arms
        x_soft = x_soft.to(dev); y = y.to(dev)
        x_hard = (x_soft > 0.5).float()      # ONLY difference: thresholded representation
        for m, o, x in ((soft_m, o_s, x_soft), (hard_m, o_h, x_hard)):
            o.zero_grad(); loss = F.binary_cross_entropy_with_logits(m(x), y)
            loss.backward(); o.step()
        if step % 3000 == 0:
            print(f"  snr={snr} step {step}/{STEPS} ({(step+1)/max(time.time()-t0,1e-3):.1f} it/s)", flush=True)
    # eval on fresh shots, paired
    soft_m.eval(); hard_m.eval()
    se = he = b = c = unc = tot = 0
    with torch.no_grad():
        for _ in range(EVAL_SHOTS // BS):
            x_soft, y, _ = ds.sample(); x_soft = x_soft.to(dev); y = y.to(dev)
            x_hard = (x_soft > 0.5).float()
            ps = (soft_m(x_soft) > 0).float(); ph = (hard_m(x_hard) > 0).float()
            sw = (ps != y).any(1); hw = (ph != y).any(1)
            se += int(sw.sum()); he += int(hw.sum())
            b += int((sw & ~hw).sum()); c += int((~sw & hw).sum())   # b=soft wrong/hard right, c=soft right/hard wrong
            xv = x_soft.cpu().numpy(); nz = xv[xv > 0]
            unc += int(((nz > 0.1) & (nz < 0.9)).sum()); tot += nz.size
    n = (EVAL_SHOTS // BS) * BS
    ls, lsl, lsh = wilson(se, n); lh, lhl, lhh = wilson(he, n)
    chi, mp = mcnemar(b, c)
    return dict(snr=snr, n=n, uncertain_frac=unc/max(tot,1),
                soft_ler=ls, soft_ci=[lsl, lsh], hard_ler=lh, hard_ci=[lhl, lhh],
                gap_hard_minus_soft=lh-ls, mcnemar={'b_soft_wrong_hard_right': b, 'c_soft_right_hard_wrong': c, 'chi2': chi, 'p': mp},
                soft_wins=bool(c > b and mp < 0.05))

def main():
    out = {'note': 'soft-readout positive control: soft vs hard decoder, identical shots, SNR sweep; d=3', 'rows': []}
    for snr in SNRS:
        print(f"=== SNR {snr} ===", flush=True)
        r = run_snr(snr); out['rows'].append(r)
        print(f"  SNR {snr}: uncertain {r['uncertain_frac']*100:.1f}% | soft {r['soft_ler']*100:.3f}% vs hard {r['hard_ler']*100:.3f}% "
              f"| gap(h-s) {r['gap_hard_minus_soft']*100:+.3f}pp | McNemar p={r['mcnemar']['p']:.4g} soft_wins={r['soft_wins']}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), 'soft_positive_control.json'), 'w'), indent=2)
    print("saved soft_positive_control.json")

if __name__ == '__main__': main()
