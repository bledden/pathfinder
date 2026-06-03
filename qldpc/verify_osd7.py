"""Control experiment: does the BP STAGE matter once OSD strength is matched?

Our negative used OSD-order-0 on the neural side but the BP-OSD bar uses OSD-order-7. So the bar's
edge might be the OSD order, not the BP. This isolates it: feed each decoder's posterior P(error)_v
into the SAME ldpc v2 OSD-CS order-7 (max_iter=0 = OSD only, no extra BP), via update_channel_probs.
The ONLY thing differing across {classical, equiv, free} is the BP source. Eval on NOVEL (leak-free)
syndromes only (never in that seed's training set) — the regime where memorization can't help.

CRITICAL BUG HISTORY: the v1 `bposd_decoder` SILENTLY IGNORES per-shot channel_probs (verified:
identical output for priors 0.001/0.3/default). The first run of this script used v1 and produced
classical==equiv==free to 16 digits -> VOID. This version uses v2 `BpOsdDecoder.update_channel_probs`,
verified to change the decode (informative prior recovers the exact error). A hard assertion below
aborts if the three decoders ever come out identical again.

Decision rule: trained ties classical on novel syndromes => conclusion stands; trained
strict-beats classical => story changes. All numbers json.load'd; CIs Wilson.
"""
import os
import json, sys
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)
import numpy as np
import torch, torch.nn as nn
from bb_code import BBCode
from neural_bp import NeuralBP
from ldpc import BpOsdDecoder
from _util import wilson_ci


def gen(bb, p, n, seed):
    rng = np.random.default_rng(seed)
    e = (rng.random((n, bb.n)) < p).astype(np.int64)
    s = ((e @ bb.HZ.astype(np.int64).T) % 2).astype(np.uint8)
    return s, e.astype(np.uint8)


def train(model, bb, p, n_train, steps, seed):
    s, e = gen(bb, p, n_train, 1000 * seed + 1)
    ts = torch.tensor(s, dtype=torch.float32); te = torch.tensor(e, dtype=torch.float32)
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=0.02)
    lf = nn.BCEWithLogitsLoss(); g = torch.Generator().manual_seed(seed)
    model.train()
    for st in range(steps):
        idx = torch.randint(0, n_train, (256,), generator=g)
        opt.zero_grad(); Lv = model(ts[idx], p); loss = lf(-Lv, te[idx])
        loss.backward(); opt.step()
    return model, set(map(tuple, s.tolist()))


def posterior(model, s_sub, p):
    model.eval()
    with torch.no_grad():
        Lv = model(torch.tensor(s_sub, dtype=torch.float32), p).cpu().numpy()
    return np.clip(1.0 / (1.0 + np.exp(Lv)), 1e-6, 1 - 1e-6)   # P(error)=sigmoid(-LLR)


def osd7_eval(bb, perr, s_sub, e_sub, dec):
    L = bb.logicals_Z().astype(np.int64); H = bb.HZ.astype(np.int64)
    N, k = s_sub.shape[0], L.shape[0]
    fl = 0; sm = 0
    preds = np.zeros((N, bb.n), dtype=np.int64)
    for i in range(N):
        dec.update_channel_probs(list(perr[i]))
        c = dec.decode(s_sub[i].astype(np.uint8)).astype(np.int64)
        preds[i] = c
        if (((c @ H.T) % 2) == s_sub[i]).all():
            sm += 1
        r = (e_sub[i].astype(np.int64) ^ c) % 2
        fl += int(((L @ r) % 2).sum())
    pl, lo, hi = wilson_ci(fl, N * k)
    return dict(per_logical=pl, ci=[lo, hi], synd_match=round(sm / N, 4), n=N), preds


def main():
    bb = BBCode(); p = 0.03; T = 12; n_train = 20000; steps = 4000; seeds = [1, 2, 3]
    N_TEST = 20000
    s_te, e_te = gen(bb, p, N_TEST, 99999)
    dec = BpOsdDecoder(bb.HZ.astype(np.uint8), error_channel=list(np.full(bb.n, p)),
                       max_iter=0, bp_method='ms', osd_method='osd_cs', osd_order=7)
    out = {'p': p, 'T': T, 'n_train': n_train, 'steps': steps, 'seeds': seeds, 'n_test': N_TEST,
           'osd_order': 7, 'eval': 'leak-free (novel syndromes only)',
           'ldpc_api': 'v2 BpOsdDecoder.update_channel_probs', 'per_seed': {}}

    agg = {'classical': [], 'equiv': [], 'free': []}
    for sd in seeds:
        eqm, trsynds = train(NeuralBP(bb, T=T, mode='equiv'), bb, p, n_train, steps, sd)
        frm, _ = train(NeuralBP(bb, T=T, mode='free'), bb, p, n_train, steps, sd)
        clm = NeuralBP(bb, T=T, mode='classical')
        mask = np.array([tuple(row) not in trsynds for row in s_te.tolist()])
        s_n, e_n = s_te[mask], e_te[mask]
        row = {'frac_novel': round(float(mask.mean()), 4), 'n_novel': int(mask.sum())}
        preds = {}
        for name, m in [('classical', clm), ('equiv', eqm), ('free', frm)]:
            r, pr = osd7_eval(bb, posterior(m, s_n, p), s_n, e_n, dec)
            row[name] = r; agg[name].append(r['per_logical']); preds[name] = pr
        # HARD SANITY: the three decoders use different posteriors; if their predictions are
        # bit-identical, update_channel_probs isn't taking effect -> abort (the v1 bug).
        ident_ce = bool((preds['classical'] == preds['equiv']).all())
        ident_cf = bool((preds['classical'] == preds['free']).all())
        row['sanity_decoders_differ'] = not (ident_ce and ident_cf)
        if ident_ce and ident_cf:
            out['ABORTED'] = f'seed {sd}: classical==equiv==free predictions identical — channel_probs ignored'
            json.dump(out, open(os.path.join(_OUT, 'verify_osd7.json'), 'w'), indent=2)
            raise SystemExit("SANITY FAIL: " + out['ABORTED'])
        out['per_seed'][str(sd)] = row
        json.dump(out, open(os.path.join(_OUT, 'verify_osd7.json'), 'w'), indent=2)
        print(f"seed {sd}: classical {row['classical']['per_logical']:.4f} | "
              f"equiv {row['equiv']['per_logical']:.4f} | free {row['free']['per_logical']:.4f} "
              f"(novel {row['n_novel']}, decoders_differ {row['sanity_decoders_differ']})")

    out['summary'] = {}
    for name in ['classical', 'equiv', 'free']:
        a = np.array(agg[name])
        out['summary'][name] = dict(mean=round(float(a.mean()), 5), std=round(float(a.std()), 5),
                                    seeds=[round(x, 5) for x in agg[name]])
    c = np.array(agg['classical'])
    for name in ['equiv', 'free']:
        a = np.array(agg[name]); diff = a.mean() - c.mean()
        pooled = (a.std()**2 + c.std()**2) ** 0.5
        out['summary'][f'{name}_minus_classical'] = dict(
            delta=round(float(diff), 5), sigma=round(float(diff / pooled), 2) if pooled else None,
            trained_strict_beats_classical=bool(diff < 0 and abs(diff) > 2 * pooled))
    json.dump(out, open(os.path.join(_OUT, 'verify_osd7.json'), 'w'), indent=2)
    print("SUMMARY classical %.4f | equiv %.4f | free %.4f" % (
        out['summary']['classical']['mean'], out['summary']['equiv']['mean'], out['summary']['free']['mean']))
    print("WROTE verify_osd7.json")


if __name__ == '__main__':
    main()
