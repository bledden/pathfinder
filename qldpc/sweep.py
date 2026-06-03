"""Scoped circuit-level sweep (staged with an early-stop ablation gate).

STAGE 1 (early-stop gate) — matched-param ablation: equiv-640 vs free_random-640 vs classical.
  5 paired seeds, p in {0.005, 0.01}, R=6, per-hyperedge-class LER, training curves.
  GATE: equiv must beat free_random beyond seed noise at BOTH p. If they tie -> STOP, report
  (inductive-bias claim collapses to "smallness"; different paper).
STAGE 2 (param-efficiency curve) — equiv vs free at param counts {640, ~2.5k, ~10k, ~70k}, p in
  {0.005,0.01}, R=6, 3 seeds. Headline figure. Checkpoint: does equiv flatline + free U-curve?
STAGE 3 (R-generalization) — train@6, eval@{4,8,10,12}, equiv vs free_random vs classical, 3 seeds.
  Headline figure 2. Run LAST (most expensive).

Run as: python3 sweep.py stage1   (then stage2, stage3 — separate invocations so each is a checkpoint)
Param scaling for the curve uses T (BP iterations): equiv params = 2*n_orb*T. To hit other equiv
param counts we vary an internal channel multiplier is not available in this min-sum BP, so the
"param-efficiency curve" varies the FREE arm's tying granularity instead: free at {n_orb, 4*n_orb,
16*n_orb, n_edge} random groups vs equiv fixed at n_orb. That sweeps params at fixed architecture.
All LER = neural posterior -> ldpc BP8+OSD0; classical = BP30+OSD-cs10. json-traced.
"""
import json, sys, time, argparse
import numpy as np
import torch, torch.nn as nn
from bb_code import BBCode
from bb_circuit import build_z_memory
from canon_dem import extract, decode_bposd, wilson
from circ_neural_bp import CircNeuralBP
from trains_canon import canon_fgdata, sample_dem, eval_ler
from stepA_canon import drp  # round-of-detector for per-class attribution (R=6 bound; ok for stage1/2)

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = 36


def hyperedge_class(Hs, R):
    rs = set((R if d >= R * N else d // N) for d in Hs)
    if 0 in rs or R in rs:
        return 'boundary' if (rs <= {0} or rs <= {R}) else 'cross'
    return 'bulk'


def eval_ler_perclass(m, ex, s_te, e_te, o_te, R, dev):
    """Aggregate LER + per-class fail attribution (by which class the residual error touches)."""
    m.eval()
    with torch.no_grad():
        Lv = m(s_te.to(dev)).cpu().numpy()
    pe1 = 1.0 / (1.0 + np.exp(np.clip(Lv, -30, 30)))
    from ldpc import BpOsdDecoder
    H = ex['H']; Lo = ex['Lo'].toarray().astype(np.int64)
    dec = BpOsdDecoder(H, error_channel=list(np.clip(pe1.mean(0), 1e-6, 1-1e-6)),
                       max_iter=8, bp_method='ms', osd_method='osd0', osd_order=0)
    detsets = ex['edges']
    cls = np.array([hyperedge_class(Hs, R) for (Hs, o) in detsets])
    s_np = s_te.numpy().astype(np.uint8); o_np = o_te.numpy().astype(np.int64); e_np = e_te.numpy().astype(np.int64)
    Nn = s_np.shape[0]; f = 0; perclass = {'bulk': 0, 'cross': 0, 'boundary': 0}
    for i in range(Nn):
        dec.update_channel_probs(list(np.clip(pe1[i], 1e-6, 1-1e-6)))
        c = dec.decode(s_np[i]).astype(np.int64)
        if not np.array_equal(((Lo @ c) % 2).astype(np.int64), o_np[i]):
            f += 1
            resid = (e_np[i] ^ c) % 2
            fired = np.where(resid == 1)[0]
            if len(fired):
                from collections import Counter
                perclass[Counter(cls[fired]).most_common(1)[0][0]] += 1
    ler = wilson(f, Nn)
    return ler, perclass


def train_arm(fgd, mode, s_tr, e_tr, steps, lr, seed, rand_seed=0, track_curve=False):
    fgd2 = dict(fgd); fgd2['rand_seed'] = rand_seed
    torch.manual_seed(seed)
    m = CircNeuralBP(fgd2, T=8, mode=mode).to(DEV)
    if mode == 'classical':
        return m, [], 0
    opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=lr)
    lf = nn.BCEWithLogitsLoss(); g = torch.Generator().manual_seed(seed); n = s_tr.shape[0]
    curve = []
    for st in range(steps):
        ib = torch.randint(0, n, (1024,), generator=g)
        opt.zero_grad(); loss = lf(-m(s_tr[ib].to(DEV)), e_tr[ib].to(DEV)); loss.backward(); opt.step()
        if track_curve and st % 200 == 0:
            curve.append(round(float(loss), 4))
    return m, curve, m.params_report


def stage1():
    bb = BBCode(); R = 6; seeds = [1, 2, 3, 4, 5]
    out = {'stage': 1, 'R': R, 'desc': 'matched-param early-stop gate: equiv-640 vs free_random-640',
           'seeds': seeds, 'points': {}}
    t0 = time.time()
    for p in [0.005, 0.01]:
        c = build_z_memory(bb, rounds=R, p=p)
        ex = extract(c.detector_error_model(decompose_errors=False))
        fgd = canon_fgdata(ex)
        s_tr, e_tr, o_tr = sample_dem(ex, 40000, 7)
        row = {'n_orb': fgd['n_orb']}
        # classical (once)
        cl = []
        for sd in seeds:
            s_te, e_te, o_te = sample_dem(ex, 10000, 100 + sd)
            (l, lo, hi), f, n = decode_bposd(ex, s_te.numpy().astype(np.uint8), o_te.numpy().astype(np.uint8), 30, 10)
            cl.append(l)
        row['classical'] = {'params': 0, 'ler_mean': round(float(np.mean(cl)), 5),
                            'ler_seeds': [round(x, 5) for x in cl]}
        for mode in ['equiv', 'free_random']:
            lers = []; perclass_acc = {'bulk': 0, 'cross': 0, 'boundary': 0}; nparams = None; curve0 = None
            for sd in seeds:
                m, curve, nparams = train_arm(fgd, mode, s_tr, e_tr, 2000, 0.05, sd,
                                              rand_seed=sd, track_curve=(sd == seeds[0]))
                if sd == seeds[0]: curve0 = curve
                s_te, e_te, o_te = sample_dem(ex, 10000, 100 + sd)
                (l, lo, hi), pc = eval_ler_perclass(m, ex, s_te, e_te, o_te, R, DEV)
                lers.append(l)
                for k in perclass_acc: perclass_acc[k] += pc[k]
            a = np.array(lers)
            row[mode] = {'params': nparams, 'ler_mean': round(float(a.mean()), 5),
                         'ler_std': round(float(a.std()), 5), 'ler_seeds': [round(x, 5) for x in lers],
                         'per_class_fails': perclass_acc, 'train_curve_seed1': curve0}
        # paired margin equiv vs free_random
        eqs = np.array(row['equiv']['ler_seeds']); frs = np.array(row['free_random']['ler_seeds'])
        diff = frs - eqs
        row['equiv_vs_free_random'] = {'paired_mean_margin': round(float(diff.mean()), 5),
                                       'paired_std': round(float(diff.std()), 5),
                                       'equiv_wins_all_seeds': bool((diff > 0).all()),
                                       'margin_over_2std': bool(diff.mean() > 2 * diff.std())}
        out['points'][f'p{p}'] = row
        json.dump(out, open('sweep_stage1.json', 'w'), indent=2)
        print(f"p={p}: classical {row['classical']['ler_mean']:.4f} equiv {row['equiv']['ler_mean']:.4f} "
              f"free_random {row['free_random']['ler_mean']:.4f} | margin {row['equiv_vs_free_random']['paired_mean_margin']:+.4f} "
              f"wins_all {row['equiv_vs_free_random']['equiv_wins_all_seeds']}", flush=True)
    # early-stop gate verdict
    both = all(out['points'][f'p{p}']['equiv_vs_free_random']['equiv_wins_all_seeds'] for p in [0.005, 0.01])
    out['STAGE1_PASS'] = bool(both)
    out['kill_switch'] = 'PROCEED to stage2' if both else 'STOP: equiv ties free_random, inductive-bias claim collapses'
    out['elapsed_min'] = round((time.time() - t0) / 60, 1)
    json.dump(out, open('sweep_stage1.json', 'w'), indent=2)
    print("STAGE1_PASS", out['STAGE1_PASS'], "|", out['kill_switch'], "|", out['elapsed_min'], "min", flush=True)


def stage2():
    bb = BBCode(); R = 6; seeds = [1, 2, 3]
    out = {'stage': 2, 'R': R, 'desc': 'param-efficiency curve: equiv(n_orb) vs free_random at growing groups',
           'seeds': seeds, 'points': {}}
    t0 = time.time()
    for p in [0.005, 0.01]:
        c = build_z_memory(bb, rounds=R, p=p)
        ex = extract(c.detector_error_model(decompose_errors=False))
        fgd = canon_fgdata(ex); n_orb = fgd['n_orb']; n_edge = len(fgd['fg'])
        s_tr, e_tr, o_tr = sample_dem(ex, 40000, 7)
        # group counts spanning n_orb(=40) .. n_edge(=4536): 40, 160, 640, n_edge(free)
        group_counts = [n_orb, 4 * n_orb, 16 * n_orb, n_edge]
        pr = {'p': p, 'n_orb': n_orb, 'n_edge': n_edge, 'equiv': None, 'curve': {}}
        # equiv reference (fixed at n_orb groups by symmetry)
        eq = []
        for sd in seeds:
            m, _, npar = train_arm(fgd, 'equiv', s_tr, e_tr, 2000, 0.05, sd)
            s_te, e_te, o_te = sample_dem(ex, 10000, 100 + sd)
            eq.append(eval_ler(m, ex, s_te, o_te, DEV)[0])
        pr['equiv'] = {'params': npar, 'ler_mean': round(float(np.mean(eq)), 5)}
        for gc in group_counts:
            mode = 'free' if gc == n_edge else 'free_random'
            lers = []; npar = None
            for sd in seeds:
                fgd2 = dict(fgd); fgd2['rand_seed'] = sd
                # for free_random with gc groups, temporarily override n_orb used for grouping:
                fgd2 = dict(fgd2); fgd2['force_groups'] = gc
                m, _, npar = train_arm_groups(fgd2, mode, s_tr, e_tr, 2000, 0.05, sd, gc)
                s_te, e_te, o_te = sample_dem(ex, 10000, 100 + sd)
                lers.append(eval_ler(m, ex, s_te, o_te, DEV)[0])
            pr['curve'][str(gc)] = {'mode': mode, 'params': npar, 'ler_mean': round(float(np.mean(lers)), 5)}
        out['points'][f'p{p}'] = pr
        json.dump(out, open('sweep_stage2.json', 'w'), indent=2)
        print(f"p={p}: equiv({n_orb}grp) {pr['equiv']['ler_mean']:.4f} | curve "
              + " ".join(f"{g}:{pr['curve'][str(g)]['ler_mean']:.4f}" for g in group_counts), flush=True)
    out['elapsed_min'] = round((time.time() - t0) / 60, 1)
    json.dump(out, open('sweep_stage2.json', 'w'), indent=2)
    print("STAGE2 done", out['elapsed_min'], "min", flush=True)


def train_arm_groups(fgd, mode, s_tr, e_tr, steps, lr, seed, gc):
    """free_random with a forced group count gc (overrides n_orb for the random partition)."""
    fgd2 = dict(fgd); fgd2['rand_seed'] = seed
    if mode == 'free_random':
        fgd2 = dict(fgd2); fgd2['n_orb'] = gc  # CircNeuralBP free_random groups into n_orb buckets
    torch.manual_seed(seed)
    m = CircNeuralBP(fgd2, T=8, mode=mode).to(DEV)
    opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=lr)
    lf = nn.BCEWithLogitsLoss(); g = torch.Generator().manual_seed(seed); n = s_tr.shape[0]
    for st in range(steps):
        ib = torch.randint(0, n, (1024,), generator=g)
        opt.zero_grad(); lf(-m(s_tr[ib].to(DEV)), e_tr[ib].to(DEV)).backward(); opt.step()
    return m, [], m.params_report


def stage3():
    bb = BBCode(); seeds = [1, 2, 3]
    out = {'stage': 3, 'desc': 'R-generalization: train@6 eval@{4,8,10,12}', 'seeds': seeds,
           'train_R': 6, 'eval_R': [4, 8, 10, 12], 'p': 0.01, 'arms': {}}
    t0 = time.time(); p = 0.01
    c6 = build_z_memory(bb, rounds=6, p=p); ex6 = extract(c6.detector_error_model(decompose_errors=False))
    fgd6 = canon_fgdata(ex6)
    s_tr, e_tr, o_tr = sample_dem(ex6, 40000, 7)
    # precompute eval DEMs/fgdata at each R
    evex = {}
    for Re in [4, 8, 10, 12]:
        cR = build_z_memory(bb, rounds=Re, p=p); exR = extract(cR.detector_error_model(decompose_errors=False))
        evex[Re] = (exR, canon_fgdata(exR))
    for mode in ['equiv', 'free_random', 'classical']:
        arm = {}
        for sd in seeds:
            m6, _, npar = train_arm(fgd6, mode, s_tr, e_tr, 2000, 0.05, sd, rand_seed=sd)
            sd_w = m6.state_dict()
            for Re in [4, 8, 10, 12]:
                exR, fgdR = evex[Re]
                # transfer: equiv weights are R-independent (orbit-tied), reload into R-sized model
                mR = CircNeuralBP(dict(fgdR, rand_seed=sd), T=8, mode=mode).to(DEV)
                if mode != 'classical':
                    # equiv: wc/wv are (T, n_orb); n_orb may differ across R for free_random but equiv
                    # n_orb is the same group structure -> copy if shapes match, else skip (honest)
                    try:
                        if mR.wc.shape == m6.wc.shape:
                            mR.wc.data.copy_(m6.wc.data); mR.wv.data.copy_(m6.wv.data)
                            transferred = True
                        else:
                            transferred = False
                    except Exception:
                        transferred = False
                else:
                    transferred = True
                s_te, e_te, o_te = sample_dem(exR, 8000, 200 + sd)
                ler = eval_ler(mR, exR, s_te, o_te, DEV)[0]
                arm.setdefault(f'R{Re}', {'lers': [], 'transferred': transferred})
                arm[f'R{Re}']['lers'].append(round(ler, 5))
        for Re in [4, 8, 10, 12]:
            arm[f'R{Re}']['ler_mean'] = round(float(np.mean(arm[f'R{Re}']['lers'])), 5)
        out['arms'][mode] = arm
        json.dump(out, open('sweep_stage3.json', 'w'), indent=2)
        print(mode, {f'R{Re}': out['arms'][mode][f'R{Re}']['ler_mean'] for Re in [4,8,10,12]}, flush=True)
    out['elapsed_min'] = round((time.time() - t0) / 60, 1)
    json.dump(out, open('sweep_stage3.json', 'w'), indent=2)
    print("STAGE3 done", out['elapsed_min'], "min", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['stage1', 'stage2', 'stage3'])
    {'stage1': stage1, 'stage2': stage2, 'stage3': stage3}[ap.parse_args().stage]()
