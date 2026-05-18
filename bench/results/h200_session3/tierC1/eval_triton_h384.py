"""M10: Triton kernel numerical equivalence + latency at H=384 (PFWL3S).
Validates that the Triton DirectionalConv3d kernel produces near-identical
predictions to the reference PyTorch implementation at the H=384 hidden
dimension used by PFWL3S — not just H=256 as the existing eval validates."""
import sys, os, json, time
sys.path.insert(0, '/workspace/pathfinder/train')
sys.path.insert(0, '/workspace/pathfinder/bench')
import numpy as np, torch, stim, copy
from model import NeuralDecoder
from triton_directional import swap_to_triton

device = torch.device('cuda')

def make_circuit(d, p):
    return stim.Circuit.generated('surface_code:rotated_memory_z', distance=d, rounds=d,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)

def detcoord_to_tensor(det, grid, det_idx, nd):
    B = det.shape[0]; T, H, W = grid
    t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
    d = torch.from_numpy(det.astype(np.float32))
    for i in range(nd):
        t[:, 0, det_idx[i, 0], det_idx[i, 1], det_idx[i, 2]] = d[:, i]
    return t

def build_det_idx(circuit):
    nd = circuit.num_detectors
    coords = circuit.get_detector_coordinates()
    ac = np.array([coords[i] for i in range(nd)])
    sp, tm = ac[:, :-1], ac[:, -1]
    tu = np.sort(np.unique(tm))
    xu = np.sort(np.unique(sp[:, 0]))
    yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
    grid = (len(tu), len(yu), len(xu))
    tm_m = {v: i for i, v in enumerate(tu)}
    xm = {v: i for i, v in enumerate(xu)}
    ym = {v: i for i, v in enumerate(yu)}
    di = np.zeros((nd, 3), dtype=np.int64)
    for did in range(nd):
        c = coords[did]
        di[did] = [tm_m[c[-1]], ym.get(c[1], 0) if len(c) > 2 else 0, xm[c[0]]]
    return grid, di, nd

CKPT = '/workspace/persist/checkpoints/pathfinder_wide_long_d7/best_model.pt'
print(f'Loading H=384 PFWL3S ckpt: {CKPT}', flush=True)
ck = torch.load(CKPT, weights_only=False, map_location=device)
print(f"  config: {ck['config']}", flush=True)

m_ref = NeuralDecoder(ck['config']).to(device)
m_ref.load_state_dict(ck['model_state_dict'])
m_ref.eval()
m_tri = copy.deepcopy(m_ref)
swap_to_triton(m_tri)
m_tri.eval()
m_ref_h = copy.deepcopy(m_ref).half(); m_ref_h.eval()
m_tri_h = copy.deepcopy(m_tri).half(); m_tri_h.eval()

results = {'H': 384, 'arch': 'PFWL3S', 'ckpt': CKPT, 'fp32': {}, 'fp16': {}}

# ===== Numerical equivalence: 20K shots per noise rate, FP32 + FP16 =====
for p in [0.003, 0.007, 0.015]:
    c = make_circuit(7, p)
    grid, det_idx, nd = build_det_idx(c)
    sampler = c.compile_detector_sampler(seed=42)
    det, obs = sampler.sample(shots=20000, separate_observables=True)
    det = det.astype(np.uint8); obs = obs.astype(np.uint8)
    for prec, (mr, mt, dtype) in [('fp32', (m_ref, m_tri, torch.float32)), ('fp16', (m_ref_h, m_tri_h, torch.float16))]:
        disagreements = 0; max_abs_diff = 0.0; rel_diff_all = []
        pf_ref_wrong = pf_tri_wrong = 0
        for i in range(0, 20000, 1000):
            bd = det[i:i+1000]; bo = obs[i:i+1000]
            syn = detcoord_to_tensor(bd, grid, det_idx, nd).to(device).to(dtype)
            with torch.no_grad():
                lg_ref = mr(syn).float()
                lg_tri = mt(syn).float()
            diff = (lg_ref - lg_tri).abs()
            max_abs_diff = max(max_abs_diff, diff.max().item())
            rel = (diff / (lg_ref.abs() + 1e-8)).mean().item()
            rel_diff_all.append(rel)
            preds_ref = (lg_ref > 0).cpu().numpy().astype(np.uint8)
            preds_tri = (lg_tri > 0).cpu().numpy().astype(np.uint8)
            disagreements += int((preds_ref != preds_tri).any(axis=1).sum())
            pf_ref_wrong += int(np.any(preds_ref != bo, axis=1).sum())
            pf_tri_wrong += int(np.any(preds_tri != bo, axis=1).sum())
        r = {'p': p, 'disagreements_per_20K': disagreements, 'max_abs_logit_diff': max_abs_diff,
             'mean_rel_logit_diff': float(np.mean(rel_diff_all)),
             'ler_ref_pct': pf_ref_wrong / 20000 * 100, 'ler_tri_pct': pf_tri_wrong / 20000 * 100}
        results[prec][f'p{p}'] = r
        print(f"  [{prec}] p={p}: disagreements={disagreements}/20K  max|diff|={max_abs_diff:.6f}  ref_LER={r['ler_ref_pct']:.3f}%  tri_LER={r['ler_tri_pct']:.3f}%", flush=True)

# ===== Latency: H=384 ref vs Triton, B in {1, 64, 1024} =====
print('=== Latency measurement (H=384, d=7) ===', flush=True)
lat = {}
m_tri.half()  # FP16 for latency
for B in [1, 64, 1024]:
    syn = torch.randn(B, 1, 7, 7, 7, device=device, dtype=torch.float16)
    # warmup
    for _ in range(5):
        with torch.no_grad():
            _ = m_ref_h(syn); _ = m_tri(syn)
    torch.cuda.synchronize()
    # measure ref
    t0 = time.perf_counter()
    for _ in range(50):
        with torch.no_grad():
            _ = m_ref_h(syn)
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - t0) / 50 * 1000
    # measure triton
    t0 = time.perf_counter()
    for _ in range(50):
        with torch.no_grad():
            _ = m_tri(syn)
    torch.cuda.synchronize()
    tri_ms = (time.perf_counter() - t0) / 50 * 1000
    lat[f'B{B}'] = {'ref_ms': ref_ms, 'tri_ms': tri_ms, 'ref_us_per_syn': ref_ms * 1000 / B, 'tri_us_per_syn': tri_ms * 1000 / B,
                    'speedup_x': ref_ms / tri_ms if tri_ms > 0 else 0}
    print(f"  B={B:>4}: ref={ref_ms:.3f}ms ({ref_ms*1000/B:.2f} us/syn)  tri={tri_ms:.3f}ms ({tri_ms*1000/B:.2f} us/syn)  speedup={ref_ms/tri_ms:.2f}x", flush=True)
results['latency'] = lat

os.makedirs('/workspace/persist/results', exist_ok=True)
with open('/workspace/persist/results/triton_h384_stability.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved /workspace/persist/results/triton_h384_stability.json', flush=True)
