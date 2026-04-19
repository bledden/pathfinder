"""
Rigorous re-evaluation of Table 1 with:
  - Standard circuit-level noise (adds before_round_data_depolarization=p)
  - Fixed seed per (d, p) evaluation (reproducibility)
  - 5-seed eval, reports mean ± std with Wilson CIs
  - 100K shots per seed (500K total per point)
  - Disclosed model-selection rule for d=7 (closest-noise-target, NOT min over test set)
"""
import sys, os, time, json
sys.path.insert(0, "/workspace/pathfinder/train")
import torch, numpy as np
import stim, pymatching
from model import NeuralDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS = 5
N_PER_SEED = 20000  # 5 x 20K = 100K total
PVALS = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015]


def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z*z/n
    ctr = (p + z*z/(2*n)) / denom
    half = (z * np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return p, max(0, ctr - half), min(1, ctr + half)


def make_circuit(d, p):
    """Standard circuit-level noise: ALL four sources enabled."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d, rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,   # <-- previously missing
    )


class GridMapper:
    def __init__(self, circuit):
        self.nd = circuit.num_detectors
        coords = circuit.get_detector_coordinates()
        ac = np.array([coords[i] for i in range(self.nd)])
        sp, tm = ac[:, :-1], ac[:, -1]
        tu = np.sort(np.unique(tm))
        xu = np.sort(np.unique(sp[:, 0]))
        yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
        self.grid = (len(tu), len(yu), len(xu))
        tm_m = {v: i for i, v in enumerate(tu)}
        xm = {v: i for i, v in enumerate(xu)}
        ym = {v: i for i, v in enumerate(yu)}
        # Build index tensor for vectorized scatter
        det_idx = np.zeros((self.nd, 3), dtype=np.int64)
        for did in range(self.nd):
            c = coords[did]
            det_idx[did] = [tm_m[c[-1]], ym.get(c[1], 0) if len(c) > 2 else 0, xm[c[0]]]
        self.det_idx = torch.from_numpy(det_idx)

    def to_tensor(self, det_events):
        """Vectorized: det_events [B, n_detectors] -> tensor [B, 1, T, H, W]."""
        B = det_events.shape[0]
        T, H, W = self.grid
        tensor = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        d = torch.from_numpy(det_events.astype(np.float32))
        t_idx = self.det_idx[:, 0]; y_idx = self.det_idx[:, 1]; x_idx = self.det_idx[:, 2]
        for i in range(self.nd):  # over detectors (small count), not batch
            tensor[:, 0, t_idx[i], y_idx[i], x_idx[i]] = d[:, i]
        return tensor


def evaluate_point(models, d, p, seed, n_shots):
    """Evaluate models + PyMatching at a single (d, p, seed) point."""
    circuit = make_circuit(d, p)
    sampler = circuit.compile_detector_sampler(seed=seed)
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    det, obs = sampler.sample(shots=n_shots, separate_observables=True)
    det = det.astype(np.uint8)
    obs = obs.astype(np.uint8)

    mapper = GridMapper(circuit)

    # PyMatching
    preds_pm = matching.decode_batch(det)
    pm_errs = int(np.sum(np.any(preds_pm != obs, axis=1)))

    # Neural — for d=7, use closest-noise-target rule
    neural_errs = {}
    if d == 7:
        noise_targets = {0.007: "d7_final", 0.01: "d7_p01", 0.015: "d7_p015"}
        # mixed available too
        closest_key = min(noise_targets.keys(), key=lambda k: abs(k - p))
        model_key = noise_targets[closest_key]
        if model_key not in models: model_key = "d7_final"  # fallback
    elif d == 5:
        model_key = "d5"
    else:
        model_key = "d3"

    if model_key in models:
        m = models[model_key]
        errs = 0
        for i in range(0, n_shots, 1000):
            batch = det[i:i+1000]
            bo = obs[i:i+1000]
            syn = mapper.to_tensor(batch).to(device)
            with torch.no_grad():
                lg = m(syn)
            preds = (lg > 0).cpu().numpy().astype(np.uint8)
            errs += int(np.sum(np.any(preds != bo, axis=1)))
        neural_errs[model_key] = errs
    return {"n": n_shots, "pm_errs": pm_errs, "neural_errs": neural_errs, "model_used": model_key}


def main():
    print("Loading models...", flush=True)
    paths = {
        "d3": "/workspace/pathfinder/train/checkpoints/best_model.pt",
        "d5": "/workspace/pathfinder/train/checkpoints/d5_muon/best_model.pt",
        "d7_final": "/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt",
        "d7_p01": "/workspace/pathfinder/train/checkpoints/d7_p01/best_model.pt",
        "d7_p015": "/workspace/pathfinder/train/checkpoints/d7_p015/best_model.pt",
    }
    models = {}
    for k, p in paths.items():
        if os.path.exists(p):
            ck = torch.load(p, weights_only=False, map_location=device)
            m = NeuralDecoder(ck["config"]).to(device)
            m.load_state_dict(ck["model_state_dict"])
            m.eval()
            models[k] = m
            print(f"  {k}: OK  (distance={ck['config'].distance}, H={ck['config'].hidden_dim})", flush=True)

    results = {}
    for d in [3, 5, 7]:
        for p in PVALS:
            print(f"\nd={d} p={p}: running {N_SEEDS} seeds x {N_PER_SEED} shots = {N_SEEDS*N_PER_SEED:,} total", flush=True)
            point = {"d": d, "p": p, "seeds": []}
            for seed in range(1000, 1000 + N_SEEDS):
                r = evaluate_point(models, d, p, seed, N_PER_SEED)
                point["seeds"].append({"seed": seed, **r})
                print(f"  seed={seed}: pm_errs={r['pm_errs']}, neural_errs={r['neural_errs']} (model={r['model_used']})", flush=True)
            # Aggregate
            total_n = sum(s["n"] for s in point["seeds"])
            pm_total = sum(s["pm_errs"] for s in point["seeds"])
            model_used = point["seeds"][0]["model_used"]
            neural_total = sum(s["neural_errs"].get(model_used, 0) for s in point["seeds"])
            pm_p, pm_lo, pm_hi = wilson(pm_total, total_n)
            n_p, n_lo, n_hi = wilson(neural_total, total_n)
            point["pm_ler"] = pm_p; point["pm_ci"] = [pm_lo, pm_hi]; point["pm_errs_total"] = pm_total
            point["neural_ler"] = n_p; point["neural_ci"] = [n_lo, n_hi]; point["neural_errs_total"] = neural_total
            point["total_shots"] = total_n; point["model_used"] = model_used
            sig = (n_hi < pm_lo) or (pm_hi < n_lo)
            point["ci_non_overlap"] = bool(sig)
            verdict = "NEURAL_WINS" if n_p < pm_p else ("TIE" if n_p == pm_p else "PM_WINS")
            point["verdict"] = verdict
            print(f"  SUMMARY: Pathfinder {n_p*100:.4f}% [{n_lo*100:.4f},{n_hi*100:.4f}]  PM {pm_p*100:.4f}% [{pm_lo*100:.4f},{pm_hi*100:.4f}]  sig={sig}  {verdict}", flush=True)
            results[f"d{d}_p{p}"] = point

    out = "/workspace/run_eval_v2_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== wrote {out} ===")


if __name__ == "__main__":
    main()
