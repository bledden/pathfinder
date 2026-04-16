"""Comprehensive evaluation for paper write-up.

Produces:
1. LER vs p curves at d=3,5,7 (8 noise rates, 100K shots)
2. Error suppression scaling (LER vs d at fixed p)
3. Comparison: Neural vs PyMatching vs PyMatching-corr vs Union-Find
4. Statistical error bars (Wilson score 95% CI)
"""
import sys, os, torch, numpy as np, time, json
sys.path.insert(0, "train")
sys.path.insert(0, "python")
sys.path.insert(0, "build")
from model import NeuralDecoder
from data import SyndromeDataset, DataConfig
from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph
import pymatching, pydecoder


def g2c(g):
    sg = pydecoder.SyndromeGraph()
    sg.n_detectors = g.n_detectors
    sg.n_observables = g.n_observables
    for s, t, p, o in g.edges:
        sg.add_edge(s, t, p, o)
    sg.build_adjacency()
    return sg


def wilson_ci(n_errors, n_total, z=1.96):
    """Wilson score 95% confidence interval."""
    if n_total == 0:
        return 0.0, 0.0
    p = n_errors / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return center, margin


# Load models
models = {}
model_info = {}
checkpoints = {
    3: "train/checkpoints/best_model.pt",
    5: "train/checkpoints/d5_muon/best_model.pt",
    7: "train/checkpoints/d7_p01/best_model.pt",
}

for d, path in checkpoints.items():
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    config = ckpt["config"]
    model = NeuralDecoder(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    models[d] = model
    n_params = sum(p.numel() for p in model.parameters())
    model_info[d] = {
        "step": ckpt.get("step", "?"),
        "params": n_params,
        "hidden_dim": config.hidden_dim,
        "n_blocks": config.n_blocks,
    }
    print(f"Loaded d={d}: H={config.hidden_dim} L={config.n_blocks} "
          f"params={n_params:,} step={ckpt.get('step', '?')}", flush=True)

noise_rates = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015]
distances = [3, 5, 7]
n_shots = 100000

results = {}
all_start = time.time()

for d in distances:
    model = models[d]
    print(f"\n" + "=" * 75, flush=True)
    print(f"d={d} EVALUATION ({n_shots:,} shots per point)", flush=True)
    print("=" * 75, flush=True)
    hdr = f"{'p':>8} {'Neural':>10} {'+-95%':>7} {'PM':>10} {'+-95%':>7} {'PM_corr':>10} {'UF':>10} {'N/PM':>6}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for p in noise_rates:
        t0 = time.time()

        # Neural decoder
        ds = SyndromeDataset(DataConfig(d, d, p))
        n_err, n_tot = 0, 0
        for _ in range(n_shots // 1000):
            syn, lab = ds.sample(1000)
            with torch.no_grad():
                pred = (model(syn) > 0).float()
            n_err += (pred != lab).any(dim=1).sum().item()
            n_tot += 1000
        n_ler, n_ci = wilson_ci(n_err, n_tot)

        # PyMatching (uncorrelated)
        sc = SurfaceCodeConfig(d, d, p)
        circ = make_circuit(sc)
        det, obs = sample_syndromes(circ, n_shots)
        dem = circ.detector_error_model(decompose_errors=True)

        pm = pymatching.Matching.from_detector_error_model(dem)
        pm_pred = pm.decode_batch(det)
        pm_errs = int(np.sum(np.any(pm_pred != obs, axis=1)))
        pm_ler, pm_ci = wilson_ci(pm_errs, n_shots)

        # PyMatching (correlated)
        pm_c = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        pmc_pred = pm_c.decode_batch(det)
        pmc_errs = int(np.sum(np.any(pmc_pred != obs, axis=1)))
        pmc_ler, _ = wilson_ci(pmc_errs, n_shots)

        # Union-Find
        graph = extract_decoder_graph(circ)
        uf_pred = pydecoder.UnionFindDecoder(g2c(graph)).decode_batch(det)
        uf_errs = int(np.sum(np.any(uf_pred != obs, axis=1)))
        uf_ler, _ = wilson_ci(uf_errs, n_shots)

        ratio = f"{n_ler/pm_ler:.2f}" if pm_ler > 1e-8 else "inf"
        dt = time.time() - t0
        print(f"  {p:.4f} {n_ler:>10.6f} {n_ci:>6.5f} {pm_ler:>10.6f} {pm_ci:>6.5f} "
              f"{pmc_ler:>10.6f} {uf_ler:>10.6f} {ratio:>5}x ({dt:.0f}s)", flush=True)

        results[f"d{d}_p{p}"] = {
            "d": d, "p": p, "n_shots": n_shots,
            "neural_ler": n_ler, "neural_ci": n_ci, "neural_errors": n_err,
            "pm_ler": pm_ler, "pm_ci": pm_ci, "pm_errors": pm_errs,
            "pmc_ler": pmc_ler, "pmc_errors": pmc_errs,
            "uf_ler": uf_ler, "uf_errors": uf_errs,
        }

# Save raw results
os.makedirs("bench/results", exist_ok=True)
with open("bench/results/comprehensive_eval.json", "w") as f:
    json.dump({"results": results, "model_info": model_info,
               "n_shots": n_shots, "noise_rates": noise_rates,
               "distances": distances}, f, indent=2)
print("\nResults saved to bench/results/comprehensive_eval.json", flush=True)

# Error suppression scaling analysis
print(f"\n" + "=" * 70, flush=True)
print("ERROR SUPPRESSION SCALING (LER vs distance at fixed p)", flush=True)
print("=" * 70, flush=True)
for p in [0.001, 0.003, 0.005, 0.007, 0.01]:
    print(f"\np = {p}:", flush=True)
    print(f"  {'d':>3} {'Neural':>10} {'PM':>10} {'N/PM':>7} {'UF':>10}", flush=True)
    for d in distances:
        key = f"d{d}_p{p}"
        if key in results:
            r = results[key]
            ratio = f"{r['neural_ler']/r['pm_ler']:.2f}x" if r['pm_ler'] > 1e-8 else "n/a"
            print(f"  {d:>3} {r['neural_ler']:>10.6f} {r['pm_ler']:>10.6f} {ratio:>7} {r['uf_ler']:>10.6f}", flush=True)

# Lambda (error suppression ratio) analysis
print(f"\n" + "=" * 70, flush=True)
print("ERROR SUPPRESSION RATIO (Lambda = LER(d) / LER(d+2))", flush=True)
print("=" * 70, flush=True)
print(f"{'p':>8} {'Neural 3->5':>12} {'Neural 5->7':>12} {'PM 3->5':>12} {'PM 5->7':>12}", flush=True)
for p in [0.001, 0.003, 0.005, 0.007, 0.01]:
    vals = {}
    for d in distances:
        key = f"d{d}_p{p}"
        if key in results:
            vals[d] = results[key]

    def ratio_str(d1, d2, decoder):
        if d1 in vals and d2 in vals:
            v1 = vals[d1][decoder]
            v2 = vals[d2][decoder]
            if v2 > 1e-8:
                return f"{v1/v2:.2f}x"
        return "n/a"

    n35 = ratio_str(3, 5, "neural_ler")
    n57 = ratio_str(5, 7, "neural_ler")
    p35 = ratio_str(3, 5, "pm_ler")
    p57 = ratio_str(5, 7, "pm_ler")
    print(f"  {p:.4f} {n35:>12} {n57:>12} {p35:>12} {p57:>12}", flush=True)

total_time = time.time() - all_start
print(f"\nTotal evaluation time: {total_time:.0f}s ({total_time/60:.0f} min)", flush=True)
print("\n=== COMPREHENSIVE EVALUATION COMPLETE ===", flush=True)
