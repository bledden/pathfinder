"""100K eval: mixed-noise model + ensemble with p01 and p007 models."""
import sys, os, torch, numpy as np
sys.path.insert(0, "train")
sys.path.insert(0, "python")
sys.path.insert(0, "build")
from model import NeuralDecoder
from data import SyndromeDataset, DataConfig
from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph
import pymatching

n = 100000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device), flush=True)

# Load all three d=7 models ON GPU
models = {}
for name, path in [
    ("mixed", "train/checkpoints/d7_mixed/best_model.pt"),
    ("p01", "train/checkpoints/d7_p01/best_model.pt"),
    ("p007", "train/checkpoints/d7_final/best_model.pt"),
]:
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    m = NeuralDecoder(ckpt["config"])
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    m = m.to(device)
    models[name] = m
    print("Loaded " + name + " (step=" + str(ckpt.get("step", "?")) + ")", flush=True)

print("", flush=True)
print("=== 100K DEFINITIVE EVAL (d=7) ===", flush=True)
header = "       p      Mixed        p01       p007   Ensemble         PM  Best/PM"
print(header, flush=True)
print("-" * len(header), flush=True)

for p in [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015]:
    ds = SyndromeDataset(DataConfig(7, 7, p))

    errs = {"mixed": 0, "p01": 0, "p007": 0, "ens": 0}
    total = 0

    for _ in range(n // 1000):
        syn, lab = ds.sample(1000)
        syn = syn.to(device)
        lab = lab.to(device)
        with torch.no_grad():
            logits = {name: m(syn) for name, m in models.items()}
            logits_ens = sum(logits.values()) / len(logits)

        for name in ["mixed", "p01", "p007"]:
            errs[name] += ((logits[name] > 0).float() != lab).any(dim=1).sum().item()
        errs["ens"] += ((logits_ens > 0).float() != lab).any(dim=1).sum().item()
        total += 1000

    lers = {k: v / total for k, v in errs.items()}

    # PyMatching
    sc = SurfaceCodeConfig(7, 7, p)
    circ = make_circuit(sc)
    det, obs = sample_syndromes(circ, n)
    dem = circ.detector_error_model(decompose_errors=True)
    pm = pymatching.Matching.from_detector_error_model(dem)
    pm_pred = pm.decode_batch(det)
    pm_ler = np.sum(np.any(pm_pred != obs, axis=1)) / n

    best = min(lers.values())
    best_name = min(lers, key=lers.get)
    ratio_str = "{:.2f}x".format(best / pm_ler) if pm_ler > 0 else "inf"

    print("  {:.4f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>7} [{}]".format(
        p, lers["mixed"], lers["p01"], lers["p007"], lers["ens"],
        pm_ler, ratio_str, best_name), flush=True)

print("", flush=True)
print("=== COMPLETE ===", flush=True)
