"""
Ensemble study: narrow-distill + PyMatching.
At each syndrome, run both decoders. Report:
  - neural-alone LER
  - PM-alone LER
  - AGREE-rate (both give same answer)
  - DISAGREE-rate
  - OR-oracle (at least one right) - upper bound on ensemble
  - AND-oracle (both right) - always-safe answer
  - Simple confidence ensemble: pick neural if |logit| > threshold else PM
"""
import sys, os
sys.path.insert(0, "/workspace/pathfinder/train")
import torch, numpy as np
import stim, pymatching
from model import NeuralDecoder
from data import SyndromeDataset, DataConfig
from python.stim_interface import build_rotated_surface_code_circuit


def run_pm_decoder(circuit, syndromes_np, labels_np):
    """Run PyMatching and return predictions (1D bool array)."""
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    preds = []
    for syn in syndromes_np:
        pred = matching.decode(syn.astype(np.uint8))
        preds.append(pred[0])
    return np.array(preds)


def evaluate(model_path, distance, noise_rate, n_shots=20000):
    """Run both decoders on same syndromes. Return dict of statistics."""
    # --- Build circuit + dataset ---
    ds_cfg = DataConfig(distance=distance, rounds=distance, physical_error_rate=noise_rate)
    ds = SyndromeDataset(ds_cfg)
    circuit = ds.circuit

    # --- Load neural model ---
    ck = torch.load(model_path, weights_only=False, map_location="cuda")
    m = NeuralDecoder(ck["config"]).cuda().eval().half()
    m.load_state_dict(ck["model_state_dict"])

    # --- Build PM decoder ---
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    # --- Sample and decode ---
    batch_size = 1000
    neural_errs = 0
    pm_errs = 0
    agree = 0
    disagree = 0
    or_oracle_errs = 0   # wrong only if BOTH wrong
    and_oracle_errs = 0  # wrong if EITHER wrong (both needed to be right)
    # Confidence-based ensemble: pick neural if |logit|>threshold else PM
    thresh = 2.0
    ensemble_errs = 0
    total = 0

    # For Stim, we need the detector-level syndrome and the logical observable
    # SyndromeDataset.sample gives us (syndrome_tensor, label_tensor) but also has the raw stim output
    # Just sample from Stim directly for matched PM input
    sampler = circuit.compile_detector_sampler()

    for _ in range(n_shots // batch_size):
        det_events, obs_flips = sampler.sample(batch_size, separate_observables=True)
        det_events = det_events.astype(np.uint8)  # [B, num_detectors]
        obs_flips = obs_flips.astype(np.uint8)    # [B, num_obs]

        # Neural: need to reshape detector events into 3D syndrome tensor
        # Use the dataset's _reshape function if available, else replicate here
        # SyndromeDataset has grid_shape and _detector_to_grid
        syn_3d = np.zeros((batch_size, 1, distance, distance, distance), dtype=np.float32)
        # Use dataset helper:
        syn_3d_tensor = ds.detectors_to_tensor(torch.from_numpy(det_events)).unsqueeze(1).float()
        x = syn_3d_tensor.cuda().half()

        with torch.no_grad():
            logits = m(x)  # [B, n_obs]
        logits_np = logits.cpu().float().numpy()
        preds_neural = (logits_np > 0).astype(np.uint8)

        # PM
        preds_pm = []
        for syn in det_events:
            pred = matching.decode(syn)
            preds_pm.append(pred)
        preds_pm = np.array(preds_pm)

        # Compare
        for i in range(batch_size):
            neural_right = (preds_neural[i] == obs_flips[i]).all()
            pm_right = (preds_pm[i] == obs_flips[i]).all()
            if not neural_right:
                neural_errs += 1
            if not pm_right:
                pm_errs += 1
            if (preds_neural[i] == preds_pm[i]).all():
                agree += 1
            else:
                disagree += 1
            if not (neural_right or pm_right):
                or_oracle_errs += 1
            if not (neural_right and pm_right):
                and_oracle_errs += 1
            # Confidence ensemble
            confidence = abs(logits_np[i][0])
            if confidence > thresh:
                chosen = preds_neural[i]
            else:
                chosen = preds_pm[i]
            if not (chosen == obs_flips[i]).all():
                ensemble_errs += 1
        total += batch_size

    return {
        "noise_rate": noise_rate,
        "n_shots": total,
        "neural_ler": neural_errs / total,
        "pm_ler": pm_errs / total,
        "agree_pct": 100.0 * agree / total,
        "disagree_pct": 100.0 * disagree / total,
        "or_oracle_ler": or_oracle_errs / total,
        "and_oracle_ler": and_oracle_errs / total,
        "confidence_ensemble_ler": ensemble_errs / total,
    }


if __name__ == "__main__":
    ckpt = "/workspace/pathfinder/train/checkpoints/d7_distill/best_model.pt"
    print(f"Distilled narrow vs PyMatching at d=7, multiple noise rates")
    print(f"{'p':>6} {'neural':>8} {'PM':>8} {'agree%':>8} {'disagree%':>10} {'OR':>8} {'AND':>8} {'conf-ens':>10}")
    for p in [0.003, 0.005, 0.007, 0.010]:
        r = evaluate(ckpt, 7, p)
        print(f"{p:>6.3f} {r['neural_ler']:>8.5f} {r['pm_ler']:>8.5f} {r['agree_pct']:>8.2f} {r['disagree_pct']:>10.2f} {r['or_oracle_ler']:>8.5f} {r['and_oracle_ler']:>8.5f} {r['confidence_ensemble_ler']:>10.5f}")
