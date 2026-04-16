"""Evaluate trained neural decoder against Union-Find and PyMatching.

Usage:
    python train/evaluate.py --checkpoint train/checkpoints/best_model.pt
"""
import argparse
import sys
import os
import time
import torch
import numpy as np
import pymatching
import stim

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))

from model import NeuralDecoder
from data import SyndromeDataset, DataConfig
from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph

try:
    import pydecoder
    HAS_CPP = True
except ImportError:
    HAS_CPP = False


def graph_to_cpp(g):
    sg = pydecoder.SyndromeGraph()
    sg.n_detectors = g.n_detectors
    sg.n_observables = g.n_observables
    for src, tgt, prob, obs in g.edges:
        sg.add_edge(src, tgt, prob, obs)
    sg.build_adjacency()
    return sg


def eval_neural(model, dataset, device, n_shots):
    model.eval()
    errors = 0
    done = 0
    bs = min(1000, n_shots)
    t_total = 0.0
    for _ in range(n_shots // bs):
        syn, lab = dataset.sample(bs)
        syn, lab = syn.to(device), lab.to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = (model(syn) > 0).float()
        t_total += time.perf_counter() - t0
        errors += (pred != lab).any(dim=1).sum().item()
        done += bs
    return errors / done, t_total / done * 1e6


def eval_pymatching(circuit, det, obs):
    m = pymatching.Matching.from_detector_error_model(circuit.detector_error_model())
    t0 = time.perf_counter()
    pred = m.decode_batch(det)
    t1 = time.perf_counter()
    err = np.sum(np.any(pred != obs, axis=1))
    return err / len(det), (t1 - t0) / len(det) * 1e6


def eval_uf(graph, det, obs):
    if not HAS_CPP:
        return None, None
    sg = graph_to_cpp(graph)
    uf = pydecoder.UnionFindDecoder(sg)
    t0 = time.perf_counter()
    pred = uf.decode_batch(det)
    t1 = time.perf_counter()
    err = np.sum(np.any(pred != obs, axis=1))
    return err / len(det), (t1 - t0) / len(det) * 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_shots", type=int, default=50000)
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']
    model = NeuralDecoder(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model: d={config.distance}, H={config.hidden_dim}, L={config.n_blocks}")
    print(f"Trained LER: {ckpt.get('ler', 'N/A')}")
    print()

    uf_hdr = "C++ UF" if HAS_CPP else ""
    print(f"{'p':>8} {'Neural':>10} {'N_us':>8} {'PyMatch':>10} {'PM_us':>8}", end="")
    if HAS_CPP:
        print(f" {'UF':>10} {'UF_us':>8}", end="")
    print()
    print("-" * (55 + (20 if HAS_CPP else 0)))

    for p in [0.001, 0.002, 0.005, 0.007, 0.01]:
        # Neural
        ds = SyndromeDataset(DataConfig(config.distance, config.rounds, p))
        n_ler, n_us = eval_neural(model, ds, device, args.n_shots)

        # PyMatching + UF
        sc = SurfaceCodeConfig(config.distance, config.rounds, p)
        circ = make_circuit(sc)
        det, obs = sample_syndromes(circ, args.n_shots)
        graph = extract_decoder_graph(circ)
        pm_ler, pm_us = eval_pymatching(circ, det, obs)
        uf_ler, uf_us = eval_uf(graph, det, obs)

        print(f"{p:>8.4f} {n_ler:>10.6f} {n_us:>7.0f} {pm_ler:>10.6f} {pm_us:>7.0f}", end="")
        if HAS_CPP and uf_ler is not None:
            print(f" {uf_ler:>10.6f} {uf_us:>7.0f}", end="")
        print()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
