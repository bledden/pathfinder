"""Logical Error Rate benchmark: Union-Find vs BP vs PyMatching."""
import sys, os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph
from union_find_py import UnionFindDecoder
from bp_decoder import BPDecoder
import pymatching


def benchmark_ler(distances, error_rates, num_shots=10000):
    print(f"{'d':>3} {'p':>8} {'PyMatch':>10} {'UF':>10} {'BP':>10} {'BP_conv':>8}")
    print("-" * 58)

    for d in distances:
        for p in error_rates:
            config = SurfaceCodeConfig(distance=d, rounds=d, physical_error_rate=p)
            circuit = make_circuit(config)

            # Scale shots with problem size to keep runtime manageable
            # (pure Python decoders are slow for large d)
            shots = num_shots if d <= 3 else (5000 if d <= 5 else 1000)
            bp_shots = min(shots, 10000 if d <= 3 else (1000 if d <= 5 else 0))

            det_events, obs_flips = sample_syndromes(circuit, shots)
            graph = extract_decoder_graph(circuit)

            # PyMatching (C++ — fast at all sizes)
            matching = pymatching.Matching.from_detector_error_model(
                circuit.detector_error_model()
            )
            pm_pred = matching.decode_batch(det_events)
            pm_ler = np.sum(np.any(pm_pred != obs_flips, axis=1)) / shots

            # Union-Find (Python — O(n*alpha(n)) per shot)
            uf = UnionFindDecoder(graph)
            uf_pred = uf.decode_batch(det_events)
            uf_ler = np.sum(np.any(uf_pred != obs_flips, axis=1)) / shots

            # BP (Python — O(iter * edges) per shot, skip for d>=7)
            if bp_shots > 0:
                bp = BPDecoder(graph, max_iterations=30)
                bp_pred, bp_conv = bp.decode_batch(det_events[:bp_shots])
                bp_obs = obs_flips[:bp_shots]
                bp_ler = np.sum(np.any(bp_pred != bp_obs, axis=1)) / bp_shots
                print(f"{d:>3} {p:>8.4f} {pm_ler:>10.6f} {uf_ler:>10.6f} {bp_ler:>10.6f} {bp_conv:>8.1%}")
            else:
                print(f"{d:>3} {p:>8.4f} {pm_ler:>10.6f} {uf_ler:>10.6f} {'skip':>10} {'skip':>8}")
        print()


def main():
    print("=== Logical Error Rate Benchmark ===")
    print("Decoders: PyMatching (MWPM), Union-Find, Belief Propagation")
    print("(BP skipped for d>=7 in pure Python — too slow)")
    print()
    benchmark_ler(
        distances=[3, 5, 7],
        error_rates=[0.001, 0.002, 0.005, 0.01],
        num_shots=10000,
    )


if __name__ == "__main__":
    main()
