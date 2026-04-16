"""Decoder latency comparison."""
import sys, os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph
from union_find_py import UnionFindDecoder
from bp_decoder import BPDecoder
import pymatching


def bench_latency(decode_fn, syndromes, n_warmup=5, n_timed=50):
    n = min(n_warmup + n_timed, syndromes.shape[0])
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        decode_fn(syndromes[i])
        t1 = time.perf_counter()
        if i >= n_warmup:
            times.append(t1 - t0)
    return np.median(times) * 1e6  # microseconds


def main():
    print("=== Decoder Latency Comparison (Python, single-threaded) ===")
    print()
    print(f"{'d':>4} {'n_det':>6} {'PyMatch':>12} {'UF':>12} {'BP(20iter)':>12}")
    print("-" * 50)

    for d in [3, 5, 7]:
        config = SurfaceCodeConfig(distance=d, rounds=d, physical_error_rate=0.005)
        circuit = make_circuit(config)
        det_events, _ = sample_syndromes(circuit, num_shots=100)
        graph = extract_decoder_graph(circuit)

        matching = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model()
        )
        pm_us = bench_latency(lambda s: matching.decode(s), det_events)

        uf = UnionFindDecoder(graph)
        uf_us = bench_latency(lambda s: uf.decode(s), det_events)

        bp = BPDecoder(graph, max_iterations=20)
        bp_us = bench_latency(lambda s: bp.decode(s), det_events)

        print(f"{d:>4} {circuit.num_detectors:>6} {pm_us:>10.0f}us {uf_us:>10.0f}us {bp_us:>10.0f}us")


if __name__ == "__main__":
    main()
