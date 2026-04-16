"""Test the C++ Union-Find decoder via pybind11 bindings."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))

from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph
import pydecoder


def graph_to_cpp(decoder_graph):
    """Convert Python DecoderGraph to C++ SyndromeGraph."""
    sg = pydecoder.SyndromeGraph()
    sg.n_detectors = decoder_graph.n_detectors
    sg.n_observables = decoder_graph.n_observables

    for src, tgt, prob, obs_mask in decoder_graph.edges:
        sg.add_edge(src, tgt, prob, obs_mask)

    sg.build_adjacency()
    return sg


class TestCppUnionFind:
    def _make_decoder(self, distance=3, p=0.001):
        config = SurfaceCodeConfig(distance=distance, rounds=distance, physical_error_rate=p)
        circuit = make_circuit(config)
        graph = extract_decoder_graph(circuit)
        cpp_graph = graph_to_cpp(graph)
        return pydecoder.UnionFindDecoder(cpp_graph), config, circuit

    def test_trivial_syndrome(self):
        dec, config, circuit = self._make_decoder()
        det = [False] * circuit.num_detectors
        result = dec.decode(det)
        assert all(not p for p in result.observable_prediction)

    def test_runs_on_real_syndromes(self):
        dec, config, circuit = self._make_decoder()
        det_events, _ = sample_syndromes(circuit, num_shots=100)
        for i in range(100):
            result = dec.decode(list(det_events[i]))
            assert len(result.observable_prediction) == circuit.num_observables

    def test_batch_decode(self):
        dec, config, circuit = self._make_decoder(distance=3, p=0.005)
        det_events, obs_flips = sample_syndromes(circuit, num_shots=1000)
        predictions = dec.decode_batch(det_events)
        assert predictions.shape == obs_flips.shape

        n_errors = np.sum(np.any(predictions != obs_flips, axis=1))
        ler = n_errors / 1000
        print(f"C++ UF LER at d=3, p=0.005: {ler:.4f}")

    def test_low_noise_accuracy(self):
        dec, config, circuit = self._make_decoder(distance=5, p=0.0005)
        det_events, obs_flips = sample_syndromes(circuit, num_shots=5000)
        predictions = dec.decode_batch(det_events)
        n_errors = np.sum(np.any(predictions != obs_flips, axis=1))
        ler = n_errors / 5000
        print(f"C++ UF LER at d=5, p=0.0005: {ler:.4f}")
        assert ler < 0.05, f"LER {ler} too high"

    def test_vs_pymatching(self):
        import pymatching
        config = SurfaceCodeConfig(distance=5, rounds=5, physical_error_rate=0.005)
        circuit = make_circuit(config)
        det_events, obs_flips = sample_syndromes(circuit, num_shots=10000)

        # PyMatching
        matching = pymatching.Matching.from_detector_error_model(circuit.detector_error_model())
        pm_pred = matching.decode_batch(det_events)
        pm_ler = np.sum(np.any(pm_pred != obs_flips, axis=1)) / 10000

        # C++ Union-Find
        graph = extract_decoder_graph(circuit)
        cpp_graph = graph_to_cpp(graph)
        dec = pydecoder.UnionFindDecoder(cpp_graph)
        uf_pred = dec.decode_batch(det_events)
        uf_ler = np.sum(np.any(uf_pred != obs_flips, axis=1)) / 10000

        print(f"PyMatching LER: {pm_ler:.6f}")
        print(f"C++ UF LER:     {uf_ler:.6f}")
        assert uf_ler < max(pm_ler * 30, 0.1), f"UF too inaccurate"

    def test_latency(self):
        """Measure C++ UF decoding latency."""
        import time
        dec, config, circuit = self._make_decoder(distance=5, p=0.005)
        det_events, _ = sample_syndromes(circuit, num_shots=200)

        # Warmup
        for i in range(10):
            dec.decode(list(det_events[i]))

        times = []
        for i in range(10, 110):
            t0 = time.perf_counter()
            dec.decode(list(det_events[i]))
            t1 = time.perf_counter()
            times.append(t1 - t0)

        median_us = np.median(times) * 1e6
        print(f"C++ UF latency at d=5: {median_us:.0f} μs")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
