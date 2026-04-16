import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph
from bp_decoder import BPDecoder


class TestBPDecoder:
    def test_trivial_syndrome(self):
        config = SurfaceCodeConfig(distance=3, rounds=3, physical_error_rate=0.001)
        circuit = make_circuit(config)
        graph = extract_decoder_graph(circuit)
        bp = BPDecoder(graph)
        syndrome = np.zeros(graph.n_detectors, dtype=bool)
        pred, converged = bp.decode(syndrome)
        assert all(not p for p in pred)
        assert converged

    def test_bp_runs(self):
        config = SurfaceCodeConfig(distance=3, rounds=3, physical_error_rate=0.005)
        circuit = make_circuit(config)
        graph = extract_decoder_graph(circuit)
        bp = BPDecoder(graph, max_iterations=20)
        det_events, _ = sample_syndromes(circuit, num_shots=50)
        for i in range(50):
            pred, _ = bp.decode(det_events[i])
            assert len(pred) == graph.n_observables

    def test_bp_accuracy(self):
        config = SurfaceCodeConfig(distance=3, rounds=3, physical_error_rate=0.002)
        circuit = make_circuit(config)
        graph = extract_decoder_graph(circuit)
        bp = BPDecoder(graph, max_iterations=30)
        det_events, obs_flips = sample_syndromes(circuit, num_shots=5000)
        preds, conv_rate = bp.decode_batch(det_events)
        n_errors = np.sum(np.any(preds != obs_flips, axis=1))
        ler = n_errors / 5000
        print(f"BP LER: {ler:.4f}, convergence: {conv_rate:.1%}")
        assert ler < 0.15, f"BP LER {ler} too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
