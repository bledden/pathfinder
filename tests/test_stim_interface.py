import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph


class TestStimInterface:
    def test_make_circuit(self):
        config = SurfaceCodeConfig(distance=3, rounds=3, physical_error_rate=0.001)
        circuit = make_circuit(config)
        assert circuit.num_detectors > 0
        assert circuit.num_observables > 0

    def test_sample_syndromes_shape(self):
        config = SurfaceCodeConfig(distance=3, rounds=3, physical_error_rate=0.001)
        circuit = make_circuit(config)
        det_events, obs_flips = sample_syndromes(circuit, num_shots=100)
        assert det_events.shape[0] == 100
        assert det_events.shape[1] == circuit.num_detectors
        assert obs_flips.shape[0] == 100

    def test_low_noise_few_detections(self):
        config = SurfaceCodeConfig(distance=3, rounds=3, physical_error_rate=0.0001)
        circuit = make_circuit(config)
        det_events, _ = sample_syndromes(circuit, num_shots=1000)
        n_trivial = np.sum(~np.any(det_events, axis=1))
        assert n_trivial > 900

    def test_extract_decoder_graph(self):
        config = SurfaceCodeConfig(distance=3, rounds=3, physical_error_rate=0.001)
        circuit = make_circuit(config)
        graph = extract_decoder_graph(circuit)
        assert graph.n_detectors == circuit.num_detectors
        assert graph.n_observables == circuit.num_observables
        assert len(graph.edges) > 0
        for src, tgt, prob, obs in graph.edges:
            assert 0 < prob < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
