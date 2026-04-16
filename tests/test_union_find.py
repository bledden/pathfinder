import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from stim_interface import SurfaceCodeConfig, make_circuit, sample_syndromes, extract_decoder_graph
from union_find_py import UnionFindDecoder


class TestUnionFind:
    def _make_decoder(self, distance=3, p=0.001):
        config = SurfaceCodeConfig(distance=distance, rounds=distance, physical_error_rate=p)
        circuit = make_circuit(config)
        graph = extract_decoder_graph(circuit)
        return UnionFindDecoder(graph), config, circuit

    def test_trivial_syndrome(self):
        """No defects -> no correction needed."""
        dec, config, circuit = self._make_decoder()
        det = np.zeros(circuit.num_detectors, dtype=bool)
        pred, converged = dec.decode(det)
        assert all(not p for p in pred)
        assert converged

    def test_decoder_runs_on_real_syndromes(self):
        """Decoder should run without crashing."""
        dec, config, circuit = self._make_decoder()
        det_events, obs_flips = sample_syndromes(circuit, num_shots=100)
        for i in range(100):
            pred, converged = dec.decode(det_events[i])
            assert len(pred) == circuit.num_observables

    def test_low_noise_high_accuracy(self):
        """At low noise, decoder should have low logical error rate."""
        dec, config, circuit = self._make_decoder(distance=5, p=0.0005)
        det_events, obs_flips = sample_syndromes(circuit, num_shots=10000)
        preds = dec.decode_batch(det_events)
        n_errors = np.sum(np.any(preds != obs_flips, axis=1))
        ler = n_errors / 10000
        print(f"UF LER at d=5, p=0.0005: {ler:.6f}")
        # UF decoder with simplified peeling: expect < 2% at this noise level
        assert ler < 0.02, f"LER {ler} too high for low noise"

    def test_decoder_better_than_no_correction(self):
        """Decoder should outperform doing nothing (no correction)."""
        for d in [3, 5]:
            config = SurfaceCodeConfig(distance=d, rounds=d, physical_error_rate=0.002)
            circuit = make_circuit(config)
            det_events, obs_flips = sample_syndromes(circuit, num_shots=10000)
            graph = extract_decoder_graph(circuit)
            dec = UnionFindDecoder(graph)
            preds = dec.decode_batch(det_events)

            # "No correction" = always predict False (no logical flip)
            no_corr_errors = np.sum(np.any(obs_flips, axis=1))
            uf_errors = np.sum(np.any(preds != obs_flips, axis=1))
            print(f"d={d}: no_corr={no_corr_errors/10000:.4f}, UF={uf_errors/10000:.4f}")
            assert uf_errors <= no_corr_errors, (
                f"UF should beat no-correction: UF={uf_errors}, none={no_corr_errors}"
            )

    def test_accuracy_scaling_with_noise(self):
        """LER should decrease when physical error rate decreases."""
        lers = {}
        for p in [0.005, 0.001]:
            dec, config, circuit = self._make_decoder(distance=3, p=p)
            det_events, obs_flips = sample_syndromes(circuit, num_shots=10000)
            preds = dec.decode_batch(det_events)
            n_errors = np.sum(np.any(preds != obs_flips, axis=1))
            lers[p] = n_errors / 10000
            print(f"UF LER at d=3, p={p}: {lers[p]:.6f}")
        assert lers[0.001] < lers[0.005], (
            f"Lower noise should give lower LER: p=0.001 -> {lers[0.001]}, p=0.005 -> {lers[0.005]}"
        )


class TestUnionFindVsPyMatching:
    def test_compare_accuracy(self):
        """Union-Find should be within reasonable range of PyMatching."""
        import pymatching

        config = SurfaceCodeConfig(distance=5, rounds=5, physical_error_rate=0.005)
        circuit = make_circuit(config)
        det_events, obs_flips = sample_syndromes(circuit, num_shots=20000)

        # PyMatching
        matching = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model()
        )
        pm_pred = matching.decode_batch(det_events)
        pm_errors = np.sum(np.any(pm_pred != obs_flips, axis=1))
        pm_ler = pm_errors / 20000

        # Our Union-Find
        graph = extract_decoder_graph(circuit)
        uf = UnionFindDecoder(graph)
        uf_pred = uf.decode_batch(det_events)
        uf_errors = np.sum(np.any(uf_pred != obs_flips, axis=1))
        uf_ler = uf_errors / 20000

        print(f"PyMatching LER: {pm_ler:.6f}")
        print(f"Union-Find LER: {uf_ler:.6f}")
        if pm_ler > 0:
            print(f"Ratio: {uf_ler / pm_ler:.1f}x")

        # UF should be within 20x of MWPM (generous bound for unweighted version)
        assert uf_ler < max(pm_ler * 20, 0.10), f"UF too inaccurate: {uf_ler} vs PM {pm_ler}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
