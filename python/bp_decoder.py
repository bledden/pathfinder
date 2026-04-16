"""Belief Propagation decoder for quantum codes.

Pure Python/numpy prototype. Min-sum message passing.
"""
import numpy as np
from typing import Tuple
from stim_interface import DecoderGraph


class BPDecoder:
    def __init__(self, decoder_graph: DecoderGraph, max_iterations: int = 50):
        self.n_checks = decoder_graph.n_detectors
        self.n_vars = len(decoder_graph.edges)
        self.n_obs = decoder_graph.n_observables
        self.max_iter = max_iterations

        # Build H matrix and channel LLR
        self.H = np.zeros((self.n_checks, self.n_vars), dtype=np.int8)
        self.channel_llr = np.zeros(self.n_vars)
        self.obs_matrix = np.zeros((self.n_obs, self.n_vars), dtype=np.int8)

        for j, (src, tgt, prob, obs_mask) in enumerate(decoder_graph.edges):
            if src >= 0:
                self.H[src, j] = 1
            if tgt >= 0:
                self.H[tgt, j] = 1
            self.channel_llr[j] = np.log((1 - prob) / max(prob, 1e-15))
            for o in obs_mask:
                self.obs_matrix[o, j] = 1

        # CSR/CSC for fast iteration
        self.check_to_vars = [np.where(self.H[c] > 0)[0] for c in range(self.n_checks)]
        self.var_to_checks = [np.where(self.H[:, v] > 0)[0] for v in range(self.n_vars)]

    def decode(self, syndrome: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Decode a single syndrome using min-sum BP."""
        # Initialize messages
        m_vc = np.zeros((self.n_vars, self.n_checks))
        m_cv = np.zeros((self.n_checks, self.n_vars))

        for v in range(self.n_vars):
            for c in self.var_to_checks[v]:
                m_vc[v, c] = self.channel_llr[v]

        s = 1 - 2 * syndrome.astype(np.float64)

        converged = False
        for _ in range(self.max_iter):
            # Check-to-variable (min-sum)
            for c in range(self.n_checks):
                vars_c = self.check_to_vars[c]
                if len(vars_c) == 0:
                    continue
                for vi, v in enumerate(vars_c):
                    sign = s[c]
                    min_abs = float('inf')
                    for v2 in vars_c:
                        if v2 == v:
                            continue
                        msg = m_vc[v2, c]
                        sign *= (1.0 if msg >= 0 else -1.0)
                        min_abs = min(min_abs, abs(msg))
                    m_cv[c, v] = sign * (min_abs if min_abs < float('inf') else 0.0)

            # Variable-to-check
            for v in range(self.n_vars):
                checks = self.var_to_checks[v]
                total = self.channel_llr[v] + np.sum(m_cv[checks, v])
                for c in checks:
                    m_vc[v, c] = total - m_cv[c, v]

            # Hard decision
            llr = np.array([
                self.channel_llr[v] + np.sum(m_cv[self.var_to_checks[v], v])
                for v in range(self.n_vars)
            ])
            error = (llr < 0).astype(np.int8)

            if np.array_equal((self.H @ error) % 2, syndrome.astype(np.int8)):
                converged = True
                break

        prediction = (self.obs_matrix @ error) % 2
        return prediction.astype(bool), converged

    def decode_batch(self, syndromes: np.ndarray) -> Tuple[np.ndarray, float]:
        """Decode a batch of syndromes.

        Returns:
            predictions: bool array [n_shots, n_observables]
            convergence_rate: fraction of shots that converged
        """
        n_shots = syndromes.shape[0]
        predictions = np.zeros((n_shots, self.n_obs), dtype=bool)
        n_conv = 0
        for i in range(n_shots):
            pred, conv = self.decode(syndromes[i])
            predictions[i] = pred
            if conv:
                n_conv += 1
        return predictions, n_conv / n_shots
