"""Interface to Stim for surface code syndrome generation."""
import stim
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class SurfaceCodeConfig:
    distance: int
    rounds: int
    physical_error_rate: float
    code_type: str = "surface_code:rotated_memory_z"


@dataclass
class DecoderGraph:
    n_detectors: int
    n_observables: int
    edges: List[Tuple[int, int, float, List[int]]]  # (src, tgt, prob, observable_mask)


def make_circuit(config: SurfaceCodeConfig) -> stim.Circuit:
    return stim.Circuit.generated(
        config.code_type,
        distance=config.distance,
        rounds=config.rounds,
        after_clifford_depolarization=config.physical_error_rate,
        before_measure_flip_probability=config.physical_error_rate,
        after_reset_flip_probability=config.physical_error_rate,
    )


def sample_syndromes(circuit: stim.Circuit, num_shots: int) -> Tuple[np.ndarray, np.ndarray]:
    sampler = circuit.compile_detector_sampler()
    return sampler.sample(shots=num_shots, separate_observables=True)


def extract_decoder_graph(circuit: stim.Circuit) -> DecoderGraph:
    dem = circuit.detector_error_model(decompose_errors=True)
    n_detectors = circuit.num_detectors
    n_observables = circuit.num_observables
    edges = []

    for instruction in dem.flattened():
        if instruction.type == "error":
            prob = instruction.args_copy()[0]
            detectors = []
            observables = []
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    detectors.append(target.val)
                elif target.is_logical_observable_id():
                    observables.append(target.val)

            if len(detectors) == 1:
                edges.append((detectors[0], -1, prob, observables))
            elif len(detectors) == 2:
                edges.append((detectors[0], detectors[1], prob, observables))

    return DecoderGraph(n_detectors=n_detectors, n_observables=n_observables, edges=edges)
