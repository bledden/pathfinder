import numpy as np, stim
from qldpc.probe.tesseract_anchor import tesseract_ler, beam_convergence

def _surface_d3():
    return stim.Circuit.generated("surface_code:rotated_memory_z", rounds=3, distance=3,
                                  after_clifford_depolarization=0.01,
                                  before_measure_flip_probability=0.01)

def test_tesseract_decodes_surface_d3_better_than_chance():
    circ = _surface_d3()
    ler = tesseract_ler(circ.detector_error_model(decompose_errors=False), circ, shots=2000, beam=16)
    assert 0.0 <= ler < 0.2

def test_beam_convergence_returns_plateau():
    circ = _surface_d3()
    res = beam_convergence(circ, shots=2000, ladder=(4, 8, 16))
    assert res["beam_star"] in (4, 8, 16)
    assert "ler_by_beam" in res and len(res["ler_by_beam"]) == 3

def test_beam_convergence_resolves_plateau_with_power():
    circ = _surface_d3()
    res = beam_convergence(circ, shots=20000, ladder=(2, 4, 8, 16), margin_rel=0.15)
    assert res["plateau_resolved"] is True
    assert res["beam_star"] < 16

def test_beam_convergence_unresolved_falls_back_to_max():
    circ = _surface_d3()
    res = beam_convergence(circ, shots=800, ladder=(2, 4, 8, 16), margin_rel=0.10)
    assert res["plateau_resolved"] is False
    assert res["beam_star"] == 16
