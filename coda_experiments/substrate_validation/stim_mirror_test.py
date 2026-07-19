"""Substrate validation, Stim tableau route (see README.md in this directory).

Mirrors the Qiskit Surface-17 circuit contributed by Coda (Conductor Quantum's
quantum-computing assistant) in a stabilizer-tableau simulation and asserts the
same structural claims its statevector run established:

  (a) the 4 Z-stabilizer ancilla bits are deterministically 0 (noiseless Z-memory),
  (b) the logical-Z parity (data qubits {0,1,2}) is even in every shot,
  (c) all 4 Z-stabilizer parities reconstructed from the transversal data readout
      are even in every shot,
  (d) the joint outcome cardinality is exactly 2^8 = 256 (4 free X-ancilla bits +
      4 free data bits after 4 Z-parity constraints and 1 logical-parity constraint),
  (e) the data register collapses to exactly 16 patterns = the X-stabilizer group's
      supports acting on |0>^9.

Run:  python -m pytest coda_experiments/substrate_validation/stim_mirror_test.py
"""
import stim
import numpy as np

Z_STABS = [[0, 1, 3, 4], [4, 5, 7, 8], [2, 5], [3, 6]]
X_STABS = [[1, 2, 4, 5], [3, 4, 6, 7], [0, 1], [7, 8]]
LOGICAL_Z = [0, 1, 2]
SHOTS = 4096


def build_circuit() -> stim.Circuit:
    c = stim.Circuit()
    for i, qs in enumerate(Z_STABS):
        for q in qs:
            c.append("CX", [q, 9 + i])
    for i, qs in enumerate(X_STABS):
        c.append("H", [13 + i])
        for q in qs:
            c.append("CX", [13 + i, q])
        c.append("H", [13 + i])
    c.append("M", list(range(9, 13)))    # meas 0-3:  Z-ancillas
    c.append("M", list(range(13, 17)))   # meas 4-7:  X-ancillas
    c.append("M", list(range(9)))        # meas 8-16: data 0-8
    return c


def test_substrate():
    shots = build_circuit().compile_sampler(seed=7).sample(SHOTS)
    zanc, data = shots[:, :4], shots[:, 8:]
    assert not zanc.any(), "Z-stabilizer bits must be deterministically 0"
    assert not (data[:, LOGICAL_Z].sum(1) % 2).any(), "logical-Z parity must stay even"
    for s in Z_STABS:
        assert not (data[:, s].sum(1) % 2).any(), f"Z-parity {s} must be even"
    assert len(set(map(bytes, shots))) == 256, "joint cardinality must be 2^8"
    assert len(set(map(bytes, data))) == 16, "data patterns must be the X-stabilizer group"


if __name__ == "__main__":
    test_substrate()
    print("substrate validation: all assertions pass")
