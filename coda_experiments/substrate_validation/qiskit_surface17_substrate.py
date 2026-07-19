"""Surface-17 (d=3 rotated, [[9,1,3]]) memory-Z substrate circuit — Qiskit route.

Contributed by Coda (Conductor Quantum's quantum-computing assistant, 2026-07-18
session), reproduced here verbatim with assertion tests added. Its original run:
4096 noiseless Aer shots — Z-stabilizers deterministic, logical-Z parity even in
every shot, outcome cardinality exactly 2^8 = 256. See README.md for the
three-route validation this belongs to. Skips cleanly if Qiskit/Aer is absent
(the Stim mirror in this directory covers the same assertions).

Run:  python coda_experiments/substrate_validation/qiskit_surface17_substrate.py
"""
import sys

Z_STABS = [[0, 1, 3, 4], [4, 5, 7, 8], [2, 5], [3, 6]]
X_STABS = [[1, 2, 4, 5], [3, 4, 6, 7], [0, 1], [7, 8]]
LOGICAL_Z = [0, 1, 2]
SHOTS = 4096


def build():
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    data = QuantumRegister(9, "d")
    zanc = QuantumRegister(4, "za")
    xanc = QuantumRegister(4, "xa")
    cbits = ClassicalRegister(17, "c")
    qc = QuantumCircuit(data, zanc, xanc, cbits)
    for i, qubits in enumerate(Z_STABS):
        for q in qubits:
            qc.cx(data[q], zanc[i])
    for i, qubits in enumerate(X_STABS):
        qc.h(xanc[i])
        for q in qubits:
            qc.cx(xanc[i], data[q])
        qc.h(xanc[i])
    for i in range(4):
        qc.measure(zanc[i], cbits[i])
    for i in range(4):
        qc.measure(xanc[i], cbits[4 + i])
    for q in range(9):
        qc.measure(data[q], cbits[8 + q])
    return qc


def main():
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        print("qiskit/qiskit-aer not installed — skipping (Stim mirror covers this)")
        return 0
    qc = build()
    counts = AerSimulator().run(qc, shots=SHOTS, seed_simulator=7).result().get_counts()
    # Qiskit bitstrings are little-endian: c[16]...c[0] left to right.
    outcomes = list(counts)
    assert len(outcomes) == 256, f"cardinality {len(outcomes)} != 256"
    for bits in outcomes:
        b = bits[::-1]  # index by classical bit number
        assert b[0:4] == "0000", "Z-stabilizer bits must be 0"
        dat = [int(b[8 + q]) for q in range(9)]
        assert sum(dat[q] for q in LOGICAL_Z) % 2 == 0, "logical-Z parity must be even"
        for s in Z_STABS:
            assert sum(dat[q] for q in s) % 2 == 0, f"Z-parity {s} must be even"
    print(f"substrate validation (Qiskit): all assertions pass over {len(outcomes)} outcomes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
