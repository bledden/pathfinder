# Independent substrate validation (Surface-17, d=3 memory-Z)

Validates the code substrate underlying the paper's §3.3 scope statement and the §5.3
input-shape correction, by three routes with no shared code:

1. **Analytic** (Coda, Conductor Quantum's assistant, session of 2026-07-18): detector
   count r(d²−1), time axis r+1, spatial grid (d+1)×(d+1) derived from the rotated
   code's stabilizer schedule and plaquette geometry — blind, matching the sealed
   expected values.
2. **Qiskit statevector execution** (Coda, same session): the Surface-17 circuit in
   `qiskit_surface17_substrate.py`, 4096 noiseless shots — Z-stabilizers deterministic,
   logical-Z parity preserved, and the unplanned structural witness: outcome cardinality
   exactly 2⁸ = 256 (4 free X-ancilla bits × 4 free data bits after the 4 Z-parity +
   1 logical-parity constraints).
3. **Stim tableau execution** (this repo): `stim_mirror_test.py` reproduces all of the
   above under a stabilizer-tableau simulator with assertions.

**Boundary (stated by Coda, kept here):** this validates the *code substrate* — the
stabilizer structure, the single logical observable, and the constraint counting behind
the (r+1, d+1, d+1) grid. It does not reproduce Stim's detector-object metadata itself;
that rests on the two Stim execution routes (the local derivation and the H200 receipt's
embedded `grids` field) plus route 1's analytic derivation.
