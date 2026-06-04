# Group-equivariant neural decoding of bivariate-bicycle codes — a negative result

This directory contains a self-contained study of whether **group-equivariant neural belief
propagation** offers any advantage for decoding the [[72,12,6]] bivariate-bicycle (BB) qLDPC code,
at both code-capacity and circuit-level noise. The honest answer, after controlled experiments, is
**no**: the Z₆×Z₆ symmetry prior provides no decoding advantage that a parameter-matched
non-equivariant baseline does not also achieve. We document it here because the negative is clean,
controlled, and consistent with the established result that an *equivariant* decoder gains nothing
from code automorphisms — belief propagation on quasi-cyclic LDPC is automorphism-equivariant
(Geiselhart et al., [arXiv:2202.00287], classical QC-LDPC). The productive direction is the
*non-equivariant* automorphism ensemble (AutDEC, [arXiv:2503.01738], quantum LDPC including BB codes
at circuit-level noise), not the equivariant prior studied here.

## What the code does

- `bb_code.py` — constructs the [[72,12,6]] BB code (Bravyi et al., *Nature* 2024): CSS checks
  H_X, H_Z, the k=12 logicals, and the Z₆×Z₆ group action. (`bb_code_selftest.json`)
- `bb_circuit.py` — circuit-level Z-basis memory (Stim). The syndrome-extraction CNOT schedule is
  found by exhaustive search: deterministic detectors require the Z-check monomial order to be the
  time-reversal of the X-check order. Validated to circuit distance 6.
- `canon_dem.py` — canonical detector-error-model extraction (check matrix, observable matrix,
  priors straight from Stim's DEM) used as the single source of truth for all circuit-level
  decoding, plus a standing regression test (BP-OSD-10 at R=6, p=0.003 must give ≤0.2% LER).
- `neural_bp.py`, `circ_neural_bp.py` — unrolled normalized min-sum neural BP on the code Tanner
  graph (code-capacity) and the DEM factor graph (circuit-level), with three weight-tying modes:
  `equiv` (tied across symmetry orbits), `free` (per-edge), `free_random` (tied across a random
  partition of matched size — the decisive control), and `classical` (untrained).
- `equiv_decoder.py`, `struct_equiv.py` — feedforward equivariant CNN decoders (generic and
  structure-aware kernels), provably equivariant.
- `stepA_orbits.py`, `stepA_canon.py`, `stepA_full.py` — orbit-structure analysis: the Z₆×Z₆ ×
  temporal-bulk symmetry group ties the circuit-level factor graph by ~110×.
- `sweep.py`, `bridge_canon.py`, `trains_canon.py` — the staged circuit-level evaluation.
- `verify_nbp.py`, `verify_osd7.py`, `bposd_*` — the leak-aware and matched-OSD controls.

Scripts write their outputs to `qldpc/results/` (created on run). Requires `numpy`, `scipy`,
`torch`, `stim`, `ldpc` (v2, for `BpOsdDecoder`).

## Key findings

1. **Code-capacity, feedforward.** A parameter-matched plain MLP beats the Z₆×Z₆-equivariant CNN at
   every training-set size; matching the kernel support to the code structure changes nothing.

2. **The apparent neural-BP win is memorization.** On a small code at low noise, common low-weight
   syndromes recur, so a trained decoder can lookup-memorize the syndrome→error map. Evaluated on
   *novel* syndromes (absent from training), the equivariant neural-BP advantage over BP-OSD
   disappears — and the gap is not an OSD-strength artifact (verified by matching OSD order).

3. **Circuit-level, matched-parameter control (decisive).** With identical parameter counts, the
   symmetry-chosen weight-tying (`equiv`) ties a *random* partition of the same size (`free_random`)
   to within seed noise, and both sit slightly above tuned classical BP-OSD. The earlier
   appearance of an equivariance advantage was a consequence of *fewer parameters*, not of the
   symmetry: any tying of the same size performs equally well.

The symmetry-equivariance prior, as a decoding lever for these codes, does not survive a controlled
comparison. This matches the equivariant-decoder no-go: BP on quasi-cyclic LDPC is
automorphism-equivariant, so the symmetry buys nothing unless it is deliberately broken (Geiselhart
et al. [arXiv:2202.00287]). The complementary positive result — that a *non-equivariant* automorphism
ensemble does help on BB codes at circuit level — is AutDEC ([arXiv:2503.01738]); our negative
concerns the equivariant prior specifically, not automorphism ensembling per se.

## Noise models and baselines

Code-capacity i.i.d. depolarizing and circuit-level depolarizing (Stim). Classical baselines are
BP-OSD (`ldpc`, min-sum + combination-sweep OSD) and Relay-BP, tuned to the regime.

## Acknowledgements

The experimental design — in particular the matched-parameter control, the novel-syndrome split,
and the maximum-likelihood ceiling check — was sharpened through review by the **Coda** expert model,
which is gratefully acknowledged for adversarial feedback on methodology.

[arXiv:2202.00287]: https://arxiv.org/abs/2202.00287
[arXiv:2503.01738]: https://arxiv.org/abs/2503.01738
