# A kernel-grounded, MLE-anchored latency–LER Pareto benchmark for circuit-level qLDPC decoding

*Working draft (2026-06-08). Methods-first per the locked outline. Abstract + Introduction drafted LAST.
Every number below is committed/json-traceable on branch `qldpc-mle-foundation`; "[measured]" vs
"[reasoned]" labels are carried from the artifacts. Sources: `docs/superpowers/specs/2026-06-05-…md`
(design, gitignored), `qldpc/{foundation,probe,zoo,kernel}/`, and the result JSONs.*

---

## Section map (drafting order: §3–9 methods/results first → §10–11 → §1 intro → abstract LAST)
- **§1 Introduction** — *[draft last]* the decoder-methodology + kernel-systems contribution; the
  classical–quantum bridge is kernel-side (simulation-mature decoder methodology grounded in real-GPU
  performance). Scoped first-to claim (see §12). 
- **§2 Setup & MLE anchor** — code, noise, DEM provenance, Tesseract MLE anchor + beam convergence.
- **§3 Degeneracy probe (B-MLE)** — coset-ML vs MLE bound; the kill-switch; why MLE-anchoring is valid.
- **§4 Decoder zoo & matched protocol** — 7 decoders, byte-identical DEM, gates G1/G2/G3, ms=0.625 bar.
- **§5 Statistics & pre-registration** — Wilson/TOST, per-shot-paired bootstrap, Holm/BH, frozen budget.
- **§6 The fused BP kernel** — CPU baseline → torch-GPU baseline → Triton bounded-degree min-sum kernel.
- **§7 LER-identity proof** — kernel == baseline (one-iteration bit-identical; LER 156=156=156).
- **§8 Latency & the Pareto frontier** — CUDA-event harness, throughput vs real-time, cycle-time budgets.
- **§9 Amenability taxonomy** — fallthrough-vs-p, Amdahl, roofline, memory; the named next lever.
- **§10 Negative results** — T10/AutDEC closure; Track-2 equivariance; LSD-pathological domain note.
- **§11 Deployment regime & outlook** — where kernel-accelerated decoders are deployment-ready; companion.

---

# METHODS (drafted first — the section a skeptical reviewer attacks)

## §2 Setup & the MLE anchor

### 2.1 Code, noise model, and DEM provenance
Primary code: the **[[72,12,6]] bivariate-bicycle (BB) code** (`BBCode`, l=m=6, A=(x³+y+y²),
B=(y³+x+x²); N=36, n=72, k=12). We benchmark **circuit-level** memory experiments at distance d=6,
R=6 rounds, in both **X- and Z-memory bases**, under the **SI1000 superconducting noise model** (Gidney):
two-qubit gate p (DEPOLARIZE2), one-qubit/idle p/10, measurement-flip 5p, reset p/10,
idle-during-measurement 2p. The Z-memory schedule is the validated time-reversed-monomial CNOT layering
that makes detectors deterministic; the X-memory variant measures the true H_X=[A|B] (row-indexed, not
the transpose — a construction-time commute guard `(measured_H @ obs.T) % 2 == 0` enforces this).
Idle noise is the standard Gidney model: DEPOLARIZE1(p/10) on data qubits not in a CNOT layer, and
DEPOLARIZE1(2p) on data qubits during the measurement layer. *(Honest detail: idle errors on a data
qubit share the detector signature of that qubit's gate errors, so they add error MASS to the DEM
without adding distinct mechanisms — verified by exhaustive single-error injection; code distance is
preserved at 6.)*

**DEM provenance (load-bearing).** All decoders consume one **byte-identical** detector-error-model
(`circuit.detector_error_model(decompose_errors=False)`), validated faithful to the canonical sparse
extraction (`canon_dem.extract`). Because the DEM is from a *known simulated* circuit (not a device fit),
gap-to-MLE is an auditable scientific object rather than gap-to-MLE-on-a-fit. *(This is the methodological
reason the headline contribution is simulation-natural; see §11 / companion for the device-DEM case.)*

### 2.2 The MLE anchor (Tesseract) and beam convergence
We anchor "optimality" to the exact **most-likely-error (MLE)** decoder — Tesseract (A*+beam,
`quantumlib/tesseract-decoder`, Apache-2.0), ingesting the same Stim DEM. Honest scoping: this is
most-likely-*error*, near-exact at large beam, **not** most-likely-*coset* (true ML) — because exact
coset-ML on BB is intractable at every reachable regime (verified: TN contraction width ≈554 at
circuit-level d6/R6; 77 even at code-capacity — BB's expander structure gives high treewidth everywhere).
We use beam B*=64 and verify convergence: on the n=18 spine code the MLE fail-count is **identical across
beam {16,32,64,128}** (`beam_convergence_check` in the verdict artifact); on the grid low-p cells the
anchor LER does not move outside the bootstrap CI from beam 64→128. *[measured]*

## §3 The degeneracy probe — bounding MLE vs coset-ML (the B-MLE verdict)
Anchoring to MLE is only legitimate if the **MLE↔coset-ML degeneracy gap** is small. We measure it at the
small scales where exact coset-ML *does* contract.
- **Exact coset-ML** via a parity-decomposed tensor-network contraction (`qldpc/foundation/tn_mld.py`):
  parity checks decomposed into bond-2 XOR chains (single high-rank factors are invalid), contracted to a
  per-shot logical-class probability vector; argmax (tolerance-tie-broken for bit-reproducibility) →
  coset-ML LER. Validated to **machine precision against brute force** on a genuine BB toy (n=18, k=4;
  per-shot probability vectors match to 5.6e-16).
- **Probe spine:** 3 genuinely-bivariate BB toys (n=12/18/24) **derived, not curated** — by a committed
  enumeration (`qldpc/probe/spine_selection.py`) selecting the minimum-exact-tie-fraction code per (l,m)
  under gates {STRONG bivariate (A,B each mix x,y), A≠B, CSS, k=4, contraction-width≤28}; plus surface
  d3/d5 + color-d3 cross-family references.
- **Pre-registered kill-switches:** *level* — any MLE/coset-ML ratio > 2× → demote; *trend (CI-aware)* —
  BB ratios strictly increasing in n AND the increase exceeds sampling noise (disjoint endpoint CIs).
  Multi-seed (≥5) × two p ∈ {0.05, 0.01}.
- **Verdict: B-MLE.** Every BB ratio ≤ **1.084**, pooled ≈1.0, non-monotone in n; neither kill-switch
  fires. *[measured]* (The p=0.01 row is monotone-increasing in point estimate but within noise — the
  CI-aware rule correctly does not fire; the old strict rule would have spuriously demoted. seed=123
  counterexample recorded in `methodology_validation`.) ⇒ MLE-anchoring is defensible on BB.

## §4 Decoder zoo & the matched protocol
Seven decoders on the **one shared DEM**: BP (min-sum), BP-OSD order-0 and order-10-combination-sweep,
BP+LSD, Relay-BP (IBM), sliding-window (qLDPCOrg, streaming), and Tesseract (the MLE anchor). Cascade
(no released code) and nv-qldpc are cited as reference points, not in the matched harness; Astra
(code-capacity only) excluded.
- **Gate G1 (DEM-identity):** each decoder's check/observable matrices are sha256-pinned to the shared
  DEM; the harness fails-fast on any hash mismatch. (Relay-BP's matrices verified byte-identical to the
  canonical extraction.)
- **Gate G2 (tie-break):** every adapter declares a deterministic tie-break policy in an approved set,
  recorded in the manifest.
- **Gate G3 (smoke-grid):** a 1k-shot single-p/d run across the full zoo on every push.
- **Classical bar:** normalized min-sum `ms_scaling_factor=0.625` (the standard, stronger bar; verified
  byte-identical wiring to the repo's BP-OSD, differing only in this scaling — documented vs the legacy
  default 1.0).

## §5 Statistics & pre-registration
- **Pre-registration committed to git BEFORE the run** (`qldpc/zoo/prereg.json`): decoder list+configs,
  p-grid {0.001,0.002,0.003,0.005}, bases {X,Z}, R=6, per-cell failure target 300, frozen pilot-sized
  shots, primary endpoint per-round λ, the gap statistic, multiplicity method, per-cell DEM hashes.
- **Gap-to-MLE statistic:** the **per-shot PAIRED bootstrap** ratio (decoder vs Tesseract failures on the
  SAME sampled shots; ≥1000 resamples) — not an aggregate ratio with a single CI.
- **Beat/tie/multiplicity:** beat = non-overlapping Wilson CIs AND both cells ≥ failure target; tie =
  TOST at a declared margin; **Holm-Bonferroni (primary) + BH-FDR (secondary) across all 48 (decoder,p,
  basis) cells, full grid reported including losses.** Holm is applied across all 48 cells (X+Z together;
  the more conservative, skeptic-proof choice — defended in one sentence).

## §6 The fused BP kernel
Staged baseline → kernel:
- **CPU reference** (`qldpc/kernel/bp_baseline.py`): from-scratch normalized-min-sum BP on the shared
  CSR/CSC Tanner layout (four phases: gather / exclude-self via two-smallest-magnitude min + sign-product
  / per-check min-sum × coset-sign / scatter); one-iteration **bit-identical to a hand-derived reference**;
  functional LER-equivalence to ldpc BP.
- **torch-GPU baseline** (`bp_gpu.py`): device-agnostic, shots-batched, bit-identical to the CPU
  reference on fp64 — an honest GPU baseline (not numpy).
- **The kernel** (`bp_triton.py`): a **fused bounded-degree min-sum BP Triton kernel for irregular
  circuit-level qLDPC DEMs.** *(Honest reframe: the circuit-level DEM is irregular — 252 checks, 1584
  bits, 4536 edges, check degree 13/20, bit degree 2/3 — NOT block-circulant; the kernel is a
  bounded-degree edge-unroll (MAXDEG_C=20, MAXDEG_B=3, `tl.static_range`, mask-on-load), portable to ANY
  qLDPC DEM. The circulant structure is real on the abstract H_Z — appendix note — but is not what the
  circuit decoder traverses.)* Two fused launches/iteration (check-update, bit-update), FP32, coalesced.

## §7 LER-identity proof
The kernel reproduces the baseline: one-iteration hard-decision agreement **1.0**; full-decode **LER
identical 156 = 156 = 156** (Triton / torch / CPU, 2000 shots, p=0.003). *[measured]* The per-shot
99.35% figure is fp32-vs-fp64 near-tie flips (one bit in 1584) — **report LER-identity as the headline,
the fp32-flip caveat in methods.** This converts the kernel from "fast" to "fast and provably correct."

## §8 Latency & the Pareto frontier
- **Harness** (`qldpc/zoo/latency.py`): CUDA-event timing for GPU decoders, perf_counter (amortized) for
  CPU decoders; warmup ≥10, measure ≥100, drop first 10%; report mean, **p99.9**, throughput, µs/syndrome,
  at batch ∈ {1,4,16,…,16384}. Bootstrap CIs (2000 resamples) per point.
- **Clock conditions (honest):** clock-locking is host-gated on RunPod (not an in-pod command); the H200
  runs at applications-clock 1980 MHz / boosts to 1980 under load. **Variance-over-runs check: CV
  Triton 0.01–1.02%, torch 0.07–0.24%** (3 runs) — confirms thermal stability despite unlocked clocks
  (the stated fallback). *[measured]*
- **THROUGHPUT vs REAL-TIME (load-bearing).** The kernel win is a **throughput** claim: at batch-1
  (real-time) Triton-BP is **1046 µs/syn and loses to CPU-BP (262 µs/syn)** — the ~1 ms launch+H2D/D2H
  is unamortized; the headline left-shift (0.79 µs/syn; **55× vs torch-GPU**, 331× vs CPU at LER-identity)
  is the **throughput regime** (batch-16k). This *measures* the throughput/real-time boundary at ~1 ms.
  Real-time SC-window decoding (~6 µs) remains FPGA/ASIC territory; the GPU kernel serves the throughput
  regime — offline post-processing of memory experiments, decoder R&D / co-design loops over millions of
  shots, and multi-patch concurrent decoding.
- **Cycle-time budgets (per decoding window, R=6 → ~6µs SC / ~1ms ion / 4.45ms atom):** only
  Triton-kernel-BP fits SC (batch ≥4096, amortized); GPU-BP fits ion; CPU-BP/OSD-0/LSD fit atom;
  Tesseract fits none. **Pareto figure** (`pareto.png`): x=latency/syndrome (µs, log), y=gap-to-MLE
  (p=0.003 X+Z mean), points with **p99.9 whiskers** + bootstrap CIs, cycle-time bands, frontier; the
  Triton-BP per-batch p99.9 tail crossing the SC band is the deployment-regime callout.

## §9 Amenability taxonomy (the methods spine; the named next lever)
*Lead the kernel paragraph with the decomposition:* the fused kernel pulls the **BP front-end** of the
throughput frontier left 7–55× at LER-identity; the **practical-decoder bar (BP-OSD-10) is ~66% bound by
dense GF2 Gaussian elimination** on the BP-residual lattice (Amdahl serial fraction [measured] via the
decoder-minus-bare-BP increment, anchored to the launch-bound receipt), setting its kernel ceiling at
~1.5× absent a dense-linalg attack. We therefore **name the OSD-GE step the next GPU lever** and provide
the amenability taxonomy (Tab. X, in §9 next to the figure):
- **Fallthrough-rate vs p** [measured]: 0.9% / 3.6% / 10.4% / 39.2% (p=.001→.005); effective latency =
  BP-floor + fallthrough×OSD-increment → 270→470 µs/syn. BP's latency win is real only where it converges.
- **Amdahl serial-fraction** (kernel ceiling per decoder): BP-OSD-10 0.66/1.5×, BP 0.05/20×, etc.
- **Roofline:** BP 0.17 flop/byte, bandwidth-bound, **no GEMM/tensor-core** (min/sum semiring forecloses
  it). **The bifurcation:** BP cannot use tensor cores; OSD's dense GE is the FIRST pipeline sub-step that
  ADMITS them (BLAS-3-shaped → blocked GEMM) — "the named largest unexploited GPU lever for the
  practical-decoder bar *on this benchmark*." This positions the paper as a map of the kernel-attack
  surface, not a single result.
- **Memory footprint:** edges×fp32×batch + aux → max batch on H200/A100/RTX.
- **Launch-overhead receipt (fusion evidence):** torch 110 CUDA launches/iter vs Triton 24 (4.58×
  fusion), host-dispatch/GPU-runtime = 0.999 (launch-bound). [measured]

## §10 Negative results (the section that earns scientific seriousness)
- **T10 / AutDEC closure.** The residual practical-decoder gap-to-MLE (≤4.7%; BP-OSD-10) is a
  **high-syndrome-weight tail** where MLE's exhaustive enumeration outperforms OSD's local procedure on
  the **same logical class** — a tail-density failure, structured along an axis (syndrome weight)
  **orthogonal to the symmetry-averaging mechanism**, not a separable symmetry-related fault class
  (weight 39 vs 34, +9.12σ, d=0.59; spread ratio 1.03; obs-flips 5.5 vs 5.7) [measured, 15k shots].
  Symmetry-as-test-time-augmentation (AutDEC-class) therefore has no leverage — a negative result that,
  with the Track-2 architectural-prior negative, closes the equivariance arc on this benchmark for both
  routes. *Useful negative:* AutDEC would help where the practical-MLE gap is dominated by separable fault
  classes, not tail density.
- **OSD worst-case (workload sensitivity).** Under uniform-random (decoder worst-case) syndromes, both
  BP-OSD-0 (21.7×) and BP-OSD-10 (12.7×) reach **≈13.3 ms/shot** — the worst-case ceiling is ~invariant
  to OSD depth (the GE on a dense syndrome, not the postproc search). Practitioners benchmarking on
  synthetic syndromes systematically overestimate OSD-class latency. [measured]
- **LSD domain note.** BP+LSD's localized-statistics clustering degrades non-gracefully on uniform-random
  syndromes (a single fully-dense shot did not complete in >11 min) — a domain-of-applicability finding
  (our realistic circuit syndromes do not stress it), warranting caution in adversarial-noise budgeting.

## §11 Deployment regime & outlook
Neutral-atom (~4.45 ms) and trapped-ion (~1 ms) are where **kernel-accelerated BP-OSD-10 / Relay-BP are
deployment-ready today**; superconducting real-time remains FPGA territory. The contribution is NOT "GPU
beats FPGA for SC" — it is "GPU kernel work makes BP-OSD-class decoders deployment-ready for slower-cycle
modalities and pulls the whole throughput frontier left." **Companion (in progress):** real-device
validation on IBM Heron r2 — the methodology here is substrate-agnostic and consumes any DEM.

## §12 Scoped contribution / "first-to" claim (LOCKED wording)
*"To our knowledge, the first kernel-grounded latency–LER Pareto analysis of circuit-level [[72,12,6]] BB
decoders with an MLE anchor, an LER-identity-proven kernel implementation, and a per-decoder Amdahl
decomposition."* (NOT "first GPU BP kernel for qLDPC" / "fastest BP".)

---

## TODO (drafting order; do NOT add new analyses — spine locked)
- [ ] §1 Introduction (draft after the body).
- [ ] Abstract — DRAFT LAST; lead with throughput + the measured ~1 ms boundary (Coda's locked formulation).
- [ ] Fill exact CI numbers + Tab. X (taxonomy) + the both-workloads supplementary table from the JSONs.
- [ ] Pre-registration as a supplementary artifact (PDF/md of prereg.json) + commit hash in §5.
- [ ] Convert to RevTeX for PRX Quantum at submission; figures: pareto.png (hero), fallthrough.png, amdahl.png.
- [ ] Revision passes: hostile-reviewer / scope / measured-vs-reasoned / figure-caption.
