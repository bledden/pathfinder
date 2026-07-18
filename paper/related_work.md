# Related work (draft section)

*Working draft (2026-06-08). Companion to `paper/qldpc_pareto.md`; same scoped claims
(throughput-lead, MLE-anchored, multiplicity-corrected, latency-grounded). Every competitor fact
below was verified against the arXiv source / vendor blog on 2026-06-08 (WebFetch). Where a fact
differed from the brief's framing, it is corrected in-place and flagged with **[corrected]**.*

---

## Ferraz spike verdict

**VERDICT: GREEN (with caveats) — artifact-blocked but DEM-portable in principle; defer the actual zoo-row run.**

Paper: Ferraz, Coutinho, Falcao, Gomes, Monteiro, Silva, *"GPU-Accelerated Syndrome Decoding for
Quantum LDPC Codes below the 63 µs Latency Threshold"* (arXiv:2508.07879; U. Coimbra / Instituto de
Telecomunicações).

**(a) Public code artifact? NO.** Exhaustive search of the full HTML (intro, methods, results,
references, acknowledgements, data-availability) found **zero** mention of GitHub, Zenodo, an
open-source release, or "available upon request." There is no artifact. This is the binding
constraint: we cannot drop it into the zoo as a runnable row today — only as a reconstructed
re-implementation or a cited reference point.

**(b) What it decodes / decoder / substrate / latency.**
- **Codes:** the Bravyi et al. *Nature* (2024) bivariate-bicycle family — explicitly **[[72,12,6]],
  [[108,8,10]], [[144,12,12]], [[288,12,18]], [[784,24,24]]**. It tests *our exact primary code*
  ([[72,12,6]]). **[note]** The brief asked "Panteleev–Kalachev? BB?" — the answer is **BB
  specifically** (the abstract name-checks the PK lineage as motivation, but the evaluated codes are
  the Bravyi BB codes).
- **Decoder:** pure **min-sum BP** (MSA) with soft-syndrome support. **BP-only — no OSD / no
  ordered-statistics / no post-processing.** A single GPU kernel runs both X and Z components.
- **Substrate:** commodity NVIDIA GPUs (RTX 3090 / RTX 4090). H_X / H_Z stored in constant memory;
  Tanner-graph message passing over an arbitrary sparse structure (no circulant/quasi-cyclic
  exploitation observed).
- **Latency:** title threshold is **Google Willow's 63 µs** surface-code real-time decoder budget
  (d=5, ~1M cycles, 2024). Reported: **< 50 µs for [[784,24,24]]**, **35 µs / 28 µs for the smaller
  codes** (incl. [[72,12,6]]) on RTX 3090 / 4090, **23.3 µs** lowest. **Noise model is not stated**
  — no code-capacity/phenomenological/circuit-level declaration, **no DEM, no Stim**. These are
  *per-decode* latency figures, i.e. effectively a single-syndrome / real-time regime.

**(c) Port feasibility to our [[72,12,6]] BB DEM → GREEN on the algorithm, RED on the substrate today.**
The decoder is a *generic* min-sum BP that ingests arbitrary sparse H_X/H_Z over a Tanner graph — it
is **not hard-wired to a code family** (no block-circulant assumption). A circuit-level DEM is just a
larger irregular Tanner graph, so the algorithm is **DEM-ingestible in principle** — the same class as
our own fused-min-sum kernel. That makes it a legitimate future zoo row **once GPU is available**.
Two honest blockers keep it from being a row *now*:
1. **No artifact** → it would have to be re-implemented from the paper (the kernel is straightforward
   min-sum, so feasible, but it is a build task, not a `pip install`).
2. **Needs GPU** → defer the run per the brief; CPU-only environment cannot benchmark it.

**Cross-claim sanity check (do NOT silently absorb their headline).** Ferraz reports **28 µs for
[[72,12,6]] BP at batch-1**, whereas our measured boundary is that **batch-1 GPU BP loses to CPU at
~1 ms/syndrome** (the ~1 ms launch + H2D/D2H is unamortized; our throughput win lives at batch-16k).
The two are not directly comparable and the gap is plausibly explained by: (i) their codes are
*code-capacity-sized* H (n=72 data bits, ~36 checks) vs our *circuit-level* DEM (252 checks, 1584
bits, 4536 edges, R=6 rounds) — a far smaller graph with far less H2D traffic; (ii) no stated DEM /
detector overhead; (iii) a persistent-kernel / resident-graph design that amortizes launch differently
than a torch dispatch. **Action for the eventual zoo run:** reproduce on our *circuit-level* DEM with
our CUDA-event harness before quoting any head-to-head latency — their 28 µs is for a code-capacity-
scale problem and is not a counterexample to our circuit-level batch-1 boundary. This is exactly the
kind of "apples-to-oranges single number" our methodology spine (matched DEM, throughput-vs-real-time
boundary) exists to discipline.

**Bottom line:** GREEN as a *planned* zoo row (DEM-portable BP, tests our exact code) — **deferred**:
needs (1) a re-implementation since there is no artifact, and (2) GPU to run. Until then it is a cited
reference point, not a matched-harness row, and its latency numbers must be re-measured on our
circuit-level DEM, not transcribed.

---

## Related work

### Fully-parallelized BP / BP-SF (Wang, Li, Mueller, arXiv:2507.00254)

Wang, Li, and Mueller's *"Fully Parallelized BP Decoding for Quantum LDPC Codes Can Outperform
BP-OSD"* (arXiv:2507.00254) attacks the same bottleneck we name — the dense Gaussian elimination
inside BP-OSD — but along an **algorithmic** axis rather than a kernel one. Their **BP-SF
(syndrome-flip)** method monitors bit-level oscillation during BP to identify unreliable bits,
generates multiple candidate syndromes by selectively flipping, and decodes each with short-depth BP,
**eliminating Gaussian elimination entirely**; on the **[[144,12,12]]** BB code it reaches LER
comparable-to-or-better-than BP-OSD at **~70% of BP-OSD's average latency**. Our contribution is
orthogonal and complementary on three axes. **(1) Mechanism:** BP-SF *removes* the GE step by
changing the algorithm; we *accelerate* the BP front-end with a fused bounded-degree min-sum **Triton
kernel** (7–55× over a bit-identical torch baseline, LER-identical) and then *name and decompose* the
GE step as the next GPU lever (BP-OSD-10 is ~66% serial in dense GF2 GE → a measured ~1.5× kernel
ceiling absent a dense-linalg attack) rather than removing it — the two approaches could stack.
**(2) Evidence standard:** they report a single decoder's **relative** latency (% of BP-OSD); we
report a **full 7-decoder zoo on one byte-identical DEM, anchored to a Tesseract MLE oracle, with
pre-registration and Holm/BH multiplicity correction across all 48 (decoder, p, basis) cells** — the
gap-to-optimal is an auditable per-shot-paired-bootstrap statistic, not a pairwise ratio.
**(3) Regime:** their parallelism is a single-shot latency argument; ours is a **throughput Pareto
frontier** with a *measured* throughput/real-time boundary (~1 ms batch-1 crossover) and roofline /
Amdahl decomposition per decoder. **[corrected]** The brief listed BP-SF's substrate as
verify-needed and implied a known artifact: in fact the paper **identifies no specific hardware
substrate** (GPU/FPGA/sim unspecified beyond "parallelizable"), reports latency **only as a fraction
of BP-OSD (no absolute µs)**, names **no peer-reviewed venue** in the arXiv record (HPCA-2026
association is plausible but not confirmed on the preprint), and **releases no public code artifact**.
We therefore cite it as the strongest *algorithmic* GE-elimination result and differentiate on
kernel-grounded measurement + benchmark methodology, without claiming a head-to-head latency number
against it.

### Production GPU RelayBP: NVIDIA CUDA-Q QEC 0.6 + NVQLink (April 2026)

NVIDIA's CUDA-Q QEC 0.6 (blog, 2026-04-14) ships a **production, GPU-accelerated RelayBP** decoder for
qLDPC codes wired into `cudaq-realtime` over **NVQLink**, giving **microsecond-latency callbacks
between GPUs and quantum controllers** for real-time, single-shot, in-the-control-loop decoding
("each syndrome arrives as an individual RPC message in a GPU-visible ring buffer"); Quantinuum has
demonstrated real-time QEC on Helios via a custom CUDA-Q QEC integration. We **concede the single-shot
real-time regime to NVQLink outright** — that is precisely the batch-1 / control-loop regime our own
measurements show the GPU *loses* (the win is throughput), and NVQLink's microsecond callback path is
purpose-built hardware-software co-design we do not compete with. Our differentiation is on **openness,
portability, and methodology**, not on beating NVQLink at its own game. The CUDA-Q QEC real-time
dispatch path is, by NVIDIA's own statement, **"entirely in C/C++ and CUDA"** with **no Python
binding** for the latency-critical data plane — i.e. **vendor-supplied and CUDA-only** (the blog
contains no ROCm/AMD support). Our **Triton RelayBP is open and ROCm-portable**: LER
statistically-indistinguishable from the `relay_bp` reference (TOST), and it runs on **AMD MI300X**, a
substrate CUDA-Q QEC cannot target. Beyond portability, we contribute the **methodology spine** —
MLE-anchored gap-to-optimal, pre-registration, multiplicity correction, fallthrough-vs-p, Amdahl /
roofline decomposition, and the explicit throughput/real-time boundary — that a closed production
decoder does not (and need not) expose. The framing is complementary: NVQLink owns single-shot
real-time on NVIDIA silicon; we provide the open, portable, audited *throughput* frontier and the
benchmark methodology, including on non-NVIDIA hardware.

### Predecoding / preprocessing — composes, does not compete

Two recent works offload OSD workload by *preprocessing the syndrome* before the main decoder, and
both **compose with — rather than compete against — our framework**: they shift fallthrough rates
*within* our zoo's decoders without altering the multiplicity-corrected Amdahl spine.

- **Arqade (Knapen, Luo, Tao, Wang, Bruno, Zhang, Sylvester, Saligane, Ravi; arXiv:2605.03180)**,
  *"Mitigating Classical Resource Costs in QEC via Generalized qLDPC Predecoding,"* is a framework
  that auto-constructs **predecoders** from lightweight "predecoding primitives" (covering >90% of
  errors) and synthesizes them to **FPGA / 4K-compatible cryogenic-ASIC** pipelines; placed in front
  of **BP-OSD**, it reduces second-level OSD utilization by **up to 72.71% (BB)** (and 52.35% color,
  36.74% RQT). It is a *predecoder that reduces the work handed to OSD*, i.e. it directly lowers the
  fallthrough rate that drives our effective-latency curve.
- **Local Syndrome-Based Preprocessing (Fan, Suzuki, Ravi, Ueno, Inoue, Tanimoto; arXiv:2509.01892)**
  is a **composable predecoder** that reads the raw syndrome, identifies likely local error events
  (e.g. "XOR=111" detector patterns), and updates BP-OSD's channel-probability vector before
  decoding — *"more than an 80% reduction in BP iterations and total decoding time"* on [[144,12,12]]
  at p=0.05% while preserving LER. It does not replace BP-OSD; it conditions it.

Both reduce the *fallthrough rate* (our §9 axis: effective latency = BP-floor + fallthrough × OSD-
increment) and therefore **slide points along our Pareto frontier without changing the underlying
multiplicity-corrected Amdahl decomposition** — the BP-OSD serial fraction (~66% dense GE) and its
~1.5× kernel ceiling are unchanged; preprocessing simply invokes that serial path *less often*.
**Preprocessing composes with our framework**: an Arqade- or local-preprocessing-fronted BP-OSD is a
drop-in zoo row whose latency our harness would measure with no methodological change. Neither
released a public code artifact at the time of writing.

### Fair decoder baselines + finite-size scaling for BB (Pandey, arXiv:2603.19062)

Pandey's *"Fair Decoder Baselines and Rigorous Finite-Size Scaling for Bivariate Bicycle Codes on the
Quantum Erasure Channel"* (arXiv:2603.19062, Texas A&M) is the closest in spirit on *statistical
rigor* — 200k shots/point, full bootstrap CIs, finite-size scaling with separated statistical vs
systematic uncertainty, erasure-aware fairness correction for the MWPM baseline, reproducible seeds —
across **five BB sizes N = 144 → 1296**. It is a strong rigor benchmark and we cite it as such, but it
lives in a different regime on **all four axes our methodology adds**, each verified against the
source: **(1) Noise — code-capacity erasure channel, explicitly "no circuit-level noise"**; ours is
**circuit-level** (SI1000, R=6 rounds, X- and Z-memory, faithful Stim DEM). **(2) Optimality anchor —
none**: it compares **only BP-OSD and (erasure-aware) MWPM**, explicitly *"without maximum-likelihood
decoding"*; ours is **MLE-anchored** to a Tesseract oracle with a degeneracy-probe (B-MLE) bounding
MLE↔coset-ML. **(3) Multiplicity — single-channel threshold/LER study**; ours applies **Holm/BH across
all 48 (decoder, p, basis) cells** with pre-registration. **(4) Latency — none**: it reports **zero
timing / runtime / wall-clock data**, being purely LER/threshold-focused; ours is a **latency–LER
Pareto** grounded in real-GPU kernel measurement. **[confirmed]** All four differentiators in the
brief hold exactly as stated.

---

## Citations (BibTeX-ready stubs — fill once finalized)

- Ferraz et al., arXiv:2508.07879 — *GPU-Accelerated Syndrome Decoding for Quantum LDPC Codes below
  the 63 µs Latency Threshold.* (BP-only, BB codes incl. [[72,12,6]], RTX 3090/4090, no artifact.)
- Wang, Li, Mueller, arXiv:2507.00254 — *Fully Parallelized BP Decoding for Quantum LDPC Codes Can
  Outperform BP-OSD.* (BP-SF, [[144,12,12]], relative latency ~70% of BP-OSD, no absolute µs, no
  artifact, venue unconfirmed.)
- NVIDIA CUDA-Q QEC 0.6 blog (2026-04-14) + NVQLink / cudaq-realtime — production CUDA-only GPU
  RelayBP, microsecond control-loop callbacks.
- Knapen et al., arXiv:2605.03180 — *Mitigating Classical Resource Costs in QEC via Generalized qLDPC
  Predecoding (Arqade).* (Predecoder → FPGA/ASIC; up to 72.71% BB OSD-utilization reduction.)
- Fan et al., arXiv:2509.01892 — *Accelerating BP-OSD for QLDPC Codes with Local Syndrome-Based
  Preprocessing.* (Composable predecoder; >80% BP-iteration/time reduction on [[144,12,12]].)
- Pandey, arXiv:2603.19062 — *Fair Decoder Baselines and Rigorous Finite-Size Scaling for Bivariate
  Bicycle Codes on the Quantum Erasure Channel.* (Erasure/code-capacity only, no MLE, no latency,
  N=144–1296.)
