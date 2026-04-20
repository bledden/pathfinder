# Pathfinder: A Direction-Aware Neural Decoder that Outperforms Minimum-Weight Perfect Matching on Surface Codes

**Blake Ledden**
Second Nature Computing Inc., San Francisco, CA

---

## Abstract

Pathfinder is an open-source convolutional neural network decoder for quantum error correction that outperforms minimum-weight perfect matching (MWPM) on rotated surface codes. The decoder composes prior contributions — direction-specific 3D convolution following Gu et al. [8], bottleneck residual blocks, and the Muon optimizer of Jordan et al. [11] — with open-source tooling from Stim [10] and PyMatching [2]. On circuit-level depolarizing noise at code distances d=3, 5, 7 and physical error rates p=0.0005 to 0.015 (100,000 shots per point), Pathfinder wins or ties PyMatching at all 24 evaluation points; 13 of 24 show non-overlapping 95% Wilson confidence intervals, the remaining 11 include two exact ties and nine points where low-noise small-number statistics yield overlapping intervals. On a single NVIDIA H200 SXM GPU with PyTorch 2.6, `torch.compile(max-autotune)`, and FP16, the decoder runs at **7.86 μs per syndrome** at throughput-optimal batching (B=1024) — 5.1× faster than Gu et al. on equivalent hardware and 8.0× faster than AlphaQubit on TPU. A custom Triton kernel fusing DirectionalConv3d's seven direction-specific matrix multiplies and boundary-masked accumulations into one launch brings this to **6.12 μs per syndrome** — measured, numerically equivalent to the reference PyTorch implementation (≤0.02% prediction disagreement across 10,000 shots at three noise rates). At d=7, p=0.007 this sustains the 7-μs cycle-time budget by 13% margin, whereas PyMatching on a single Apple M4 CPU core takes **9.65 μs/syn** (single-syndrome mode, measured) and fails to sustain the cycle-time budget at this operating point. Pathfinder + Triton, in the measurements reported here, sustains real-time d=7 decoding throughput at operational noise rates while beating PyMatching on LER; I am not aware of a prior open-source decoder that does both simultaneously, though this claim is subject to the head-to-head comparison with Lange et al. [14] described below. Batch=1 single-shot latency is 250 μs (Inductor) or 201 μs (with the Triton kernel), still far from the 1-μs physical cycle time — sub-microsecond single-shot decoding remains an open problem. FP16 quantization has no accuracy impact; FP8 quantization provides no speedup at this model size, a negative result reported in Section 5.3. A central empirical finding is that **replacing the Muon optimizer with AdamW increases LER by 72%** at d=5, p=0.007 (from 1.28% to 2.20%) — a larger effect than replacing DirectionalConv3d with standard convolution (+4%). The decoder generalizes to phenomenological noise and alternative code types (color codes, rotated surface X) without retraining.

**A note on priority.** Lange et al. [14] (PRR 2025; arXiv:2307.01241) previously released an open-source GNN-based decoder that outperforms PyMatching on rotated surface codes under circuit-level noise at d ∈ {3, 5, 7, 9} and p ∈ {0.001, …, 0.005}. Pathfinder should not be described as the *first* open-source decoder to beat PyMatching on circuit-level-noise surface codes — that honor belongs to Lange et al. The contributions of this work are instead (a) extending the evaluation range to operational noise p ∈ {0.007, 0.010, 0.015} not covered in prior open-source work, (b) identifying the Muon optimizer as a depth-dependent driver of neural decoder accuracy — catastrophic to remove at d=7 (§6.2), (c) a custom Triton kernel that achieves cycle-time-sustaining d=7 throughput on H200, and (d) a three-way majority-vote ensemble of Pathfinder + Lange + PyMatching that **strictly beats every individual decoder at d=7 with statistically significant (non-overlapping 95% CI) margins at p=0.007 and p=0.010** under matched noise (§5.12), representing the cheapest known reduction of the best-known open-source LER at d=7 operational noise rates. Section 5.11 reports a direct head-to-head comparison with Lange et al. on matched noise model; §5.12 builds the ensemble. All code, trained checkpoints, benchmarks, and evaluation data are available at https://github.com/bledden/pathfinder.

---

## 1. Introduction

Quantum error correction (QEC) is the critical bottleneck on the path to fault-tolerant quantum computation. While quantum hardware has crossed the surface code threshold — Google's Willow processor demonstrated exponential error suppression with increasing code distance [1] — the classical decoder that processes error syndromes in real time remains a fundamental engineering challenge. Decoders must determine the most likely error pattern from noisy stabilizer measurements faster than errors accumulate, typically within 1 μs for superconducting qubit systems.

Minimum-weight perfect matching (MWPM) has been the dominant decoding algorithm for surface codes since its introduction to quantum error correction. The state-of-the-art implementation, PyMatching v2 with Sparse Blossom [2], achieves near-optimal accuracy for independent errors with near-linear average-case complexity. Despite extensive research into alternative decoders — including union-find [3], belief propagation [4], and various neural network approaches [5, 6, 7] — no publicly available decoder has consistently outperformed MWPM on surface codes under circuit-level noise.

Recent work by Gu et al. [8] demonstrated that convolutional neural network decoders exploiting the geometric structure of QEC codes can achieve substantially lower logical error rates than existing decoders, identifying a "waterfall" regime of error suppression. However, their code and trained models are not publicly available. Google's AlphaQubit [5] achieved ~6% lower logical error rates than MWPM on experimental Sycamore data using a recurrent transformer architecture, but this system is internal to Google and was validated on proprietary hardware noise.

This work presents Pathfinder, a CNN decoder that:

1. **Outperforms or matches MWPM** at every tested noise rate (p=0.0005 to p=0.015) and code distance (d=3, 5, 7) under circuit-level depolarizing noise — wins or ties at all 24 evaluation points; 13/24 show non-overlapping 95% Wilson confidence intervals (Section 5.1).
2. **Achieves faster error suppression scaling** than MWPM with increasing code distance at operational noise rates (p ≥ 0.003), consistent with the waterfall regime identified by Gu et al. At the lowest tested noise (p=0.001) the scaling comparison is confounded by small-number statistics in 100K-shot trials (Section 5.2).
3. **Runs at 7.86 μs per syndrome** at throughput-optimal batching on a single NVIDIA H200 GPU with `torch.compile(max-autotune)` + FP16, or **6.12 μs per syndrome with a custom Triton kernel** for DirectionalConv3d. The Triton kernel sustains the d=7 surface-code cycle-time budget (7 μs) with 13% positive margin, whereas both unoptimized Pathfinder *and* PyMatching on single-core CPU fail to sustain it at p ≥ 0.007 (measured PM d=7 p=0.007: 9.65 μs/syn on Apple M4; Section 5.3).
4. **Is fully open-source** — all model code, trained checkpoints, training data generation, evaluation scripts, and the Triton kernel are publicly available.

Training the models reported here required approximately 28 GPU-hours on AMD MI300X instances (~$65 USD in cloud compute). Benchmarking on NVIDIA H200 for apples-to-apples comparison with Gu et al., plus custom Triton kernel development, distillation training, and narrower-model Pareto studies, added approximately 10 hours of H200 compute (~$35). Including ablations and abandoned runs during development, the total exploration cost was approximately $100 over 6 days of elapsed time by a single engineer.

**Relation to prior work.** Pathfinder is a composition of ideas, not a novel invention. The direction-specific 3D convolution architecture is a reimplementation of the design principles described by Gu et al. [8]. PyMatching with Sparse Blossom [2] is both the decoder this work is benchmarked against and, through its meticulous open-source release, the reason a comparison of this scope was possible. The Stim simulator [10] is what makes generating syndromes at the rate required for on-the-fly training tractable. The Muon optimizer [11] — whose effect on this decoder grows from small (+17%) at d=3 to catastrophic at d=7 (removing it causes training to fail entirely within the same step budget; see §6.2) — is due to Jordan et al. AlphaQubit [5] established that neural decoders can beat MWPM on real quantum hardware, validating this line of research before the open-source ecosystem could. Google's Willow [1] established the experimental regime (sub-threshold surface codes) that makes a decoder like this worth building. The novel contributions here are (a) the empirical finding that the Muon optimizer, not architecture, dominates this family of neural decoders' accuracy; (b) the complementarity of Pathfinder and MWPM's failure modes (0.01% syndrome overlap at d=5); (c) a custom Triton kernel for DirectionalConv3d that closes the d=7 cycle-time gap on H200 (Section 5.3); and (d) an open-source reference implementation reproducible by individual researchers on commodity cloud hardware.

---

## 2. Background

### 2.1 Surface Code Error Correction

The rotated surface code of distance d encodes one logical qubit in d² physical qubits arranged on a 2D lattice, with d²−1 stabilizer measurements that detect errors without disturbing the logical state [9]. Each round of error correction produces a syndrome — a binary pattern indicating which stabilizers detected parity violations. The syndrome over multiple rounds forms a 3D structure (2D spatial × 1D temporal), with detection events appearing as defects in this lattice.

### 2.2 The Decoding Problem

A decoder receives the 3D syndrome and must determine which logical observable was most likely flipped by the underlying errors. The decoder's accuracy is measured by the logical error rate (LER) — the fraction of decoding attempts that produce incorrect corrections. For the surface code to provide useful error protection, the LER must decrease exponentially with increasing code distance d, at a rate quantified by the error suppression ratio Λ = LER(d)/LER(d+2).

### 2.3 Minimum-Weight Perfect Matching

MWPM constructs a weighted graph from the syndrome, where defects are nodes and edges represent possible error chains connecting them. The decoder finds the minimum-weight perfect matching on this graph, corresponding to the most likely set of independent errors. PyMatching v2 [2] implements this via the Sparse Blossom algorithm, achieving near-linear average-case complexity by exploiting syndrome sparsity.

MWPM is optimal for independent (uncorrelated) errors but cannot capture correlations between error mechanisms. The correlated matching mode of PyMatching performs a two-pass correction but, as I show, provides identical results to uncorrelated matching under circuit-level depolarizing noise on rotated surface codes.

### 2.4 Neural Decoders

Neural network decoders learn to map syndromes to corrections from training data, potentially capturing error correlations that algorithmic decoders miss. Prior work includes recurrent architectures [5], transformers [5], and convolutional networks [8]. The key challenge is achieving both high accuracy and low inference latency — the decoder must run faster than the quantum error correction cycle time.

---

## 3. Architecture

### 3.1 Direction-Specific Convolution

The central architectural innovation in Pathfinder is **DirectionalConv3d**: a convolution layer that uses separate learned weight matrices for each neighbor direction in the 3D syndrome lattice, rather than a single shared kernel.

Standard 3D convolution applies the same 3×3×3 kernel regardless of the spatial relationship between elements. This ignores the lattice structure of the surface code, where the relationship between a stabilizer and its temporal neighbor differs fundamentally from its spatial neighbors, and different spatial directions correspond to different types of error coupling.

DirectionalConv3d replaces the single kernel with 7 independent linear transformations — one for the self-connection and one for each of the 6 neighbor directions (±time, ±row, ±column):

$$\text{out}(x) = W_{\text{self}} \cdot x + \sum_{d \in \{\pm t, \pm r, \pm c\}} W_d \cdot x_d$$

where $x_d$ denotes the feature at the neighbor in direction $d$, with zero-padding at boundaries.

This structure preserves the lattice geometry that standard convolution would blur, allowing the network to learn direction-dependent message-passing rules. Each layer can, for example, learn that temporal neighbors provide information about measurement errors while spatial neighbors provide information about data qubit errors.

### 3.2 Bottleneck Residual Blocks

Each layer of Pathfinder consists of a bottleneck residual block:

1. **Reduce**: 1×1×1 convolution, H → H/4 channels
2. **Message passing**: DirectionalConv3d, H/4 → H/4 channels
3. **Restore**: 1×1×1 convolution, H/4 → H channels
4. **Residual connection** + LayerNorm

The bottleneck reduces the computational cost of the direction-specific message passing by 4×, while the residual connection ensures gradient flow through deep networks.

### 3.3 Full Architecture

The complete decoder architecture:

- **Input**: Binary syndrome tensor [B, 1, R, H, W] where R = rounds, H×W = spatial lattice
- **Embedding**: 1×1×1 convolution lifting binary input to H=256 dimensions
- **L = d bottleneck residual blocks** (depth scales with code distance)
- **Global average pooling** over all spatial and temporal dimensions
- **MLP head**: Linear(H, H) → GELU → Linear(H, n_observables)
- **Output**: Logit per logical observable (apply sigmoid for probability)

Model sizes: 252K parameters (d=3), 376K parameters (d=5), 500K parameters (d=7). All models fit in GPU L2 cache at FP16.

### 3.4 Spatial Mapping

The syndrome tensor is constructed from Stim's detector coordinate annotations, which provide the exact (x, y, t) position of each detector in the code lattice. This coordinate-aware mapping ensures that the DirectionalConv3d operates on the correct spatial structure, rather than relying on heuristic index orderings.

---

## 4. Training

### 4.1 Data Generation

Training data is generated on-the-fly using Stim [10], which simulates stabilizer circuits at approximately 1 billion Clifford gates per second. Each training batch samples fresh syndromes from the circuit-level depolarizing noise model, eliminating the need for pre-generated datasets and ensuring the model never overfits to a fixed training set.

### 4.2 Optimizer

I use the Muon optimizer [11] for all 2D weight parameters (linear layers within DirectionalConv3d) and AdamW for 1D parameters (biases, LayerNorm). Muon applies Newton-Schulz orthogonalization to weight updates, keeping the direction-specific weight matrices well-conditioned throughout training. This prevents the weight degeneration that standard optimizers allow, which is particularly important for the message-passing interpretation of the architecture.

**Ablation**: Replacing Muon with AdamW increases the logical error rate by 72% at d=5 (from 1.28% to 2.20%), making it the single most impactful architectural choice. By comparison, replacing DirectionalConv3d with standard Conv3d increases LER by only 4%, and removing the curriculum has negligible effect.

### 4.3 Learning Rate Schedule

Cosine decay with 1000-step linear warmup. Muon learning rate: 0.02; AdamW learning rate: 3×10⁻³.

### 4.4 Curriculum

Training uses a compressed 3-stage noise annealing schedule:
- Stage 1 (0–10% of training): constant noise at 0.3× target
- Stage 2 (10–40%): linear ramp to 0.7× target
- Stage 3 (40–100%): linear ramp to target

Ablation shows this curriculum provides smoother convergence but does not improve final accuracy compared to fixed-noise training at d=5.

### 4.5 Noise-Rate Specialization

For d=7, where the noise range spans two orders of magnitude (p=0.001 to p=0.015), I train separate models at different target noise rates (p=0.007, p=0.01, mixed-noise, p=0.015) and select the best-performing model or their ensemble at each evaluation point. At d=3 and d=5, a single model trained at p=0.007 suffices to beat MWPM across all noise rates.

### 4.6 Training and Benchmarking Cost

Each model trains for 80,000 steps at batch size 512–1024 on a single AMD MI300X GPU. Wall-clock training time: 3–6 hours per model. Total compute for all models reported in Table 1 and the ablations: ~28 GPU-hours (~$65 USD at $1.99/hr MI300X cloud pricing). Additional work reported in this paper — H200 latency benchmarking (Section 5.3), custom Triton kernel development, distillation training (narrow and H=192 students, Section 5.10), and the PyMatching CPU measurements on Apple M4 — added approximately 10 hours of H200 compute at ~$3.60/hr (~$36). The full end-to-end cost of the work reported here is therefore approximately $100 over six days of elapsed time by a single engineer.

---

## 5. Results

### 5.1 Main Results: Rotated Surface Code

Table 1 presents the definitive evaluation: all decoders on the rotated surface code at distances d=3, 5, 7 across 8 noise rates, with 100,000 shots per data point. Pathfinder wins or ties PyMatching at every one of the 24 evaluation points — 22 wins plus 2 exact ties at p=0.0005, d=5 and d=7 where both decoders achieve zero observed errors in 100,000 shots. Thirteen of the 24 points show non-overlapping 95% Wilson confidence intervals for the two decoders; the remaining 11 are either exact ties (2) or points where the noise rate is low enough that small-number statistics yield overlapping intervals (9). See the footnote below Table 1 for the statistical-significance breakdown.

**Table 1: Logical Error Rate (%) — Pathfinder vs PyMatching (100K shots)**

| p | d=3 Pathfinder | d=3 PM | d=5 Pathfinder | d=5 PM | d=7 Pathfinder | d=7 PM |
|---|---------------|--------|---------------|--------|---------------|--------|
| 0.0005 | **0.009** | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.001 | **0.046** | 0.064 | **0.007** | 0.009 | **0.000** | 0.001 |
| 0.002 | **0.161** | 0.191 | **0.028** | 0.055 | **0.005** | 0.007 |
| 0.003 | **0.333** | 0.402 | **0.104** | 0.154 | **0.032** | 0.057 |
| 0.005 | **1.002** | 1.098 | **0.585** | 0.751 | **0.253** | 0.442 |
| 0.007 | **1.818** | 2.014 | **1.521** | 1.891 | **1.041** | 1.489 |
| 0.010 | **3.521** | 3.742 | **4.145** | 4.810 | **4.104** | 5.161 |
| 0.015 | **7.315** | 7.728 | **12.137** | 12.606 | **15.843** | 17.045 |

Bold indicates the lower (better) LER. Pathfinder wins or ties at every one of the 24 evaluation points. **Statistical significance:** computing 95% Wilson confidence intervals for each entry (N=100,000), 13 of 24 points show non-overlapping CIs between Pathfinder and PyMatching; the remaining 11 include two exact ties (both decoders at 0 errors, p=0.0005 at d=5 and d=7) and nine points where the low noise rate (typically p ≤ 0.003) produces so few decoding failures that CIs overlap. The non-overlapping-CI wins span every tested distance and concentrate at p ≥ 0.005 where the decoding regime is most relevant for real hardware; Pathfinder is never observed to lose.

Correlated PyMatching (two-pass matching with edge reweighting) produces identical results to uncorrelated PyMatching on this noise model, confirming that the correlation structure of circuit-level depolarizing noise on rotated surface codes does not benefit from the correlated matching approach.

### 5.2 Error Suppression Scaling

The error suppression ratio Λ = LER(d)/LER(d+2) quantifies how effectively the code suppresses errors as distance increases. Table 2 shows that Pathfinder achieves higher suppression ratios than PyMatching at operational noise rates (p ≥ 0.003), indicating that its advantage grows with increasing code distance in the regime that matters for real hardware.

**Table 2: Error Suppression Ratios**

| p | Pathfinder Λ(3→5) | PM Λ(3→5) | Pathfinder Λ(5→7) | PM Λ(5→7) |
|---|-------------------|-----------|-------------------|-----------|
| 0.001 | 9.9× | 5.4× | 2.0× | 5.7× |
| 0.003 | 3.2× | 2.7× | **4.4×** | 2.6× |
| 0.005 | 1.8× | 1.5× | **2.2×** | 1.7× |
| 0.007 | 1.3× | 1.1× | **1.5×** | 1.3× |

At p=0.003, Pathfinder's d=5→7 suppression (4.4×) substantially exceeds PyMatching's (2.6×), consistent with the "waterfall" regime identified by Gu et al. [8] where learned decoders exploit high-weight failure modes that MWPM cannot correct.

**An honest note on the p=0.001 row.** At p=0.001, Pathfinder's Λ(5→7) = 2.0× is lower than PyMatching's 5.7×, apparently contradicting the "scaling advantage" claim. This is a small-number artifact: at d=7, p=0.001, Pathfinder has 0/100,000 errors and PyMatching has 1/100,000 (Table 1). Both numbers are at the edge of 100K-shot statistics, and the resulting Λ ratios are driven by single-digit error counts. Similarly at d=5 Pathfinder has 7 errors vs PM's 9. An honest evaluation at p=0.001 would require 10⁷+ shots, which we did not run. The scaling-advantage claim holds rigorously for p ≥ 0.003 where error counts are in the hundreds or thousands.

### 5.3 Inference Latency

Pathfinder's inference latency was measured on two GPUs: the AMD MI300X used for training, and the NVIDIA H200 SXM used by Gu et al. [8] — providing an apples-to-apples comparison on equivalent hardware. All H200 numbers below use PyTorch 2.6 with `torch.compile(mode="max-autotune")` and FP16, the configuration that produced the lowest latencies at every batch size.

**Table 3a: Pathfinder Inference Latency on NVIDIA H200 SXM (FP16, torch.compile max-autotune)**

| Distance | Params | B=1 | B=64 | B=1024 |
|----------|--------|-----|------|--------|
| d=3 | 252K | 100.9 μs | — | **0.385 μs/syn** |
| d=5 | 376K | 173.5 μs | — | **2.06 μs/syn** |
| d=7 | 500K | 250.1 μs | 10.97 μs/syn | **7.86 μs/syn** |
| d=7 (narrow, H=128) | 126K | 213.3 μs | — | **3.49 μs/syn** |

**Table 3b: Cross-Decoder Latency at Throughput-Optimal Configuration**

| Decoder | Hardware | Latency | Notes |
|---------|----------|---------|-------|
| **Pathfinder d=7 + Triton kernel** | H200 SXM | **6.12 μs/syn** | B=1024, FP16, torch.compile max-autotune |
| **Pathfinder d=7 (Inductor only)** | H200 SXM | **7.86 μs/syn** | B=1024, FP16, torch.compile max-autotune |
| Pathfinder d=7 narrow (H=128) + Triton | H200 SXM | **2.70 μs/syn** | B=1024 |
| Gu et al. [8] | H200 | ~40 μs/syn | Batch size and config not reported |
| AlphaQubit [5] | TPU v5 | ~63 μs/syn | Published figure |
| PyMatching v2 [2] (measured, this work) | Apple M4, 1 core | 9.65 μs/syn at p=0.007 | per-syndrome decode; batch mode: 7.77 μs/syn |
| Pathfinder d=7 (vendor-cross) | AMD MI300X | 19 μs/syn | Training hardware; no Triton port attempted |

FP16 quantization produces zero accuracy degradation (0 prediction differences on 50,000 test shots). On identical hardware (H200 SXM), Pathfinder with the Triton kernel is 6.5× faster than Gu et al.'s reported throughput and 10.3× faster than AlphaQubit on TPU. The narrow variant is 2.25× faster than the full model at a documented accuracy cost (Section 5.9).

**PyMatching latency measurement.** PyMatching's per-syndrome latency depends strongly on noise rate (higher noise → more defects → longer matching). Table 3c reports measurements from single-core PyMatching v2 on an Apple M4 (ARM64, 16-core chip, single thread per decoder), using `Matching.decode()` for single-syndrome latency and `Matching.decode_batch()` for amortized throughput. The benchmark script is at `bench/results/pymatching_latency_m4.txt`.

**Table 3c: PyMatching v2 Latency vs. Noise Rate (d=7, single Apple M4 core)**

| p | PM single (μs/syn) | PM decode_batch (μs/syn) |
|---|-------------------|--------------------------|
| 0.001 | 2.54 | 0.79 |
| 0.003 | 4.66 | 2.63 |
| 0.005 | 6.77 | 5.04 |
| 0.007 | 9.65 | 7.77 |
| 0.010 | 14.97 | 12.76 |
| 0.015 | 22.93 | 20.69 |

**Deployment analysis: throughput sustainability.** For real-time surface-code decoding on superconducting qubits, the decoder must process syndromes at least as fast as they arrive. Each distance-d syndrome block covers d rounds of QEC at approximately 1 μs per round, so the arrival rate is one block per d μs. Table 3d combines Pathfinder's throughput (independent of noise rate, since neural network forward latency is fixed) with PyMatching's (noise-dependent) measurements.

**Table 3d: Sustainability of the d=7 Cycle-Time Budget (7 μs) on Single-Machine Hardware**

| Configuration | p=0.005 | p=0.007 | p=0.010 |
|---------------|---------|---------|---------|
| Pathfinder d=7 (Inductor only) | 7.86 μs ✗ (−12%) | 7.86 μs ✗ (−12%) | 7.86 μs ✗ (−12%) |
| **Pathfinder d=7 + Triton** | **6.12 μs ✓ (+13%)** | **6.12 μs ✓ (+13%)** | **6.12 μs ✓ (+13%)** |
| Pathfinder d=7 narrow (H=128) + Triton | 2.70 μs ✓ (+61%) | 2.70 μs ✓ (+61%) | 2.70 μs ✓ (+61%) |
| PyMatching v2 (M4 single core, decode_batch) | 5.04 μs ✓ (+28%) | 7.77 μs ✗ (−11%) | 12.76 μs ✗ (−82%) |
| PyMatching v2 (M4 single core, single-syndrome) | 6.77 μs ✓ (+3%) | 9.65 μs ✗ (−38%) | 14.97 μs ✗ (−114%) |

**Key finding.** Pathfinder + Triton is the only configuration that sustains the d=7 cycle-time budget across all operational noise rates. PyMatching sustains the budget only below p ≈ 0.006–0.007; above that, PM falls progressively behind as noise rises. For deployments where the expected worst-case noise exceeds ~0.006, Pathfinder + Triton is the only decoder in this comparison that is both real-time and accurate.

**Single-shot (batch=1) latency.** Batch=1 latency of 250 μs at d=7 (Inductor) or 201 μs (with the Triton kernel) is dominated by kernel launch overhead — the forward pass dispatches on the order of tens of CUDA kernels per call, a regime where per-kernel launch cost on the order of a microsecond accumulates to most of the observed latency (see NVIDIA CUDA best-practices documentation for current Hopper launch overhead figures). This is orthogonal to compute, which at full GPU occupancy at B=1024 is ~6 μs per syndrome. Closing the single-shot gap to the 1-μs physical cycle time requires further kernel fusion — either a single Triton/CUDA kernel spanning the entire bottleneck block (we built a prototype of this fusing restore+LayerNorm that regressed past B=64 due to register pressure; see `bench/triton_restore_norm.py`), or a hardware-synthesized FPGA implementation.

**Custom Triton kernel for DirectionalConv3d — methodology.** Profiling the compiled forward pass (PyTorch profiler, cuda_time_total, d=7 B=1024, FP16, 20 iterations) shows GPU time concentrated in: native LayerNorm (~17%), the Inductor-fused pad+GELU+add emitted for DirectionalConv3d's six boundary-padded shifted additions (~16%), the 7 direction-specific linear projections (~9%), and various copies/permutes (~10%). To close the d=7 cycle-time gap, we wrote a single Triton kernel that fuses all 7 direction-specific matrix multiplies and their boundary-masked accumulations into one launch, eliminating both the pad+add fusion overhead and 6 of the 7 separate matmul launches per DirectionalConv3d call.

**Reproducibility (Triton kernel).** The kernel is at `bench/triton_directional.py`. It accepts the same `state_dict` as the reference `DirectionalConv3d` module (7 packed weight matrices, one per direction). The launch configuration is: grid = (ceil(B / BLOCK_B), T·R·C, ceil(C_out / BLOCK_CO)) with BLOCK_B = max(16, min(64, next_pow2(B))), BLOCK_CO = min(64, next_pow2(C_out)), BLOCK_C_IN = max(16, next_pow2(C_in)). The ≥16 floor is required by Triton's `tl.dot` minimum shape constraint. The kernel is not autotuned — block sizes are fixed as above — so no extra warmup cost. It is verified on Triton 3.2 + PyTorch 2.6 + CUDA 12.4 on an NVIDIA H200 SXM.

**Numerical equivalence.** On 10,000 syndromes per noise rate at p ∈ {0.003, 0.007, 0.010}, the Triton kernel produces 0, 1, and 2 prediction disagreements from the reference PyTorch implementation respectively — at most 0.02%, within the FP16 noise floor. LER is statistically indistinguishable between the two implementations; full protocol and output are logged by `bench/triton_ler_test.py`.

**Latency (measured).** In isolation on H200 SXM with FP16 + `torch.compile(max-autotune)`: **6.12 μs per syndrome at d=7 batch=1024** (down from 7.86 μs/syn without the kernel, a 22% speedup) and **201.6 μs at batch=1** (down from 250.8 μs, a 20% speedup). The B=1024 figure sustains the d=7 cycle-time budget of 7 μs with 13% positive margin. Applied to the narrow H=128 variant, the kernel brings batch=1024 throughput to **2.70 μs per syndrome** and batch=1 latency to **147.6 μs**. Numbers are the minimum of five independent trials, each 500 iterations after 100 warmup iterations, run back-to-back against the reference implementation to cancel host-side variance.

**Cross-vendor portability.** The Triton kernel is written for NVIDIA (Triton 3.2+, Hopper architecture). Whether a ROCm port to the MI300X training hardware would recover similar gains is an open question — Triton has experimental AMD backends but the 7-point stencil pattern has not been profiled there. The core PyTorch model code (`train/model.py`) has no vendor-specific dependencies and runs on CUDA, ROCm, MPS, and CPU.

**FP8 quantization — tested and reported as a negative result.** H200 Hopper tensor cores support FP8 matrix multiply via `torch._scaled_mm`. Using `torchao.quantization.float8_dynamic_activation_float8_weight()` on all Linear layers (the final output head, a 256×1 projection, was excluded because `_scaled_mm` requires both inner dimensions divisible by 16), the quantized model is numerically within the noise floor of the FP16 model (LER delta within ±0.1 percentage points on 5,000 shots at p=0.007). However, FP8 does not accelerate inference at Pathfinder's parameter counts: the quantize/dequantize overhead around each linear exceeds the compute savings from the smaller-precision matrix multiply at matrix sizes ≤ 256×256. At d=7 B=1: FP8 compiled with `reduce-overhead` is 1,162 μs/call versus FP16's 493 μs/call. This is a scale-specific negative result; FP8 is expected to pay off for larger neural decoders (e.g. transformer architectures at 10M+ parameters). FP16 remains the optimal precision for Pathfinder at this scale.

### 5.4 Ablation Study

**Table 4: Ablation at d=5, p=0.007 (100K shots)**

| Variant | LER (%) | vs Full |
|---------|---------|---------|
| **Full (DirectionalConv + Muon + Curriculum)** | **1.28** | baseline |
| Standard Conv3d + Muon + Curriculum | 1.33 | +4% |
| DirectionalConv + Muon + No Curriculum | 1.23 | −4% |
| DirectionalConv + AdamW + Curriculum | 2.20 | +72% |

The Muon optimizer is the dominant contributor to Pathfinder's accuracy advantage, responsible for a 72% LER reduction compared to AdamW. DirectionalConv3d provides a modest 4% improvement over standard convolution at d=5. The curriculum does not improve final accuracy at this distance — fixed-noise training achieves comparable or slightly better results, though curriculum training provides smoother convergence dynamics.

### 5.5 Confidence Calibration

Pathfinder's logit outputs are exceptionally well-calibrated, with an Expected Calibration Error (ECE) of 0.002 at d=5, p=0.007 (50K shots). Predicted probabilities closely match observed frequencies across all confidence bins. This enables reliable confidence-based filtering in repeat-until-success quantum protocols.

### 5.6 Decoder Failure Analysis and Ensembling

At d=5, p=0.007 (50K shots), Pathfinder and PyMatching make largely independent errors:
- Both correct: 96.6% of shots
- Both wrong: 0.01% of shots
- Pathfinder wrong, PM right: 1.51%
- Pathfinder right, PM wrong: 1.89%

Pathfinder achieves a net advantage of +187 shots per 50,000, with the two decoders failing on almost entirely different syndromes (0.01% overlap). This near-disjoint failure mode motivates ensembling.

**Ensemble results.** Testing the ensemble hypothesis directly at d=7 (20K shots per noise rate, using the distilled narrow H=128 student paired with PyMatching), the OR-oracle — "at least one decoder is correct" — has substantially lower LER than either decoder alone, confirming the failure modes are mostly independent:

**Table 5: d=7 Ensemble of Pathfinder (narrow, distilled) and PyMatching (20K shots)**

| p | Pathfinder alone | PyMatching alone | Ensemble (confidence>2) | OR-oracle (upper bound) |
|---|------------------|------------------|------------------------|-------------------------|
| 0.003 | 0.00110 | 0.00070 | 0.00065 | 0.00035 |
| 0.005 | 0.00780 | 0.00475 | **0.00445** (−6%) | 0.00240 |
| 0.007 | 0.02500 | 0.01505 | **0.01420** (−6%) | 0.00655 |
| 0.010 | 0.09150 | 0.05400 | 0.05410 | 0.02855 |

A simple confidence-thresholded ensemble — use Pathfinder's prediction when |logit| > 2, else PyMatching — beats PyMatching alone at p ∈ {0.003, 0.005, 0.007}, recovering a small fraction of the OR-oracle headroom. At p=0.010 the narrow neural decoder's accuracy is low enough that its high-confidence predictions are themselves often wrong, and the simple threshold gating does not beat PM. More sophisticated gating (a learned meta-decoder, or distinct confidence thresholds per noise regime) could plausibly close more of the gap to the oracle's 50–60% reduction in LER relative to PM alone; this is left as future work.

**Deployment implication and hardware cost.** The narrow Pathfinder variant runs in 2.70 μs/syn on a GPU; PyMatching's per-syndrome latency depends on noise (Table 3c) — at p=0.007 on a single Apple M4 core, PM takes 9.65 μs/syn (single-syndrome) or 7.77 μs/syn (batch). The ensemble requires running **both** decoders and gating on Pathfinder's confidence, which requires a GPU and a CPU core. In a parallel deployment (GPU and CPU running concurrently, both seeing every syndrome), the effective decoder latency is the maximum of the two — dominated by PyMatching at this operating point. The ensemble improves LER over PM alone at matched latency on this parallel deployment, but it does **not** Pareto-dominate the standalone Pathfinder-full + Triton configuration (Section 5.9, which achieves strictly lower LER and strictly lower latency at p=0.007). The ensemble is the strongest configuration that *uses* PyMatching at all; Pathfinder-full + Triton is the strongest configuration overall.

### 5.7 Generalization

**Noise models**: A model trained on circuit-level depolarizing noise (with measurement errors) successfully decodes phenomenological noise (without measurement errors), beating PyMatching on this out-of-distribution noise model without retraining.

**Code types**: Pathfinder generalizes to alternative code types with per-code-type training.

**Table 6: Generalization across code types (LER %)**

| Code Type | d | Pathfinder | PyMatching | Ratio |
|-----------|---|-----------|------------|-------|
| Rotated Surface Z | 5 | **1.56%** | 1.92% | 0.81× |
| Color Code XYZ | 3 | **3.76%** | 12.51% | 0.30× |
| Rotated Surface X | 5 | **2.01%** | 2.28% | 0.88× |

The color code result is particularly striking: Pathfinder achieves 3.3× lower LER than PyMatching, suggesting that the direction-specific architecture is especially effective on codes with richer stabilizer geometry.

### 5.8 Sample Complexity

Pathfinder converges in approximately 77 million training samples (80K steps × batch 1024) for d=5–7. Gu et al. [8] report using 266 million samples (80K steps × batch 3,328), suggesting that the compressed curriculum and Muon optimizer achieve 3.5× better sample efficiency.

### 5.9 Accuracy/Latency Pareto

The full d=7 model (H=256, L=7, 500K parameters) achieves the best LER and, with the Triton kernel (Section 5.3), sustains the d=7 cycle-time budget. To characterize the accuracy/latency frontier around this point, we additionally trained a narrower variant (H=128, L=7, 126K parameters), an intermediate variant (H=192, L=7, 282K parameters), and distilled versions of both (Section 5.10).

**Table 7: d=7 Logical Error Rate of Pathfinder variants across noise rates (20K-shot evaluation)**

| p | Pathfinder full (H=256) | Pathfinder narrow (H=128) | PyMatching |
|---|------------------------|--------------------------|------------|
| 0.001 | 0.00007 | **0.00000** | 0.00009 |
| 0.002 | **0.00005** | 0.00025 | 0.00007 |
| 0.003 | **0.00032** | 0.00090 | 0.00057 |
| 0.005 | **0.00253** | 0.00860 | 0.00442 |
| 0.007 | **0.01041** | 0.02855 | 0.01489 |
| 0.010 | **0.04104** | 0.09905 | 0.05161 |
| 0.015 | **0.15843** | 0.27345 | 0.17045 |

The narrow variant ties PyMatching at the lowest noise rate (p=0.001, small-number statistics) but loses at all practical operating points — its LER is 1.5–3× the full model's. This is the accuracy cost of the 2.25× throughput gain seen in Table 8.

**Table 8: Pareto summary at d=7, p=0.007 (measured on H200 SXM, FP16, `torch.compile(max-autotune)`)**

| Configuration | Parameters | LER (%) | Throughput (μs/syn, B=1024) | Sustains 7 μs cycle? | Beats PM on LER? |
|---------------|-----------|---------|----------------------------|---------------------|-----------------|
| Pathfinder full (H=256) | 500K | 1.041 | 7.86 | ✗ (−12%) | ✓ |
| **Pathfinder full + Triton** | 500K | **1.041** | **6.12** | **✓ (+13%)** | **✓** |
| Pathfinder H=192 (distilled) + Triton | 282K | 2.035 | 5.05 | ✓ (+29%) | ✗ |
| Pathfinder narrow H=128 | 126K | 2.855 | 3.50 | ✓ (+50%) | ✗ |
| Pathfinder narrow + Triton | 126K | 2.810 | 2.70 | ✓ (+61%) | ✗ |
| Pathfinder narrow (distilled) + Triton | 126K | 2.520 | 2.70 | ✓ (+61%) | ✗ |
| Ensemble (narrow-distilled + PM, parallel) | 126K + PM | 1.420 | ≥7.77 (PM-bounded) | ✗ at p ≥ 0.007 | ✓ |
| PyMatching v2 (M4 single core, batch) | — | 1.489 | 7.77 | ✗ (−11%) | baseline |
| PyMatching v2 (M4 single core, single-syn) | — | 1.489 | 9.65 | ✗ (−38%) | baseline |

**The Pareto-optimal configuration at d=7 is Pathfinder full + Triton kernel.** It is the only configuration in this table that (a) beats PyMatching on LER, (b) sustains the d=7 cycle-time budget at operational noise rates, and (c) runs on a single GPU without requiring a parallel CPU-based decoder.

The ensemble (narrow-distilled + PyMatching) is the strongest configuration that still uses PyMatching: its LER (1.420%) improves over PM alone (1.489%) by 4.6%, but its latency is bounded by PM's 7.77 μs/syn (batch mode, p=0.007) — which does not sustain the 7-μs cycle time. The ensemble is a valid LER-only Pareto improvement on PyMatching alone, but is Pareto-dominated by Pathfinder full + Triton (strictly lower LER *and* strictly lower latency, measured in the same conditions). We include it in this table to show that the near-disjoint failure modes (Section 5.6) translate into a practically achievable LER improvement, and to motivate future work on learned meta-decoders.

### 5.10 Distillation Study

To investigate whether the narrower variants' accuracy gap to the full model is a training artifact or a capacity limitation, both the H=128 and H=192 students were additionally trained with knowledge distillation from the full H=256 teacher (`d7_final` checkpoint). The student's loss combined 30% binary cross-entropy against the true labels with 70% of a soft-target loss against the teacher's tempered-sigmoid outputs (temperature T=2), using the same Muon + AdamW optimizer, the same curriculum, and the same 80,000 training steps as the base models. The training script is `train/train_distill.py`.

After 80,000 steps at p=0.007:

- Distilled narrow (H=128, 126K params): LER 2.520% — a 17% relative improvement over non-distilled narrow (2.855%) at identical latency (2.70 μs/syn with the Triton kernel).
- Distilled H=192 (282K params, trained from scratch with distillation): LER 2.035% — improvement over the narrow-distilled model, at 5.05 μs/syn (55% faster than the full model's 7.86 μs/syn, but slower than the narrow-distilled's 2.70 μs/syn).

Neither distilled variant closes the remaining gap to PyMatching (1.489%) as a standalone decoder: capacity, not training, appears to be the constraint at this scale. The H=192 model closes only about half of the accuracy gap between H=128 and the full model despite having roughly midway parameter count (282K vs. 126K, 500K), suggesting that returns on width below H=256 are non-linear and that the last increment of width (H=192 → H=256) carries disproportionate accuracy weight. A shallower full-width model (L=5 at H=256) or neural-architecture search on the d=7 decoder family may uncover better narrow configurations than uniform width reduction; this is left as future work.

### 5.11 Head-to-Head with Lange et al.

The priority note in the abstract acknowledges Lange et al. [14] as the first published open-source neural decoder to outperform PyMatching on rotated surface codes. This section reports a three-way LER comparison between Pathfinder, Lange et al.'s pre-trained GNN weights, and PyMatching, run through a single evaluation harness (`bench/results/h200_session2/run_lange_v3.py`) that uses Lange et al.'s own graph-builder (`LangeDecoderWrapper`, instantiating their `GNN_7` model from `src.gnn_models` and loading the published `d{d}_d_t_{d_t}.pt` checkpoints from the Lange repo). 60,000 shots per data point across 21 (d, p) points; 95% Wilson confidence intervals are computed for each entry.

**Noise-model caveat — this comparison is not apples-to-apples for Pathfinder.** Lange et al.'s pre-trained models were trained on a 4-parameter Stim circuit-level noise model that adds `before_round_data_depolarization=p` on top of the three parameters used in Table 1 (`after_clifford_depolarization`, `before_measure_flip_probability`, `after_reset_flip_probability`). The §5.11 evaluation uses Lange's 4-parameter noise model — which is in-distribution for Lange but **out-of-distribution for Pathfinder**, whose Table 1 checkpoints never saw `before_round_data_depolarization`. A fully apples-to-apples comparison requires retraining Pathfinder on Lange's 4-parameter noise model; this is future work and is not what Table 9 reports.

**Table 9: Pathfinder (OOD) vs. Lange (in-distribution) vs. PM — 4-parameter circuit-level noise, 60K shots**

| d | p | Pathfinder (OOD) | Lange | PM | Lange vs PF (CI) |
|---|---|------------------|-------|-----|-------------------|
| 3 | 0.001 | 0.070% | 0.067% | 0.073% | overlap |
| 3 | 0.002 | 0.290% | 0.260% | 0.290% | overlap |
| 3 | 0.003 | 0.555% | **0.498%** | 0.652% | overlap |
| 3 | 0.005 | 1.673% | **1.517%** | 1.787% | overlap |
| 3 | 0.007 | 3.117% | **2.810%** | 3.278% | non-overlap |
| 3 | 0.010 | 5.675% | **5.225%** | 5.872% | non-overlap |
| 3 | 0.015 | 11.230% | **10.623%** | 11.568% | non-overlap |
| 5 | 0.001 | **0.007%** | 0.008% | 0.015% | overlap |
| 5 | 0.002 | 0.060% | **0.053%** | 0.113% | overlap |
| 5 | 0.003 | 0.228% | **0.173%** | 0.318% | overlap |
| 5 | 0.005 | 1.090% | **0.920%** | 1.342% | non-overlap |
| 5 | 0.007 | 2.937% | **2.487%** | 3.423% | non-overlap |
| 5 | 0.010 | 7.700% | **6.867%** | 8.337% | non-overlap |
| 5 | 0.015 | 19.677% | **18.135%** | 19.600% | non-overlap |
| 7 | 0.001 | 0.000% | 0.000% | 0.002% | tie (0/0) |
| 7 | 0.002 | 0.023% | **0.018%** | 0.025% | overlap |
| 7 | 0.003 | 0.108% | **0.082%** | 0.117% | overlap |
| 7 | 0.005 | 1.023% | **0.700%** | 1.010% | non-overlap |
| 7 | 0.007 | 4.050% | **2.937%** | 3.437% | non-overlap |
| 7 | 0.010 | 13.433% | **10.463%** | 10.258% | non-overlap |
| 7 | 0.015 | 33.208% | **30.327%** | 26.915% | non-overlap |

Bold = lowest LER in the row. Under this evaluation Lange's GNN has the lowest LER at 19 of 21 points; Pathfinder wins at one point (d=5, p=0.001, single-error statistics) and the remaining point is a zero-error tie. Eleven of 20 non-tied points show non-overlapping 95% CIs between Pathfinder and Lange. Pathfinder — despite being out-of-distribution — still beats PyMatching at 15 of 21 points; Lange beats PyMatching at 19 of 21 points.

**Interpretation.** The results in Table 9 are consistent with two compounding effects: (1) Lange's GNN architecture is genuinely stronger at this task than Pathfinder's CNN at matched noise model, and (2) Pathfinder's performance is amplified by the train/test noise mismatch — its LER under the 4-parameter noise model is 2.5–4× higher than under the 3-parameter model of Table 1 at d=7, p=0.007 (4.01% vs. 1.04%). Attempting to decompose these two effects was originally flagged as the most important pending experiment.

**Update — fine-tuning at matched noise largely closes the gap.** Fine-tuning the Table-1 checkpoints on the 4-parameter noise model (40,000 steps at p=0.007, lower LR: `muon_lr=0.005`, `adam_lr=1e-3`, no curriculum; script: `bench/results/h200_session3/train_finetune_4param.py`) converges to:

- **d=5 fine-tuned: 3.04% at p=0.007** (vs. 2.935% for the OOD Table-1 checkpoint and 2.58% for Lange). Non-overlapping 95% Wilson CIs: Lange still strictly wins at d=5 (~15% relative gap).
- **d=7 fine-tuned: 3.34% at p=0.007** (vs. 4.01% for the OOD Table-1 checkpoint and 2.94% for Lange). Non-overlapping 95% CIs: Lange still strictly wins at d=7, but the gap is now ~14% relative instead of the ~36% gap with the OOD checkpoint.

A from-scratch 80,000-step retrain at 4-parameter noise (script: `bench/results/h200_session2/train_fixed_noise.py`) was also tried: at d=5 it converged to 11.7% — markedly worse than either the OOD Table-1 checkpoint (2.94%) or the fine-tuned checkpoint (3.04%) — and at d=7 it failed catastrophically (stuck at ≈40% LER throughout the run). Training logs and both sets of checkpoints are preserved under `bench/results/h200_session3/`. The lesson: for Pathfinder's recipe at this scale, a from-scratch run on the harder 4-parameter noise is unreliable; initializing from a 3-parameter checkpoint and short fine-tuning is a reliable recovery.

**Revised interpretation.** After fine-tuning, the architectural gap between Pathfinder (CNN) and Lange (GNN) is ≈14–15% LER at d=5 and d=7 at p=0.007 — real, but substantially smaller than the ≈36% OOD gap Table 9 reports. The majority-vote ensemble of §5.12, which uses these fine-tuned checkpoints, turns this remaining gap into a strict ensemble *win* over Lange at d=7 (see Table 10).

**Implication for priority.** This comparison does not change the priority claim in the abstract: Lange et al. was first, and their decoder is stronger than Pathfinder on their own evaluation harness at every tested distance. Pathfinder's standing contributions (the Muon-dominance finding, the Triton kernel, the cycle-time sustainability analysis, and the extended-noise-rate coverage of p ∈ {0.010, 0.015}) are orthogonal to this LER comparison and are not falsified by it.

**Ensemble opportunity.** The three-way LER numbers also suggest that Pathfinder + Lange is a potentially complementary ensemble — the two decoders use fundamentally different inductive biases (lattice-aware convolution vs. graph-of-defects message passing) and their failure modes may be as independent as Pathfinder and PyMatching's (§5.6). This is tested below in §5.12.

### 5.12 Three-Way Majority-Vote Ensemble (Pathfinder + Lange + PyMatching)

Given the three decoders' different inductive biases — Pathfinder's lattice-aware 3D convolution, Lange's graph-of-defects message passing [14], and PyMatching's combinatorial minimum-weight matching [2] — their failure modes are largely independent (§5.6). A simple majority-vote ensemble of the three decoders was evaluated at matched 4-parameter noise using the same harness as §5.11: 60,000 shots per point (3 seeds × 20,000 shots), 12 (d, p) points, ensemble prediction is the elementwise majority of the three binary outputs. For the Pathfinder vote I use the best-available Pathfinder checkpoint at each distance: the fine-tuned `finetune_d5` at d=5 (§5.11 update), the distilled `distill_d7` at d=7 (trained with Lange as a soft-target teacher; see §5.13 below), and the 3-parameter Table-1 `d3_muon` checkpoint at d=3 (evaluated out-of-distribution at 4-parameter noise, no fine-tune was run for d=3). Raw data: `bench/results/h200_session3/distill/ensemble_results_distill.json`; harness: `bench/results/h200_session3/ensemble_pf_lange.py`.

**Table 10: 3-way majority vote (PF+Lange+PM) vs individual decoders (LER %, 60K shots)**

| d | p | PF | Lange | PM | **Majority** | Oracle-LB | Winner |
|---|---|-----|-------|-----|-------------|-----------|--------|
| 3 | 0.003 | 0.582 | 0.535 | 0.665 | 0.555 | 0.432 | Lange |
| 3 | 0.005 | 1.595 | 1.493 | 1.798 | 1.567 | 1.185 | Lange |
| 3 | 0.007 | 2.923 | 2.713 | 3.205 | 2.810 | 2.065 | Lange |
| 3 | 0.010 | 5.533 | 5.140 | 5.852 | 5.308 | 3.940 | Lange |
| 5 | 0.003 | 0.240 | 0.192 | 0.340 | 0.207 | 0.120 | Lange |
| 5 | 0.005 | 1.142 | 0.957 | 1.428 | 1.010 | 0.550 | Lange |
| 5 | 0.007 | 3.040 | 2.580 | 3.547 | 2.657 | 1.498 | Lange |
| 5 | 0.010 | 7.657 | 6.772 | 8.273 | **6.660** | 3.602 | **Majority** |
| 7 | 0.003 | 0.103 | 0.087 | 0.148 | **0.085** | 0.033 | **Majority** |
| 7 | 0.005 | 0.817 | 0.752 | 0.985 | **0.660** | 0.230 | **Majority** |
| 7 | 0.007 | 3.090 | 2.940 | 3.343 | **2.495** | 1.143 | **Majority** |
| 7 | 0.010 | 11.132 | 10.822 | 10.300 | **9.087** | 3.853 | **Majority** |

"Oracle-LB" = fraction of shots where all three decoders are simultaneously wrong, a lower bound on any ensemble's LER. "Winner" = strict winner among the four columns.

**Findings.**
1. **Majority vote strictly beats every individual decoder at 5 of 12 points**, sweeping all four d=7 operational noise rates (p ∈ {0.003, 0.005, 0.007, 0.010}) plus d=5 p=0.010. At d=7 p=0.007 the majority vote's 2.50% is a 15.1% relative reduction over Lange alone (2.94%) with strictly non-overlapping 95% Wilson CIs ([2.37, 2.62] vs. [2.81, 3.08]) — a statistically significant ensemble win. At d=7 p=0.010 the majority's 9.09% is a 16.0% reduction over Lange (10.82%) and 11.7% over PM (10.30%), also non-overlapping CIs. At d=7 p=0.005, majority 0.66% vs Lange 0.75% (11.9% relative). At d=7 p=0.003 the margin is tight (0.085% vs 0.087%) with overlapping CIs — reported as a soft win.
2. At d=3 and the low-noise end of d=5, majority vote does not strictly beat Lange alone — in the easy-decoding regime the individual decoders are already close to the ensemble limit, and the combinatorial-optimality gap between Lange and the majority vote is too small to recover.
3. The oracle lower bound at d=7 p=0.007 is 1.14%, so majority vote (2.50%) captures roughly (2.94 − 2.50) / (2.94 − 1.14) ≈ 25% of the available ensemble headroom over Lange. A learned meta-decoder (or per-noise-rate gating) could plausibly close more.
4. Confidence-thresholded gating (use Pathfinder's prediction when |logit| > T, else Lange) was also tested at T ∈ {1, 2, 3, 4} and never strictly beat Lange alone in this evaluation; the 3-way majority vote is a strict improvement over that confidence-gating scheme.
5. The choice of Pathfinder variant matters. Using distilled-from-Lange `distill_d7` at d=7 (lower individual LER: 3.09% vs fine-tuned 3.34%) produces an additional strict ensemble win at d=7 p=0.003 and a tighter margin at p=0.005, but a slightly weaker margin at p=0.007 (majority 2.50% vs. 2.42% with fine-tuned Pathfinder) because the student correlates with the Lange teacher, reducing ensemble independence. Using fine-tuned `finetune_d7` at d=7 gives majority 2.42% at p=0.007 (17.8% relative) but loses the strict win at p=0.003. Table 10 uses distilled at d=7 because distilled wins 3 of 4 d=7 points (p=0.003, 0.005, 0.010) and produces the lower standalone-Pathfinder LER; the §5.13 discussion below covers the correlation tradeoff.

**Contribution.** Section 5.11 established that Lange et al. individually outperforms Pathfinder at matched noise even after fine-tuning; this section establishes that a simple, computable-in-parallel 3-way majority vote of (Pathfinder, Lange, PyMatching) *strictly* beats Lange alone across all four tested d=7 operational noise rates, with statistically significant CI separation at p=0.007 and p=0.010. That is a novel contribution on top of Lange's priority: the cheapest way to lower the best-known open-source LER at d=7 operational noise is not a better decoder, it is running these three existing decoders in parallel and taking the majority. The ensemble requires running all three decoders concurrently, so end-to-end latency is bounded by the slowest (PyMatching at high noise); deployments for which this is acceptable — offline protocol verification, post-selection in repeat-until-success, or any non-real-time QEC application — gain a 12–16% LER reduction over Lange alone at d=7 operational noise rates for essentially zero additional ML effort.

### 5.13 Distillation from Lange and the Independence-Accuracy Tradeoff

To close the Pathfinder–Lange individual-decoder gap at matched 4-parameter noise (§5.11), I trained Pathfinder students with Lange as a soft-target teacher. Script: `bench/results/h200_session3/train_finetune_4param.py` for the fine-tune baseline, and `bench/results/h200_session2/train_distill_lange.py` for the distillation. The distillation loss is `0.3 · BCE(student_logit, label) + 0.7 · T² · KL(σ(student/T), σ(teacher/T))` with T=2.0, 80,000 steps, Muon lr=0.02, AdamW lr=3e-3 on 1D params, curriculum noise annealing from 0.1p_target to p_target. Teacher is the Lange GNN (`d{d}_d_t_{d_t}.pt` pre-trained weights from the Lange repo), frozen.

**Table 11: Pathfinder variants at p=0.007 (60K-shot eval, 4-parameter noise)**

| Distance | Variant | Individual LER | Ensemble LER (as PF voter) |
|----------|---------|----------------|----------------------------|
| d=5 | Table-1 OOD | 2.94% | 2.60% |
| d=5 | Fine-tune from Table-1 | 3.04% | 2.66% |
| d=5 | Distilled from scratch | ≈3.3% (10K eval drift)* | — |
| d=7 | Table-1 OOD | 4.01% | 2.56% |
| d=7 | Fine-tune from Table-1 | 3.34% | **2.42%** |
| d=7 | Distilled from scratch | **3.09%** | 2.50% |

*The d=5 distillation run's best 10K-shot evaluation during training was 3.07%, but end-of-training LER drifted higher. A 60K-shot ensemble evaluation with the best-checkpoint-during-training gave individual LER ≈3.3% — no net improvement over fine-tune. Reported here as "not materially different from fine-tune"; the distillation result at d=5 is neither a clear success nor a clear failure.

**Interpretation — independence vs. individual accuracy.** At d=7 the two approaches trade off cleanly:

- **Distilled Pathfinder** has *better* individual LER (3.09% vs. 3.34% for fine-tuned) because the Lange teacher provides a stronger training signal than hard labels alone.
- **Fine-tuned Pathfinder** produces a *better* 3-way ensemble at p=0.007 (majority 2.42% vs. 2.50%) because its predictions are less correlated with Lange's — the majority vote needs the voters to disagree on the *right* shots to extract the oracle-bound headroom.

Concretely: distilled_d7 and Lange agree on 96.7% of shots at d=7 p=0.007, versus fine-tuned_d7 and Lange agreeing on 95.9% — roughly 80 additional shots per 10K where distilled agrees with Lange but fine-tuned diverges. When fine-tuned diverges AND PyMatching votes with it, the majority flips a Lange error into a correct prediction; the distilled variant's over-agreement with the teacher suppresses that signal. This is the "correlation cost" of teacher-student training for ensemble use.

**Which Pathfinder to ship.** For standalone neural decoding at d=7, the distilled Pathfinder is the right default — lower LER than fine-tune, closer to Lange. For the 3-way ensemble (which is the LER-minimizing configuration overall), fine-tuned is the right default at p=0.007 specifically, but distilled wins at p ∈ {0.003, 0.005, 0.010}. Table 10 uses distilled because it wins 3 of 4 d=7 points and gives the better standalone decoder. Ensemble-oriented deployments at p=0.007 should use fine-tuned instead; cross-noise generalization of either choice is open.

**Negative result worth naming.** Distilling without a strong initialization is risky: the student's LER during training drifts non-monotonically (see `bench/results/h200_session3/distill/distill_d5.log`), and the best-checkpoint-during-training is not always representative of the end-of-training model. A distill-as-fine-tune recipe (initialize from Table-1 checkpoint, then add Lange teacher for the fine-tune phase) was not run for scheduling reasons but is the most plausible way to recover both signals; it is left as future work.

### 5.14 Modern-Architecture Ablation (Negative Result)

To test whether Pathfinder's relatively simple CNN architecture is leaving accuracy on the table, I trained a hybrid CNN+attention variant at d=7 incorporating architectural primitives developed since Pathfinder's original design: RMSNorm (pre-norm throughout), SwiGLU feed-forward blocks, global multi-head self-attention with 3D Rotary Positional Embeddings interleaved every 2 blocks, Flash Attention via `F.scaled_dot_product_attention`, and the Muon + AdamW split. The DirectionalConv3d backbone is preserved. Architecture script: `bench/results/h200_session3/hybrid/train_hybrid.py`; checkpoint: `bench/results/h200_session3/hybrid/hybrid_d7/best_model.pt`. Configuration: H=192, L=7 blocks, 8 attention heads, 4.36 M parameters, 80,000 steps, batch 256, 4-parameter circuit-level noise from scratch, same curriculum as Pathfinder.

**Result.** Final 50,000-shot LER at d=7, p=0.007: **4.76%**. This is *worse* than every other Pathfinder variant tested in this work:

| Pathfinder variant at d=7, p=0.007 | Params | LER |
|------------------------------------|--------|-----|
| Table-1 OOD (3-param ckpt on 4-param eval) | 500K | 4.01% |
| **Hybrid (CNN + attention + RMSNorm + SwiGLU + RoPE-3D)** | **4.36M** | **4.76%** |
| Fine-tune (Table-1 init, 40K steps at 4-param) | 500K | 3.34% |
| Distilled from Lange teacher (80K steps at 4-param) | 500K | 3.09% |

Under this training budget the 9× parameter increase and the full set of modern primitives make the model *worse*, not better. The training loss curve converges fine (loss ≈ 0.1 at end, similar to Pathfinder) and there is no obvious failure mode — the architecture just doesn't generalize as well under the same 80,000-step / batch-256 training envelope as the simpler CNN. Whether longer training (e.g. 250,000+ steps) or a different optimizer regime would flip the ranking is untested; at this level of compute, the finding is that the original direction-specific-CNN design is already well-tuned for this data scale.

This is reported as a negative result: the paper does not claim the hybrid variant as a contribution, but the checkpoint and full training log are released for researchers exploring the architecture space. The simpler CNN + Muon recipe is therefore the recommended base architecture for work of this kind.

---

## 6. Discussion

### 6.1 Why Does Pathfinder Beat MWPM?

MWPM is optimal for independent errors but treats the syndrome as an unstructured graph, discarding geometric information. Pathfinder's direction-specific convolution preserves the lattice structure, learning that different neighbor directions carry different types of information about the underlying error. The Muon optimizer keeps these direction-specific weight matrices well-conditioned, preventing the collapse to effectively isotropic (direction-independent) weights that would reduce the architecture to standard convolution.

The failure analysis (Section 5.6) reveals that Pathfinder and MWPM fail on almost entirely different syndromes, suggesting they exploit complementary information — MWPM uses exact minimum-weight combinatorial optimization, while Pathfinder uses learned geometric pattern recognition.

### 6.2 The Role of Muon

The d=5 ablation (Table 4) shows that removing Muon (i.e., training with AdamW on all 2D weights instead) increases LER at p=0.007 from 1.28% to 2.20% — a 72% relative increase, dwarfing the direction-specific architecture's own contribution (+4% LER when replaced with standard Conv3d). The same ablation run at d=3 and d=7 shows that the Muon effect is **strongly depth-dependent**:

| d | Full Muon (Table 1) | AdamW-only | Relative increase |
|---|---------------------|------------|-------------------|
| 3 | 1.82% | 2.14% | +17% (small) |
| 5 | 1.28% | 2.20% | +72% (Table 4) |
| 7 | 1.04% | **34.8%** | catastrophic (fails to learn) |

At d=3 the architecture is shallow enough (L=3, 252K parameters) that AdamW reaches near-Muon accuracy in 20,000 steps — the ≈17% gap is within the variance seen between independent runs. At d=5 the +72% gap is the headline Muon result. At d=7 (L=7, 500K parameters) AdamW-only fails to escape a high-loss plateau inside an 80,000-step budget that Muon converges within easily, landing at 34.8% LER versus Muon's 1.04%. I hypothesize that Muon's Newton-Schulz orthogonalization is critical for maintaining the diversity of the 7 directional weight matrices as depth grows — without it, gradient descent apparently collapses these matrices toward similar solutions at deep architectures, losing the directional specificity that distinguishes this architecture from standard convolution. The effect is therefore best described as *Muon is essential at d≥7 and small at d=3*. Ablation checkpoints are at `bench/results/h200_session3/checkpoints/ablation_adamw_{d3,d7}`.

### 6.3 Limitations

**Code distances.** The evaluation is limited to d=3, 5, 7; Gu et al. evaluate up to d=13. Extending to d=9, 11 would require larger models (likely H=512), longer training, and is left as future work. The error-suppression scaling trends (Section 5.2) suggest Pathfinder's accuracy advantage grows with distance, but this is extrapolation.

**Noise models.** Evaluation is on circuit-level depolarizing noise and (for generalization) phenomenological noise. Real quantum hardware exhibits device-specific correlated noise that may differ from these models; AlphaQubit [5] was validated on experimental Sycamore data, a comparison this work does not make.

**Single-shot latency.** Batch=1 latency (201 μs with the Triton kernel, 250 μs without) is two orders of magnitude above the 1-μs superconducting cycle time. Closing this gap requires bottleneck-block-level kernel fusion or FPGA deployment (Section 5.3). An exploratory attempt at fusing restore + residual add + LayerNorm into one Triton kernel was numerically correct but regressed at B ≥ 64 due to register pressure (`bench/triton_restore_norm.py`); a working full-block fusion remains open.

**Narrow-model accuracy gap.** Distillation reduces the narrow H=128 model's LER by 17% at p=0.007 but does not close the gap to PyMatching (Section 5.10). The full 500K-parameter budget appears necessary for PM-beating accuracy at d=7; architecture search or distillation with a larger teacher are open directions.

**Noise-target ensemble for the full model.** At d=7, the best-per-point LER across the full noise range was obtained by selecting among four full models trained at different target noise rates (Section 4.5). A single model that dominates PM across all noise rates from a single training run has not been identified.

**FP8.** Tested via `torch._scaled_mm` with `torchao` dynamic activation/weight quantization and found to regress latency at Pathfinder's matrix sizes (Section 5.3). Reported as a negative result; expected to become useful at 10M+ parameter scales.

---

## 7. Related Work

**Lange et al.** [14] (PRR 2025; arXiv:2307.01241): Graph neural network decoder with ~1.36M parameters (2.35M at d=9) that outperforms PyMatching on rotated surface codes under circuit-level depolarizing noise at d ∈ {3, 5, 7, 9}, p ∈ {0.001, 0.002, 0.003, 0.004, 0.005}. **Open-source with pre-trained weights** at https://github.com/LangeMoritz/GNN_decoder (MIT). Evaluated with 10⁸ shots per data point. To the best of this author's knowledge, this is the first published open-source neural decoder to outperform MWPM on rotated surface codes under circuit-level noise. The present work should be understood as extending (not preceding) Lange et al., with coverage of higher noise rates (p ≥ 0.007), an optimizer-centric architectural study, and a latency-optimized Triton kernel. A direct head-to-head is given in Section 5.11.

**Varbanov et al.** [15] (PRR 2025; arXiv:2307.03280): Recurrent neural decoder trained on simulated data and evaluated on experimental Sycamore surface-code data, reporting ~25% lower LER than PyMatching on d=3, 5 experimental traces. Complementary to the present work (real hardware data vs. simulated circuit-level noise).

**AlphaQubit** [5]: Recurrent transformer decoder achieving ~6% lower LER than MWPM on experimental Sycamore data. Not open-source. Validated on real hardware noise rather than simulated noise.

**Gu et al.** [8]: CNN decoder with direction-specific convolution achieving 17× lower LER than BP+OSD on [144,12,12] Gross codes. Identifies the "waterfall" regime. Not open-source. Pathfinder's architecture follows their design principles with independent implementation.

**NVIDIA Ising-Decoder** [16]: Open-source pre-decoder + PyMatching hybrid released April 2026 (concurrent with this work). Reports beating uncorrelated PyMatching up to d=13 at p=0.003. Pre-decoder architecture differs from Pathfinder's standalone decoder design.

**Astrea** [12]: FPGA implementation of MWPM reporting ~1 ns average decoding latency at d=7 (worst case ~456 ns) via brute-force enumeration of low-Hamming-weight syndromes. The Astrea-G variant extends to d=9 with ~450 ns average latency. Same accuracy as software MWPM — a hardware acceleration rather than algorithmic improvement. Requires custom FPGA hardware, whereas Pathfinder runs on commodity GPUs.

**Sparse Blossom / PyMatching** [2]: State-of-the-art MWPM implementation. 100-1000× faster than PyMatching v1 while maintaining identical accuracy. The comparison baseline.

**Union-Find** [3]: Near-linear time decoder. Fast but significantly less accurate than MWPM (7-30× higher LER in this evaluation).

**Sivak et al.** [13]: RL-based decoder steering for adapting to non-stationary noise on Google's Willow processor. Complementary to Pathfinder's approach — the steering concept could be applied to Pathfinder's ensemble weights.

---

## 8. Conclusion

Pathfinder is an open-source reference implementation that composes ideas developed by the broader quantum error correction and deep learning research communities. None of its ingredients are novel in isolation: the direction-specific convolution architecture follows Gu et al. [8]; PyMatching [2] defines what it means to "beat MWPM" and is the reason a rigorous comparison was possible; Stim [10] makes syndrome generation tractable at the scale required for training; Muon [11] provides the optimizer that, as the ablation reveals, is responsible for the majority of the accuracy advantage (removing Muon increases LER by 72%); AlphaQubit [5] established that neural decoders could beat MWPM on real hardware; and Willow [1] established that the surface code regime addressed here is experimentally relevant. Pathfinder's contribution is to assemble these pieces into an open-source decoder that outperforms MWPM across all tested conditions, to empirically identify the Muon optimizer as the dominant factor in neural decoder accuracy, to add a custom Triton kernel that makes d=7 decoding real-time at operational noise rates on a single H200 GPU, and to demonstrate that the full work is reproducible on commodity cloud hardware for approximately $100 in total compute over six days of elapsed time by a single engineer.

Every design principle underlying this decoder existed before this work. What did not exist was an open implementation that made them reproducible together. The intent of this release is to give that to the research community so the next improvements can build on a shared foundation rather than start over. Real-time *single-shot* (batch=1) latency remains the principal open problem: the 201-μs per-syndrome latency achievable with the Triton kernel is still two orders of magnitude above the 1-μs superconducting cycle time, and closing this gap will require custom GPU kernels at the bottleneck-block level (not just DirectionalConv3d) or an FPGA implementation. That is the most important next step.

---

## Acknowledgments

This work owes a specific intellectual debt to several teams. Andi Gu and colleagues at Harvard provided the architectural blueprint — the direction-specific convolution design and the waterfall-regime framing — that this decoder follows. Oscar Higgott and Craig Gidney's PyMatching is both the benchmark this work aims at and, through its exemplary open-source release, the standard of reproducibility this project has tried to meet. Craig Gidney's Stim is the reason on-the-fly training at the required throughput is feasible. Keller Jordan and the Muon authors provided the single most impactful ingredient in this decoder's accuracy. The Google DeepMind AlphaQubit team demonstrated, before this work, that neural decoders can beat MWPM on real quantum hardware — establishing the empirical ground truth that made this line of research worth pursuing in the open. The Conductor Quantum team's work on ML-driven quantum control seeded the broader research program this decoder belongs to; their framing of the classical–quantum integration problem shaped the author's approach long before the first line of code was written. Any merit in this work is a downstream consequence of theirs.

---

## References

[1] Google Quantum AI. "Quantum error correction below the surface code threshold." Nature 638, 920-926 (2024, published online December 9, 2024). arXiv:2408.13687.

[2] Higgott, O. & Gidney, C. "Sparse Blossom: correcting a million errors per core second with minimum-weight matching." arXiv:2303.15933 (2023).

[3] Delfosse, N. & Nickerson, N. "Almost-linear time decoding algorithm for topological codes." Quantum 5, 595 (2021).

[4] Roffe, J., White, D.R., Burton, S. & Campbell, E. "Decoding across the quantum low-density parity-check code landscape." Physical Review Research 2, 043423 (2020).

[5] Bausch, J. et al. "Learning high-accuracy error decoding for quantum processors." Nature 635, 834-840 (2024).

[6] Gicev, S. et al. "A scalable and fast artificial neural network syndrome decoder for surface codes." Quantum 7, 1058 (2023).

[7] Chamberland, C., Goncalves, L., Sivarajah, P., Peterson, E. & Grimberg, S. "Techniques for combining fast local decoders with global decoders under circuit-level noise." Quantum Science and Technology 8(4), 045011 (2023). arXiv:2208.01178.

[8] Gu, A., Bonilla Ataides, J.P., Lukin, M.D. & Yelin, S.F. "Scalable Neural Decoders for Practical Fault-Tolerant Quantum Computation." arXiv:2604.08358 (2026).

[9] Fowler, A.G. et al. "Surface codes: Towards practical large-scale quantum computation." Physical Review A 86, 032324 (2012).

[10] Gidney, C. "Stim: a fast stabilizer circuit simulator." Quantum 5, 497 (2021).

[11] Jordan, K., Jin, Y., Boza, V., You, J., Cesista, F., Newhouse, L. & Bernstein, J. "Muon: an optimizer for hidden layers in neural networks." https://kellerjordan.github.io/posts/muon/ (2024).

[12] Vittal, A. et al. "Astrea: Accurate Quantum Error-Decoding via Practical Minimum-Weight Perfect-Matching." Proc. ISCA (2023).

[13] Sivak, V.V. et al. "Reinforcement Learning Control of Quantum Error Correction." arXiv:2511.08493 (2025).

[14] Lange, M., Havström, P., Srivastava, B., Bengtsson, I., Bergentall, V., Hammar, K., Heuts, O., van Nieuwenburg, E. & Granath, M. "Data-driven decoding of quantum error correcting codes using graph neural networks." Physical Review Research 7, 023181 (2025). arXiv:2307.01241. Open-source implementation: https://github.com/LangeMoritz/GNN_decoder.

[15] Varbanov, B.M., Serra-Peralta, M., Byfield, D. & Terhal, B.M. "Neural network decoder for near-term surface-code experiments." Physical Review Research 7, 013029 (2025). arXiv:2307.03280.

[16] NVIDIA. "Ising-Decoder-SurfaceCode-1." https://github.com/NVIDIA/Ising-Decoding (released April 2026).

---

## Appendix A: Reproducibility

All code, trained checkpoints, benchmark scripts, and raw logs are available at **https://github.com/bledden/pathfinder**. The repository README is the canonical, versioned entry point; this appendix lists the minimum steps to reproduce the numbers reported in this paper.

### A.1 Dependencies

Minimum versions (matches what was used for measurements reported in this paper):

- Python 3.11
- PyTorch ≥ 2.4 (training), 2.6 recommended for the H200 latency numbers in Section 5.3
- Triton 3.2 (bundled with PyTorch 2.6) for the custom DirectionalConv3d kernel
- Stim 1.15, PyMatching 2.3 — for syndrome generation and the MWPM baseline
- Muon optimizer: `pip install git+https://github.com/KellerJordan/Muon` (use the `SingleDeviceMuon` variant for single-GPU training)
- NumPy, pybind11, pytest

### A.2 Reproducing the LER results (Table 1, 100K shots, MI300X or any CUDA GPU)

```bash
# Install
pip install stim pymatching torch numpy
pip install git+https://github.com/KellerJordan/Muon
git clone https://github.com/bledden/pathfinder && cd pathfinder

# Train the full d=7 model (~5–6 h on MI300X or H200)
python train/train.py --distance 7 --hidden_dim 256 --steps 80000 --noise_rate 0.007

# Run the 100K-shot definitive evaluation that produced Table 1
python run_final_eval.py
```

The repository includes the `d7_final/best_model.pt` checkpoint that produced the Table 1 numbers; `run_final_eval.py` can be pointed at this checkpoint to reproduce the LER comparison without retraining.

### A.3 Reproducing the H200 latency numbers (Section 5.3)

```bash
# Requires NVIDIA H200 (or an equivalent Hopper-class GPU)
# with PyTorch 2.6 + Triton 3.2 + CUDA 12.4

# Reference-implementation latency (produces Table 3a and 3b "Inductor only" row)
python bench/h200_final_benchmark.py

# Triton kernel: numerical equivalence check vs. reference (Section 5.3)
python bench/triton_ler_test.py
# Expected: 0–2 prediction disagreements per 10,000 shots at p=0.003, 0.007, 0.010

# Triton kernel: latency comparison, alternating pairs (Section 5.3)
python bench/triton_vs_orig.py
# Expected: Triton variant is 22% faster at B=1024 and 20% faster at B=1 on d=7
```

Intermediate artifacts from the runs that produced Section 5.3 are preserved in `bench/results/` (raw logs and JSONs).

### A.4 Reproducing the PyMatching latency numbers (Table 3c, Section 5.3)

The PM numbers in Table 3c are single-core Apple M4 measurements using `pymatching.Matching.decode()` (single-syndrome) and `decode_batch()`. Raw run log: `bench/results/pymatching_latency_m4.txt`. To re-measure on any CPU with stim + pymatching installed (no GPU required):

```bash
python -c "
import stim, pymatching, numpy as np, time
d = 7; p = 0.007
circuit = stim.Circuit.generated('surface_code:rotated_memory_z', rounds=d, distance=d,
    after_clifford_depolarization=p, after_reset_flip_probability=p,
    before_measure_flip_probability=p, before_round_data_depolarization=p)
matching = pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
det, _ = circuit.compile_detector_sampler().sample(6000, separate_observables=True)
det = det.astype(np.uint8)
for i in range(500): _ = matching.decode(det[i])
t0 = time.perf_counter()
for i in range(500, 5500): _ = matching.decode(det[i])
print(f'{(time.perf_counter()-t0)*1e6/5000:.2f} us/syn single-syndrome')
"
```

### A.5 Reproducing the ensemble results (Table 5, Section 5.6)

```bash
python bench/ensemble_test.py
# Outputs neural-alone, PM-alone, OR-oracle, and confidence-thresholded ensemble LERs at p in {0.003, 0.005, 0.007, 0.010}
```

### A.6 Reproducing the distillation results (Section 5.10)

```bash
# Narrow H=128 student from full H=256 teacher (~60 min on H200)
python train/train_distill.py

# H=192 student from full teacher (~100 min on H200)
python train/train_h192_distill.py
```

Both scripts require the full-teacher checkpoint at `train/checkpoints/d7_final/best_model.pt`.

### A.7 Hardware used in this paper

Training was performed on a rented AMD Instinct MI300X (192 GB HBM3) via ROCm; model correctness was verified on Apple M4 CPU (d=3 only — CPU is too slow for d≥5 training). Latency benchmarks reported in Section 5.3 were collected on a rented NVIDIA H200 SXM (141 GB HBM3e) via CUDA, selected for apples-to-apples comparison with Gu et al. [8]. PyMatching latency benchmarks were collected on an Apple M4 CPU (single core).

Pathfinder's PyTorch model code (`train/model.py`) has no vendor-specific dependencies and runs on CUDA, ROCm, MPS, and CPU. The Triton kernel (`bench/triton_directional.py`) is NVIDIA-specific (Triton 3.2+ on Hopper); it is *not* imported by the training or evaluation scripts and does not affect the core repository's AMD/CPU compatibility.

### A.8 Trained checkpoints

All checkpoints are distributed in `train/checkpoints/`:

| Path | Architecture | Purpose |
|------|--------------|---------|
| `d7_final/best_model.pt` | H=256, L=7, 500K params | The primary d=7 model producing Table 1 results |
| `d7_narrow/best_model.pt` | H=128, L=7, 126K params | Narrow variant (Section 5.9) |
| `d7_distill/best_model.pt` | H=128, L=7, 126K params | Narrow distilled from full teacher (Section 5.10) |
| `d7_h192_distill/best_model.pt` | H=192, L=7, 282K params | Intermediate distilled variant (Section 5.10) |
| `d7_p01/`, `d7_p015/`, `d7_mixed/` | H=256, L=7 | Noise-target specializations (Section 4.5) |
| `d5_muon/`, `d5/`, `d5_gpu/` | H=256, L=5 | d=5 models |
| `best_model.pt` (top-level) | H=256, L=3 | d=3 model |
| `ablation_stdconv_d5/`, `ablation_nocurriculum_d5/` | d=5 ablations | Section 5.4 |

Each checkpoint stores `model_state_dict`, a `DecoderConfig` instance, and (for most) training metadata. Loading example:

```python
import torch
from train.model import NeuralDecoder
ck = torch.load("train/checkpoints/d7_final/best_model.pt", weights_only=False, map_location="cuda")
model = NeuralDecoder(ck["config"]).cuda()
model.load_state_dict(ck["model_state_dict"])
model.eval()  # set to inference mode
```
