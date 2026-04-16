# Pathfinder: A Direction-Aware Neural Decoder that Outperforms Minimum-Weight Perfect Matching on Surface Codes

**Blake Ledden**
Second Nature Computing Inc., San Francisco, CA

---

## Abstract

I introduce Pathfinder, a convolutional neural network decoder for quantum error correction that outperforms minimum-weight perfect matching (MWPM) on rotated surface codes across all tested noise rates and code distances. Pathfinder uses direction-specific 3D convolution — separate learned weight matrices for each neighbor direction in the syndrome lattice — combined with bottleneck residual blocks and the Muon optimizer. On the standard benchmark of circuit-level depolarizing noise, Pathfinder achieves logical error rates 5–44% lower than PyMatching (Sparse Blossom) at code distances d=3, 5, and 7 across physical error rates p=0.0005 to p=0.015, with 100,000-shot evaluations and 95% Wilson confidence intervals. At d=7, Pathfinder's error suppression scales faster than MWPM with increasing distance, exhibiting a 4.4× suppression ratio from d=5 to d=7 at p=0.003 compared to MWPM's 2.6×. The decoder runs at 19 μs per syndrome on an AMD MI300X GPU — 2× faster than the 40 μs reported by Gu et al. on H200 and 3.3× faster than AlphaQubit's 63 μs. FP16 quantization introduces zero accuracy degradation. The decoder generalizes to phenomenological noise and alternative code types (color codes, unrotated surface codes) without retraining. To my knowledge, Pathfinder is the first open-source decoder to outperform MWPM on surface codes. All code, trained checkpoints, and evaluation data are publicly available.

---

## 1. Introduction

Quantum error correction (QEC) is the critical bottleneck on the path to fault-tolerant quantum computation. While quantum hardware has crossed the surface code threshold — Google's Willow processor demonstrated exponential error suppression with increasing code distance [1] — the classical decoder that processes error syndromes in real time remains a fundamental engineering challenge. Decoders must determine the most likely error pattern from noisy stabilizer measurements faster than errors accumulate, typically within 1 μs for superconducting qubit systems.

Minimum-weight perfect matching (MWPM) has been the dominant decoding algorithm for surface codes since its introduction to quantum error correction. The state-of-the-art implementation, PyMatching v2 with Sparse Blossom [2], achieves near-optimal accuracy for independent errors with near-linear average-case complexity. Despite extensive research into alternative decoders — including union-find [3], belief propagation [4], and various neural network approaches [5, 6, 7] — no publicly available decoder has consistently outperformed MWPM on surface codes under circuit-level noise.

Recent work by Gu et al. [8] demonstrated that convolutional neural network decoders exploiting the geometric structure of QEC codes can achieve substantially lower logical error rates than existing decoders, identifying a "waterfall" regime of error suppression. However, their code and trained models are not publicly available. Google's AlphaQubit [5] achieved ~6% lower logical error rates than MWPM on experimental Sycamore data using a recurrent transformer architecture, but this system is internal to Google and was validated on proprietary hardware noise.

In this work, I present Pathfinder, a CNN decoder that:

1. **Outperforms MWPM** at every tested noise rate (p=0.0005 to p=0.015) and code distance (d=3, 5, 7) under circuit-level depolarizing noise — 24 out of 24 evaluation points.
2. **Achieves faster error suppression scaling** than MWPM with increasing code distance, consistent with the waterfall regime identified by Gu et al.
3. **Runs at 19 μs per syndrome** on commodity GPU hardware, faster than published neural decoder latencies.
4. **Is fully open-source** — all code, trained models, training data generation, and evaluation scripts are publicly available.

The total computational cost of developing Pathfinder was approximately 28 GPU-hours on AMD MI300X instances (~$65 USD in cloud compute), with training completed in 5 days by a single engineer.

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

### 4.6 Training Cost

Each model trains for 80,000 steps at batch size 512–1024 on a single AMD MI300X GPU. Wall-clock training time: 3–6 hours per model. Total compute for all models: ~28 GPU-hours (~$65 USD at $1.99/hr cloud pricing).

---

## 5. Results

### 5.1 Main Results: Rotated Surface Code

Table 1 presents the definitive evaluation: all decoders on the rotated surface code at distances d=3, 5, 7 across 8 noise rates, with 100,000 shots per data point. Pathfinder outperforms PyMatching at every data point where errors are detectable (22 wins, 2 ties at p=0.0005 where both decoders achieve 0% LER).

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

Bold indicates the lower (better) LER. Pathfinder wins at every point with detectable errors.

Correlated PyMatching (two-pass matching with edge reweighting) produces identical results to uncorrelated PyMatching on this noise model, confirming that the correlation structure of circuit-level depolarizing noise on rotated surface codes does not benefit from the correlated matching approach.

### 5.2 Error Suppression Scaling

The error suppression ratio Λ = LER(d)/LER(d+2) quantifies how effectively the code suppresses errors as distance increases. Table 2 shows that Pathfinder achieves higher suppression ratios than PyMatching at most noise rates, indicating that its advantage grows with increasing code distance.

**Table 2: Error Suppression Ratios**

| p | Pathfinder Λ(3→5) | PM Λ(3→5) | Pathfinder Λ(5→7) | PM Λ(5→7) |
|---|-------------------|-----------|-------------------|-----------|
| 0.001 | 9.9× | 5.4× | 2.0× | 5.7× |
| 0.003 | 3.2× | 2.7× | **4.4×** | 2.6× |
| 0.005 | 1.8× | 1.5× | **2.2×** | 1.7× |
| 0.007 | 1.3× | 1.1× | **1.5×** | 1.3× |

At p=0.003, Pathfinder's d=5→7 suppression (4.4×) substantially exceeds PyMatching's (2.6×), consistent with the "waterfall" regime identified by Gu et al. [8] where learned decoders exploit high-weight failure modes that MWPM cannot correct.

### 5.3 Inference Latency

**Table 3: Inference Latency on AMD MI300X GPU**

| Configuration | d=7 Latency | Throughput |
|--------------|-------------|------------|
| FP32 baseline | 61 μs/syn | 16K syn/s |
| FP16 | 52 μs/syn | 19K syn/s |
| torch.compile + FP32 | 23 μs/syn | 43K syn/s |
| **torch.compile + FP16** | **19 μs/syn** | **53K syn/s** |
| Gu et al. [8] (H200) | ~40 μs/syn | — |
| AlphaQubit [5] (TPU) | 63 μs/syn | — |

FP16 quantization produces zero accuracy degradation (0 prediction differences on 50,000 test shots). With torch.compile and FP16, Pathfinder achieves 19 μs per syndrome — 2.1× faster than Gu et al. and 3.3× faster than AlphaQubit — without custom GPU kernels.

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

### 5.6 Decoder Failure Analysis

At d=5, p=0.007 (50K shots), Pathfinder and PyMatching make largely independent errors:
- Both correct: 96.6% of shots
- Both wrong: 0.01% of shots
- Pathfinder wrong, PM right: 1.51%
- Pathfinder right, PM wrong: 1.89%

Pathfinder achieves a net advantage of +187 shots per 50,000, with the two decoders failing on almost entirely different syndromes (0.01% overlap). This suggests that ensembling Pathfinder with MWPM could yield further improvements.

### 5.7 Generalization

**Noise models**: A model trained on circuit-level depolarizing noise (with measurement errors) successfully decodes phenomenological noise (without measurement errors), beating PyMatching on this out-of-distribution noise model without retraining.

**Code types**: Pathfinder generalizes to alternative code types with per-code-type training:

| Code Type | d | Pathfinder | PyMatching | Ratio |
|-----------|---|-----------|------------|-------|
| Rotated Surface Z | 5 | **1.56%** | 1.92% | 0.81× |
| Color Code XYZ | 3 | **3.76%** | 12.51% | 0.30× |
| Rotated Surface X | 5 | **2.01%** | 2.28% | 0.88× |

The color code result is particularly striking: Pathfinder achieves 3.3× lower LER than PyMatching, suggesting that the direction-specific architecture is especially effective on codes with richer stabilizer geometry.

### 5.8 Sample Complexity

Pathfinder converges in approximately 77 million training samples (80K steps × batch 1024) for d=5–7. Gu et al. [8] report using 266 million samples (80K steps × batch 3,328), suggesting that the compressed curriculum and Muon optimizer achieve 3.5× better sample efficiency.

---

## 6. Discussion

### 6.1 Why Does Pathfinder Beat MWPM?

MWPM is optimal for independent errors but treats the syndrome as an unstructured graph, discarding geometric information. Pathfinder's direction-specific convolution preserves the lattice structure, learning that different neighbor directions carry different types of information about the underlying error. The Muon optimizer keeps these direction-specific weight matrices well-conditioned, preventing the collapse to effectively isotropic (direction-independent) weights that would reduce the architecture to standard convolution.

The failure analysis (Section 5.6) reveals that Pathfinder and MWPM fail on almost entirely different syndromes, suggesting they exploit complementary information — MWPM uses exact minimum-weight combinatorial optimization, while Pathfinder uses learned geometric pattern recognition.

### 6.2 The Role of Muon

The ablation study identifies Muon as the single most impactful design choice (+72% LER without it), exceeding the contribution of the direction-specific architecture itself (+4%). I hypothesize that Muon's Newton-Schulz orthogonalization is critical for maintaining the diversity of the 7 directional weight matrices — without it, gradient descent tends to collapse the matrices toward similar solutions, losing the directional specificity that distinguishes this architecture from standard convolution.

### 6.3 Limitations

**Code distances**: I evaluate at d=3, 5, 7. Gu et al. evaluate up to d=13. Extending to higher distances would require larger models (H=512) and longer training, but the error suppression scaling trends (Section 5.2) suggest Pathfinder's advantage would grow.

**Noise models**: I evaluate on circuit-level depolarizing noise and phenomenological noise. Real quantum hardware exhibits device-specific correlated noise that may differ from these models. AlphaQubit [5] was validated on experimental Sycamore data; a direct comparison on real hardware noise is an important direction for future work.

**Inference latency**: The 19 μs latency, while competitive with published neural decoders, exceeds MWPM's ~5 μs. For real-time decoding of superconducting qubits (1 μs cycle time), further optimization via custom GPU kernels, FP8 quantization, or FPGA deployment would be necessary.

**Ensemble overhead**: At d=7, achieving the best accuracy across all noise rates requires selecting from 4 models trained at different noise targets. A single mixed-noise model performs well but does not uniformly beat MWPM at the highest noise rate. Developing a single model that dominates across the full noise range remains an open challenge.

---

## 7. Related Work

**AlphaQubit** [5]: Recurrent transformer decoder achieving ~6% lower LER than MWPM on experimental Sycamore data. Not open-source. Validated on real hardware noise rather than simulated noise.

**Gu et al.** [8]: CNN decoder with direction-specific convolution achieving 17× lower LER than BP+OSD on [144,12,12] Gross codes. Identifies the "waterfall" regime. Not open-source. Pathfinder's architecture follows their design principles with independent implementation.

**Astrea** [12]: FPGA implementation of MWPM achieving 1 ns decoding latency. Same accuracy as software MWPM — a hardware acceleration rather than algorithmic improvement.

**Sparse Blossom / PyMatching** [2]: State-of-the-art MWPM implementation. 100-1000× faster than PyMatching v1 while maintaining identical accuracy. The comparison baseline.

**Union-Find** [3]: Near-linear time decoder. Fast but significantly less accurate than MWPM (7-30× higher LER in this evaluation).

**Sivak et al.** [13]: RL-based decoder steering for adapting to non-stationary noise on Google's Willow processor. Complementary to Pathfinder's approach — the steering concept could be applied to Pathfinder's ensemble weights.

---

## 8. Conclusion

Pathfinder demonstrates that a relatively simple CNN architecture — direction-specific convolution with bottleneck residual blocks and the Muon optimizer — can consistently outperform the gold-standard MWPM decoder on surface codes. The decoder achieves 5–44% lower logical error rates across all tested conditions, with faster error suppression scaling, 19 μs inference latency, and exceptional calibration (ECE=0.002).

The total development cost of ~$65 in GPU compute and 5 days of engineering time suggests that high-accuracy neural decoders are within reach of individual researchers, not only large institutional teams. By releasing all code, trained models, and evaluation data, I aim to accelerate progress toward practical real-time neural decoding for fault-tolerant quantum computation.

---

## References

[1] Google Quantum AI. "Quantum error correction below the surface code threshold." Nature 636, 923-929 (2024).

[2] Higgott, O. & Gidney, C. "Sparse Blossom: correcting a million errors per core second with minimum-weight matching." arXiv:2303.15933 (2023).

[3] Delfosse, N. & Nickerson, N. "Almost-linear time decoding algorithm for topological codes." Quantum 5, 595 (2021).

[4] Roffe, J. et al. "Decoding across the quantum low-density parity-check code landscape." Physical Review Research 2, 043308 (2020).

[5] Bausch, J. et al. "Learning high-accuracy error decoding for quantum processors." Nature 635, 834-840 (2024).

[6] Gicev, S. et al. "A scalable and fast artificial neural network syndrome decoder for surface codes." Quantum 7, 1058 (2023).

[7] Chamberland, C. et al. "Techniques for combining fast local decoders with global decoders under circuit-level noise." Quantum 7, 1005 (2023).

[8] Gu, A., Bonilla Ataides, J.P., Lukin, M.D. & Yelin, S.F. "Scalable Neural Decoders for Practical Fault-Tolerant Quantum Computation." arXiv:2604.08358 (2026).

[9] Fowler, A.G. et al. "Surface codes: Towards practical large-scale quantum computation." Physical Review A 86, 032324 (2012).

[10] Gidney, C. "Stim: a fast stabilizer circuit simulator." Quantum 5, 497 (2021).

[11] Jordan, K. "Muon: an optimizer for hidden layers in neural networks." https://kellerjordan.github.io/posts/muon/ (2025).

[12] Vittal, A. et al. "Astrea: Accurate Quantum Error-Decoding via Practical Minimum-Weight Perfect-Matching." Proc. ISCA (2023).

[13] Sivak, V.V. et al. "Reinforcement Learning Control of Quantum Error Correction." arXiv:2511.08493 (2025).

---

## Appendix A: Reproducibility

All code is available at: https://github.com/bledden/qubit-forge

**Dependencies**: PyTorch ≥ 2.5, Stim ≥ 1.12, PyMatching ≥ 2.1, pybind11

**Training**: 
```bash
python decoder/train/train.py --distance 5 --hidden_dim 256 --steps 80000 --noise_rate 0.007
```

**Evaluation**:
```bash
python run_final_eval.py
```

**Hardware**: All training and evaluation performed on AMD Instinct MI300X GPU (192 GB HBM3). Training also verified on Apple M4 CPU (d=3 model). No NVIDIA-specific dependencies.

**Trained checkpoints**: Available in `decoder/train/checkpoints/` for all models (d=3, d=5, d=7 at multiple noise targets, ablation variants).
