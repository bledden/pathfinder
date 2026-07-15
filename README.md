# Pathfinder

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21303610.svg)](https://doi.org/10.5281/zenodo.21303610)

Two open-source decoder systems for quantum error correction on rotated surface codes, both built on **Pathfinder** — a direction-specific 3D CNN [Gu et al. 2026] trained with the Muon optimizer.

## TL;DR

- **System 1: throughput-optimized decoder.** Canonical Pathfinder (H=256, 500K params) + a custom Triton kernel for DirectionalConv3d → **6.12 μs/syndrome at d=7 batch=1024 on NVIDIA H200 SXM** — batched throughput below a 7-μs-per-syndrome service-rate target at every operational noise rate. This is amortized throughput, not streaming real-time (batch-1 latency is 201 μs), and a cross-hardware comparison — PyMatching is CPU-timed (paper §5.3, §A.3). On accuracy, **never statistically beaten by PyMatching** across Table 1's 24 points under 3-parameter circuit-level noise: 20 point-estimate wins, 12 with non-overlapping 95% Wilson CIs, using one fixed validation-selected checkpoint per distance (paper §5.1, Table 1).

- **System 2: lowest-LER decoder (Pathfinder-Triad with PFWL3S voter).** A 3-way majority vote of (PFWL3S, Lange et al. 2025, PyMatching), where **PFWL3S** = three independent random-seed H=384 Pathfinders trained 160K steps each with Lange-teacher distillation, ensembled by averaging logits. Achieves **LER 2.384% at d=7 p=0.007 (100K shots)** vs Lange's 2.956% — **strict statistical win** with non-overlapping 95% Wilson CIs (**Triad vs Lange: 19.4% relative LER reduction**; the PFWL3S voter alone is 2.492%, 15.7% vs Lange). Strict-CI wins at 5 (d, p) operational points: **d=7 p ∈ {0.007, 0.010, 0.015} and d=9 p ∈ {0.007, 0.010}** (paper §5.12, §6.3). Latency is Lange-bounded at ≈72 μs/syn — for non-real-time deployment (offline verification, post-selection in repeat-until-success).

PFWL3S as an *individual* decoder also strictly beats Lange at 4 (d, p) points (d=5 p=0.015, d=7 p ∈ {0.007, 0.010, 0.015}; paper §5.13) — to my knowledge the first open-source individual neural decoder reported to do so.

## A note on priority and fairness

[**Lange et al.** (Phys. Rev. Research 7, 023181, 2025)](https://github.com/LangeMoritz/GNN_decoder) previously released the first open-source neural decoder to outperform PyMatching on rotated surface codes under circuit-level noise. **Pathfinder is not the first open-source decoder to beat PyMatching on this task** — Lange et al. holds that priority. At matched-noise head-to-head, Lange's GNN has lower individual LER than canonical Pathfinder (paper §5.11).

**Fairness check (controlled, resolved — paper §5.11 Table 9b).** Lange's published weights were trained at p ∈ {0.001, …, 0.005}, so the strict-CI wins at p ∈ {0.007, 0.010, 0.015} initially compared in-distribution PFWL3S against *out-of-distribution* Lange. To control for this I fine-tuned Lange's GNN at p=0.007 using its own training infrastructure: that closed most of the OOD gap (d=7 p=0.007: 2.956% → 2.739%), but **PFWL3S still strictly beats this single fine-tuned Lange** at all three operational rates *under the marginal-CI test* — the CI-edge gaps shrink to 0.049 / 0.55 / 0.27 pp but never invert. The strict-CI win is therefore a *controlled* result, not an out-of-distribution artifact. I further controlled for **ensemble size and training budget** (paper §5.11 audit C3): against a *full-recipe 3-seed fine-tuned-Lange ensemble* (the strongest possible baseline, d=7 p=0.007 = 2.652%), PFWL3S still wins under both the marginal-CI test **and** the more-powerful paired **McNemar** test at p=0.007 (McNemar p=0.0025) and p=0.010 (p<10⁻⁴, robust to Bonferroni over all 24 comparisons), but at **p=0.015 the win does not survive the paired test** (McNemar p=0.09; PyMatching also ties PFWL3S there). Honest bottom line: PFWL3S beats even a full-recipe Lange ensemble at p=0.007 and p=0.010 (most robustly p=0.010); at p=0.015 it ties the strongest control.

## Distinct contributions of this work

1. **Pathfinder-Triad** — among the open-source decoders evaluated in this study, the only decoder system to statistically-significantly beat Lange on matched noise, at 5 (d, p) points across two code distances (paper §5.12, §6.3; the d=9 pair is vs published out-of-distribution Lange, with no fine-tuned control).
2. **PFWL3S** — to my knowledge, the first open-source individual neural decoder to strictly beat Lange's GNN with non-overlapping 95% Wilson CIs at any operational noise rate (paper §5.13, 4 (d, p) wins).
3. **The DirectionalConv3d Triton kernel** — 12× faster than Lange's GNN on identical H200 hardware for single-seed canonical Pathfinder, with batched throughput below the 7-μs/syn service-rate target (paper §5.3 + §5.11).
4. **Depth-dependent Muon ablation** — the optimizer's effect goes from +17% LER at d=3 to +72% at d=5 to catastrophic at d=7 (1.04% → 34.8% LER without Muon; paper §6.2).
5. **Extended-noise-rate Table 1** — evaluation down to p=0.0005 and up to p=0.015, broader than prior open-source coverage.
6. **Documented negative results** strengthening the headline claims:
   - **Triad-distillation arc** (paper §5.13, ~$110 GPU): 6 recipe variants showing the Triad's coverage advantage is *architectural*, not absorbable into a single PF student.
   - **Modern-primitives hybrid** (paper §5.14): CNN+attention+SwiGLU at 9× parameter count is *worse* than the simpler CNN at matched compute.
   - **d=9 PFWL3S-H256-d9** (paper §6.3): individually loses to Lange (the recipe-level reversal does *not* extend to d=9 at H=256). However, Pathfinder-Triad with this PFWL3S-H256-d9 voter still strictly beats Lange at d=9 p=0.007 and p=0.010 (extending the §5.12 result from d=7 to d=9).
7. **Real-hardware runs (IBM Heron r2)** (paper §5.15): PFWL3S, trained only on simulated noise, shows **no statistically resolved difference from PyMatching** at d=3 r=3 on `ibm_fez` (10K shots, Wilson-CI criterion) — which does not establish equivalence or an advantage. A soft (analog-IQ) readout follow-up (§5.15.2) also resolves no difference on this clean, matching-saturated chip; a synthetic readout-SNR positive control confirms the soft pipeline is not inert (peak gain at SNR≈4, McNemar p=3×10⁻⁴, Holm/Bonferroni-robust).

## Results

### Pathfinder vs PyMatching, 3-parameter noise — Logical Error Rate (%), 100K shots

| p | d=3 PF | d=3 PM | d=5 PF | d=5 PM | d=7 PF | d=7 PM |
|---|---|---|---|---|---|---|
| 0.0005 | **0.009** | 0.011 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.001 | **0.046** | 0.064 | **0.007** | 0.009 | 0.000 | 0.000 |
| 0.002 | **0.161** | 0.191 | **0.028** | 0.055 | 0.005 | **0.004** |
| 0.003 | **0.333** | 0.402 | **0.104** | 0.154 | **0.022** | 0.040 |
| 0.005 | **1.002** | 1.098 | **0.585** | 0.751 | **0.267** | 0.411 |
| 0.007 | **1.818** | 2.014 | **1.521** | 1.891 | **1.071** | 1.548 |
| 0.010 | **3.521** | 3.742 | **4.145** | 4.810 | **4.140** | 5.257 |
| 0.015 | **7.315** | 7.728 | **12.137** | 12.606 | **15.546** | 16.883 |

Bold = lower (better). One fixed checkpoint per distance, **no per-rate model selection** (the d=7 checkpoint is validation-selected on held-out shots and reported on disjoint test shots). Pathfinder is **never statistically beaten**: 20 point-estimate wins, 3 zero-error ties, and one single-failure statistical tie (d=7 p=0.002: PF 5 vs PM 4 failures in 100K shots). 12 of 24 points show non-overlapping 95% Wilson CIs — all Pathfinder wins, spanning every distance and concentrating at p ≥ 0.005 (paper §5.1).

### Pathfinder-Triad vs Lange, 4-parameter noise (operational rates) — Logical Error Rate (%), 100K shots

| (d, p) | PFWL3S (3-seed) | Lange | Pathfinder-Triad | Triad vs Lange (rel. LER ↓) |
|---|---|---|---|---|
| (5, 0.015) | **17.205%** | 17.898% | **16.779%** | strict CI · 6.3% |
| (7, 0.007) | **2.492%** | 2.956% | **2.384%** | strict CI · 19.4% |
| (7, 0.010) | **9.173%** | 10.764% | **8.689%** | strict CI · 19.3% |
| (7, 0.015) | **27.328%** | 30.200% | **25.872%** | strict CI · 14.3% |
| (9, 0.007) | 3.310% (PFWL3S-H256-d9) | 2.623% | **2.277%** | strict CI · 13.2% |
| (9, 0.010) | 15.061% (PFWL3S-H256-d9) | 13.085% | **10.852%** | strict CI · 17.1% |

*"rel. LER ↓" = relative reduction of the Triad's LER vs Lange = (Lange − Triad)/Lange. "strict CI" = non-overlapping 95% Wilson intervals (vs published Lange). The matched-control / paired-McNemar analysis of these wins is in paper §5.11 (audits C2/C3).*

Bold = strictly better. Note PFWL3S-H256-d9 *loses* individually to Lange at d=9; the Triad still wins because PM and PF catch independent errors (paper §6.3 analysis). Triad's coverage advantage is structural, not a recipe artifact (Triad-distill negative result, paper §5.13).

### Inference Latency at d=7 (H200 SXM, FP16, torch.compile max-autotune)

| Configuration | B=1 latency | B=1024 throughput | Below 7-μs/syn service-rate target? |
|---|---|---|---|
| **Canonical Pathfinder + Triton** | 201 μs | **6.12 μs/syn** | **✓ (13% below)** |
| Canonical Pathfinder (Inductor only) | 250 μs | 7.86 μs/syn | ✗ (12% above) |
| Lange GNN (measured here on H200) | 1,918 μs | 71.67 μs/syn | ✗ (12× above) |
| PFWL3S (3-seed-avg, reference impl) | — | ≈61 μs/syn | ✗ (paper §5.13 latency) |
| PFWL3S (3-seed-avg, Triton extrapolated) | — | ≈24 μs/syn | ✗ (3.5× above) |
| Pathfinder-Triad (Lange-bounded) | — | ≈72 μs/syn | ✗ (10× above) |
| PyMatching v2 (Apple M4 single core, p=0.007) | 9.65 μs | 7.77 μs/syn (batch) | ✗ at p ≥ 0.007 |

The 12× faster claim is **canonical 1-seed Pathfinder + Triton vs Lange**, both at d=7 batch=1024 on H200. The service-rate target is amortized batched throughput — batch-1 latency remains two orders of magnitude above streaming real-time requirements (paper §5.3). The strict-CI-winning PFWL3S and Pathfinder-Triad systems are above the target; they're for non-real-time deployment.

## Architecture

**DirectionalConv3d**: instead of a single 3×3×3 convolution kernel, Pathfinder uses 7 separate weight matrices — one for each neighbor direction in the 3D syndrome lattice (self, ±t, ±row, ±col). Preserves the lattice geometry that standard convolution blurs.

```
Input: Binary syndrome [B, 1, R, d, d]
  -> Embedding (1×1×1 conv, 1 -> H)
  -> L = d Bottleneck Blocks:
       Reduce (H -> H/4) -> DirectionalConv3d -> Restore (H/4 -> H) + Residual + LayerNorm
  -> Global Average Pool
  -> MLP -> Logit per logical observable
```

**Muon optimizer**: Newton-Schulz orthogonalization for 2D weight matrices. The single most impactful design choice — 72% LER improvement over AdamW at d=5; *catastrophic to remove at d=7* (1.04% → 34.8% LER in the same step budget). See paper §6.2 + Figure 4.

Model sizes (canonical Pathfinder): 252K params (d=3), 376K params (d=5), 500K params (d=7). All fit in GPU L2 cache at FP16.

PFWL3S widens to H=384 (d=5: ~850K per ckpt; d=7: 1.09M per ckpt; total per-distance: ~2.55M for d=5, ~3.27M for d=7 across 3 seeds).

## Quick Start

### Install

```bash
pip install stim pymatching torch numpy pytest
pip install git+https://github.com/KellerJordan/Muon
git clone https://github.com/bledden/pathfinder.git
cd pathfinder
```

### Run Table 1 evaluation (uses included d=3, d=5, d=7 ckpts)

The simplest entry point is the top-level dispatcher (`python cli.py --help` lists all subcommands and which paper section each one produces):

```bash
python cli.py eval-table1            # paper §5.1 Table 1, ~3 min on GPU
python cli.py --help                 # full subcommand list
```

The existing `run_*.py` scripts at the repo root remain for backwards compatibility; `cli.py` is a thin dispatcher that forwards arguments to them.

```bash
python run_final_eval.py             # equivalent direct invocation
```

### Train from scratch (Table 1 recipe)

```bash
# d=3 on CPU (~65 min) or GPU (~10 min)
python train/train.py --distance 3 --hidden_dim 256 --steps 20000

# d=5 on GPU (~3 hr)
python train/train.py --distance 5 --hidden_dim 256 --steps 80000

# d=7 on GPU (~5.5 hr)
python train/train.py --distance 7 --hidden_dim 256 --steps 80000
```

### Reproduce the §5.13 PFWL3S strict-CI win at d=7

```bash
# 1. Fine-tune at 4-parameter noise (~20 min on H200, init from Table-1 ckpt)
python bench/results/h200_main/train_finetune_4param.py \
  --distance 7 --steps 40000 --batch 256 --noise_rate 0.007 \
  --init train/checkpoints/d7_final/best_model.pt

# 2. Train PFWL3S: 3 seeds × 160K steps with Lange-teacher distillation (~9 hr total on H200)
#    Requires Lange GNN repo cloned for the teacher
git clone https://github.com/LangeMoritz/GNN_decoder
pip install torch-geometric torch-cluster
for SEED in 0 1 2; do
  python bench/results/h200_main/triad_distill/scripts/train_seeded_wide_long_triad.py \
    --seed $SEED --distance 7 --hidden_dim 384 --steps 160000 \
    --batch 128 --noise_rate 0.007 --alpha_kl 0.7 --alpha_bce 0.3 \
    --ckpt checkpoints/pfwl3s_d7_seed${SEED}
done

# 3. Eval at 100K shots with 3-seed-avg ensembling
python bench/results/h200_main/triad_distill/scripts/eval_triad_distill.py
```

### Use a Pre-trained Model

```python
import torch
from train.model import NeuralDecoder

ckpt = torch.load("train/checkpoints/d7_final/best_model.pt", weights_only=False)
model = NeuralDecoder(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Inference: prepare a batched syndrome tensor [B, 1, R, H, W] using
# Stim's detector coordinates (see paper §3.4 / train/data.py)
with torch.no_grad():
    logits = model(syndromes)   # [B, n_observables]
    predictions = (logits > 0).int()
```

## Tests

```bash
python -m pytest tests/ -v --ignore=tests/test_cpp_decoder.py
```

`test_cpp_decoder.py` requires building the C++ extension via CMake (see paper §A.1); the rest of the test suite runs in pure PyTorch.

## Ablations and Negative Results

| Section | Topic | Finding |
|---|---|---|
| §5.4 | Muon vs AdamW (d=5) | Muon: +72% LER reduction |
| §5.13 Table 11 | Pathfinder-Wide / XL / Wide-Long / XLong / PFWL3S | Capacity ceiling at H=384; multi-seed averaging breaks through |
| §5.13 d=3 rescue | PFWL3S at d=3 with α_kl=0.3 | Rescues 14% → 3.18% but still loses to Lange — d=3 deployment uses canonical fine-tune |
| §5.13 Triad-distill | 6 recipes (~$110 GPU) trying to beat Triad with single decoder | Triad coverage is architectural, not absorbable |
| §5.14 | CNN+attention+SwiGLU hybrid (4.36M params) | Worse than simpler 500K CNN at matched budget |
| §6.2 | Depth-dependent Muon | +17% (d=3) → +72% (d=5) → catastrophic (d=7) |
| §6.3 | d=9 PFWL3S-H256-d9 | Individually loses to Lange; Triad still strict-wins at p=0.007/0.010 |

## Generalization

| Generalization axis | Result |
|---|---|
| 4-parameter circuit-level noise (Lange's noise model) — canonical Pathfinder | Loses to Lange ~14% relative at d=7, non-overlapping CIs (McNemar p=1.7×10⁻⁸; §5.11) |
| 4-parameter noise — PFWL3S | **Strict-CI wins** at 4 (d, p) operational points (§5.13) |
| Phenomenological noise (data errors only) | **PyMatching wins 15/15 — Pathfinder does not generalize cleanly** (§5.7, retraction of an earlier claim) |
| Color code XYZ (d=3) | **Pathfinder beats PM 3.3×** (3.76% vs 12.51%, §5.7 Table 6) — direction-specific architecture is especially effective on richer stabilizer geometry |
| 3-parameter noise (cross-eval of PFWL3S trained on 4-param) | PFWL3S strictly beats PM at 9/18 operational points (§5.1 Table 1b) |

## Compute Cost (full project)

| Section | Compute | Cost (USD) |
|---|---|---|
| Original Table 1 + ablations + distillation (MI300X) | ~28 GPU-hr | ~$65 |
| Triton kernel + H200 latency (H200) | ~10 GPU-hr | ~$40 |
| Lange head-to-head + Triad eval (H200) | ~15 GPU-hr | ~$60 |
| Pathfinder-Wide / Wide-Long / XLong + d=5 multi-seed + d=3 rescue | ~50 GPU-hr | ~$200 |
| Triad-distillation arc (6 recipes + mega-ensemble) | ~30 GPU-hr | ~$110 |
| Hybrid CNN+attention (negative result) | ~3 GPU-hr | ~$12 |
| d=9 PFWL3S + Triad eval | ~15 GPU-hr | ~$60 |
| **Total** | **~150 GPU-hr** | **~$550** |

Six weeks calendar time; single engineer working part-time across multiple short pod sessions.

## Paper

**"Pathfinder: Direction-Aware Neural Decoding and Complementary-Failure Ensembles for Surface Codes"** — 57 pages, 11 figures.

- Preprint PDF: [`paper/pathfinder.pdf`](paper/pathfinder.pdf) · archived at [doi:10.5281/zenodo.21303610](https://doi.org/10.5281/zenodo.21303610)
- Source: [`paper/pathfinder.md`](paper/pathfinder.md) · reviewer-style audit: [`paper/AUDIT_2026-05-13.md`](paper/AUDIT_2026-05-13.md)

To cite this work, see [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button).

## qLDPC: kernel-grounded latency–LER benchmark (in progress)

A kernel-grounded, exact-MLE-anchored, multiplicity-corrected **latency–LER Pareto benchmark** of
circuit-level decoders for the [[72,12,6]] bivariate-bicycle code under the SI1000 noise model. A
matched-protocol decoder zoo (BP, BP-OSD-0/10, BP+LSD, Relay-BP, sliding-window) is measured against an
exact maximum-likelihood **anchor** (Tesseract), with pre-registration, Wilson/TOST intervals,
paired-bootstrap gap-to-MLE, and Holm/BH multiplicity control.

- **Cross-vendor Triton kernels** — fused min-sum **BP** and **Relay-BP** decoders, LER-faithful to the
  reference implementations, run *unmodified* on both **NVIDIA (H200)** and **AMD (MI300X)** GPUs.
- **Amenability taxonomy** — per-decoder OSD/LSD fallthrough-rate vs p, Amdahl serial-fraction (kernel
  ceiling), roofline, and memory footprint.
- **Negative result** — under a parameter-matched control the Z₆×Z₆ equivariance prior gives no decoding
  advantage (consistent with the AED-QC-LDPC theorem); the residual practical-decoder gap to MLE is a
  high-syndrome-weight tail orthogonal to symmetry-averaging.

The current code lives on the
[`qldpc-mle-foundation`](https://github.com/bledden/pathfinder/tree/qldpc-mle-foundation) branch. Paper
in preparation.

## Acknowledgements

Methodology for the real-hardware and qLDPC studies — the maximum-likelihood decoding-ceiling check,
the novel-syndrome (train/test-leakage) split, and the matched-parameter control — was sharpened
through adversarial review by the **Coda** expert model, which is gratefully acknowledged.

## Key References

- **Lange et al.** "Data-driven decoding of quantum error correcting codes using graph neural networks." Phys. Rev. Research 7, 023181 (2025). arXiv:2307.01241. Open source: [github.com/LangeMoritz/GNN_decoder](https://github.com/LangeMoritz/GNN_decoder)
- **Gu et al.** "Scalable Neural Decoders for Practical Fault-Tolerant Quantum Computation." arXiv:2604.08358 (2026)
- **Higgott & Gidney.** "Sparse Blossom: correcting a million errors per core second with minimum-weight matching." arXiv:2303.15933 (2023)
- **Gidney.** "Stim: a fast stabilizer circuit simulator." Quantum 5, 497 (2021)
- **Jordan et al.** "Muon: an optimizer for hidden layers in neural networks." (2024) [https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/)
- **Bausch et al. (AlphaQubit).** "Learning high-accuracy error decoding for quantum processors." Nature 635, 834-840 (2024)

## License

- **Code** (everything except the manuscript and figures): MIT — see [`LICENSE`](LICENSE).
- **Manuscript** (`paper/`) **and figures**: Creative Commons Attribution 4.0 International (**CC BY 4.0**) — <https://creativecommons.org/licenses/by/4.0/>. Reuse freely with attribution.

© 2026 Blake Ledden.
