# Pathfinder

A direction-aware neural decoder for quantum error correction that outperforms minimum-weight perfect matching (MWPM) on surface codes.

Wins or ties PyMatching at 24/24 evaluation points across d=3, 5, 7 and p=0.0005 to 0.015 in the measurements reported here (13/24 show non-overlapping 95% Wilson CIs; see paper §5.1). 6.12 μs/syn inference latency at d=7 B=1024 on NVIDIA H200 with a custom Triton kernel, sustaining the 7-μs superconducting cycle-time budget.

**A note on priority.** [Lange et al. (Phys. Rev. Research 7, 023181, 2025)](https://github.com/LangeMoritz/GNN_decoder) previously released an open-source GNN decoder that also outperforms PyMatching on rotated surface codes under circuit-level noise. Pathfinder is **not** the first open-source decoder to beat PyMatching on this task. Pathfinder's distinct contributions are (1) extending the tested operational noise range to p ∈ {0.007, 0.010, 0.015}, which Lange et al. did not cover; (2) identifying the Muon optimizer as the dominant factor in decoder accuracy (+72% LER without it); (3) a custom Triton kernel that sustains the d=7 cycle-time budget on H200 GPUs; (4) an ensemble with PyMatching that exploits their near-disjoint failure modes. A rigorous head-to-head comparison with Lange et al. at matched noise model is pending (see paper §5.11).

## Results

Definitive evaluation: 100,000 shots per data point, circuit-level depolarizing noise.

### Pathfinder vs PyMatching -- Logical Error Rate (%)

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

Bold = lower (better). Pathfinder wins or ties at every point.

### Inference Latency

| Configuration | d=7 Latency | Throughput |
|--------------|-------------|------------|
| torch.compile + FP16 | **19 us/syn** | 53K syn/s |
| Gu et al. (H200) | ~40 us/syn | -- |
| AlphaQubit (TPU) | 63 us/syn | -- |

FP16 quantization: zero accuracy loss (verified on 50K shots).

### Comparison with Other Decoders

| Decoder | Beats PM? | Latency | Open Source |
|---------|-----------|---------|-------------|
| **Pathfinder (this work)** | **Yes, 24/24 points (Table 1)** | **6.12 us (H200 + Triton)** | **Yes** |
| Lange et al. (PRR 2025) | Yes (first, d=3-9) | not measured here | Yes |
| AlphaQubit (Google) | Yes (~6%) | 63 us | No |
| Gu et al. (Harvard) | Yes (17x on Gross codes) | ~40 us | No |
| Astrea (Georgia Tech) | No (same as PM) | 1 ns | No |
| PyMatching v2 | Baseline | ~5-10 us (noise-dependent) | Yes |
| Union-Find | No (7-30x worse) | ~20 us | Yes |

Head-to-head with Lange at matched noise model: Lange wins 19/21 points on their own evaluation harness (Pathfinder is OOD on their 4-parameter noise model). See paper §5.11.

## Architecture

**DirectionalConv3d**: The key innovation. Instead of a single 3x3x3 convolution kernel, Pathfinder uses 7 separate weight matrices -- one for each neighbor direction in the 3D syndrome lattice (self, +t, -t, +row, -row, +col, -col). This preserves the lattice geometry that standard convolution blurs.

```
Input: Binary syndrome [B, 1, R, d, d]
  -> Embedding (1x1x1 conv, 1 -> 256)
  -> L = d Bottleneck Blocks:
       Reduce (H -> H/4) -> DirectionalConv3d -> Restore (H/4 -> H) + Residual + LayerNorm
  -> Global Average Pool
  -> MLP -> Logit per observable
```

**Muon optimizer**: Newton-Schulz orthogonalization for 2D weight matrices. The single most impactful design choice -- 72% LER improvement over AdamW.

Model sizes: 252K params (d=3), 376K params (d=5), 500K params (d=7). All fit in GPU L2 cache at FP16.

## Quick Start

### Install

```bash
pip install stim pymatching torch pybind11 numpy pytest
git clone https://github.com/bledden/pathfinder.git
cd pathfinder
mkdir build && cd build
cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
make -j
```

### Train

```bash
# d=3 on CPU (~65 min)
python train/train.py --distance 3 --hidden_dim 256 --steps 20000

# d=5 on GPU (~3 hrs)
python train/train.py --distance 5 --hidden_dim 256 --steps 80000

# d=7 on GPU (~5.5 hrs)
python train/train.py --distance 7 --hidden_dim 256 --steps 80000
```

### Evaluate

```bash
python run_final_eval.py
```

### Use a Pre-trained Model

```python
import torch
from train.model import NeuralDecoder
from train.data import SyndromeDataset, DataConfig

# Load pre-trained d=5 model
ckpt = torch.load("train/checkpoints/d5_muon/best_model.pt", weights_only=False)
model = NeuralDecoder(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Generate and decode syndromes
ds = SyndromeDataset(DataConfig(distance=5, rounds=5, physical_error_rate=0.007))
syndromes, labels = ds.sample(1000)

with torch.no_grad():
    predictions = model.predict(syndromes)  # bool tensor [1000, 1]
```

## Tests

```bash
python -m pytest tests/ -v
```

## Ablation Study

| Variant | LER (%) | Impact |
|---------|---------|--------|
| Full (DirectionalConv + Muon + Curriculum) | 1.28 | baseline |
| Standard Conv3d + Muon + Curriculum | 1.33 | DirectionalConv: +4% |
| DirectionalConv + Muon + No Curriculum | 1.23 | Curriculum: ~0% |
| DirectionalConv + AdamW + Curriculum | 2.20 | **Muon: +72%** |

## Code Types

Pathfinder generalizes beyond the rotated surface code:

| Code Type | Pathfinder | PyMatching | Improvement |
|-----------|-----------|------------|-------------|
| Rotated Surface Z (d=5) | **1.56%** | 1.92% | 1.2x |
| Color Code XYZ (d=3) | **3.76%** | 12.51% | **3.3x** |
| Rotated Surface X (d=5) | **2.01%** | 2.28% | 1.1x |

## Additional Findings

- **Calibration**: ECE = 0.002 (exceptionally well-calibrated confidence scores)
- **Failure analysis**: Pathfinder and MWPM fail on almost entirely different syndromes (0.01% overlap), suggesting ensembling would yield further improvement
- **Noise generalization**: Beats PyMatching on phenomenological noise without retraining
- **Sample efficiency**: Converges in 77M samples vs Gu et al.'s 266M (3.5x more efficient)
- **Error suppression**: Pathfinder's advantage grows with code distance (waterfall regime)

## Paper

See `paper/pathfinder.md` for the full write-up.

## Key References

- Gu et al. "Scalable Neural Decoders for Practical Fault-Tolerant Quantum Computation." arXiv:2604.08358 (2026)
- Higgott and Gidney. "Sparse Blossom." arXiv:2303.15933 (2023)
- Gidney. "Stim: a fast stabilizer circuit simulator." Quantum 5, 497 (2021)
- Jordan. "Muon: an optimizer for hidden layers." (2025)

## Training Cost

Total: ~28 GPU-hours on AMD MI300X (~$65 USD).

## License

MIT
