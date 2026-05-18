# Hybrid CNN+GNN d=7 3-seed evaluation (§5.16)

This directory contains the data backing paper §5.16 — the architecturally-novel
single-DefectGNN-layer fusion inside the Pathfinder backbone (HybridDecoder),
trained 3 seeds × 160K steps × H=384 at d=7 p=0.007 with Lange-distillation, and
evaluated against the matched PFWL3S 3-seed baseline.

## Contents

- `hybrid_eval_d7.json` — 100K shots/p × 8 noise rates: Hybrid LER, PFWL3S LER,
  Lange LER, PM LER, MajHyb LER, MajPF LER, all with Wilson 95% CIs.
- `eval_hybrid_d7.log` — stdout from the eval script (per-p summary lines).
- `eval_hybrid_d7.py` — the eval script itself (ensemble averaging logic, ckpt
  loading, Wilson CI computation).
- `hybrid_model.py` — `HybridDecoder` + `DefectGNNLayer` source (mirror of
  `train/hybrid_model.py`).
- `hybrid_seed{0,1,2}.log` — per-seed training stdout (180K-line each: in-training
  EVAL LER at every 10K-step checkpoint).
- `seed{0,1,2}/best_model.pt` — the three Hybrid CNN+GNN checkpoints (6.3 MB each)
  that the eval script averages.

## Headline result

| p | Hybrid LER (95% CI) | PFWL3S LER | Hyb vs PFWL3S |
|---|---|---|---|
| 0.0005 | 0.0000% [0.0000, 0.0038] | 0.0000% | overlap |
| 0.001  | 0.0000% [0.0000, 0.0038] | 0.0000% | overlap |
| 0.002  | 0.0140% [0.0083, 0.0235] | 0.0140% | overlap |
| 0.003  | 0.0740% [0.0590, 0.0929] | 0.0930% | overlap |
| 0.005  | 0.6640% [0.6155, 0.7163] | 0.6570% | overlap |
| 0.007  | 2.5110% [2.4158, 2.6098] | 2.4920% | overlap |
| 0.010  | 9.2660% [9.0878, 9.4473] | 9.1730% | overlap |
| 0.015  | 27.4360% [27.1603, 27.7134] | 27.3280% | overlap |

8/8 overlap → Hybrid statistically indistinguishable from PFWL3S at all noise
rates. Honest negative on the single-layer architectural-fusion hypothesis.

## Reproducing

From a fresh clone with the same environment (`train/requirements.txt` +
`torch_cluster`):

```bash
# Train 3 seeds (~7.5 H200-hr total)
for s in 0 1 2; do
  python train/train_seeded_hybrid.py --seed $s --distance 7 --hidden_dim 384 \
    --steps 160000 --batch 128 --noise_rate 0.007 \
    --alpha_kl 0.7 --alpha_bce 0.3 \
    --ckpt checkpoints/hybrid_d7_seed$s
done

# Eval (~10 min, 100K shots/p × 8 p × 3 decoders)
python bench/results/h200_main/hybrid_d7_3seed/eval_hybrid_d7.py
```
