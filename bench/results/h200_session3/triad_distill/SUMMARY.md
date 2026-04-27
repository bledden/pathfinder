# Triad-Distillation Experiment — Data Archive

**Experiment goal:** train a single PFWL3S decoder that strictly beats the Pathfinder-Triad ensemble (PFWL3S + Lange + PM majority vote) at d=7 — i.e., make a single CNN absorb the Triad's coverage advantage through knowledge distillation.

**Outcome:** the goal was NOT achieved. The Triad has an architectural advantage from independent error-mode coverage that no single PF student can absorb through any tested KD recipe. However, the experiments produced incremental improvements and a clean negative-result narrative.

## Recipes tested (in order)

| Recipe | Best individual LER (d=7 p=0.007) | 3-seed-avg LER | Triad LER | Verdict |
|---|---:|---:|---:|---|
| Original Lange-only PFWL3S (baseline, in paper) | 2.80% | 2.49% | 2.38% | Triad strict-wins |
| **Soft Triad-distill** (PF+Lange+PM teachers, soft mean target, from-scratch) | 2.71% | not run | — | hit ceiling, no breakthrough |
| **Hardlabel Triad-distill** (binary majority vote as label, from-scratch) | 2.71% | not run | — | identical ceiling to soft |
| **Warm-init Triad-distill** (init from existing PFWL3S, soft Triad teacher, 80K steps) | 2.51% (best of 3 seeds) | **2.507%** | **2.393%** | overlap CIs, Triad still lower point |
| **H=512 Triad-distill** (from-scratch, soft Triad, muon_lr=0.002) | 2.51% (best of 3 seeds) | **2.558%** | **2.405%** | Triad still strict-wins at p=0.010, 0.015 |
| **PF+PM-only KD** (drop Lange teacher, warm-init, 80K steps) | 2.57% (seed 0 only) | not run | — | similar ceiling, marginal benefit |
| **7-ckpt mega-ensemble** (warm-init 3 + H=512 3 + PF+PM 1, all averaged) | n/a | **2.458%** | **2.399%** | overlap CIs at p=0.007/0.010, Triad strict-win at p=0.015 |

## Per-rate eval at d=7 (100K shots, mega-ensemble — best PF result)

| p | PF (7-ckpt mega) | Lange | PM | Triad (Maj) | PF vs Triad |
|---|---:|---:|---:|---:|---|
| 0.0005 | 0.000% | 0.000% | 0.001% | 0.000% | tie |
| 0.001 | 0.000% | 0.000% | 0.001% | 0.000% | tie |
| 0.002 | 0.013% | 0.017% | 0.022% | 0.015% | overlap (PF lower) |
| 0.003 | 0.106% | 0.086% | 0.152% | 0.096% | overlap |
| 0.005 | 0.663% | 0.727% | 0.984% | 0.639% | overlap |
| 0.007 | **2.458%** | 2.956% | 3.366% | **2.399%** | overlap, Triad lower point |
| 0.010 | 8.837% | 10.764% | 10.307% | **8.554%** | overlap, Triad lower point |
| 0.015 | 26.337% | 30.200% | 27.163% | **25.499%** | **Triad STRICT WIN** (84 bp) |

## Strict-CI wins preserved
PFWL3S mega-ensemble strictly beats **Lange** at d=7 p ∈ {0.007, 0.010, 0.015} — same 3 strict-CI wins as the original PFWL3S in the paper. The improvement is in the absolute LER, not in the strict-win count.

## Directory structure

- `ckpts/` — 9 best_model.pt + final_model.pt files, ~88 MB total
  - `pathfinder_triad_distill_d7_seed0/` — soft Triad from-scratch, 2.71%
  - `pathfinder_triad_hardlabel_d7_seed0/` — hardlabel Triad from-scratch, 2.71%
  - `pathfinder_triad_warminit_d7_seed{0,1,2}/` — warm-init Triad, 2.62/2.60/2.51%
  - `pathfinder_triad_h512_d7_seed{0,1,2}/` — H=512 Triad, 2.51/2.56/2.78%
  - `pathfinder_pf_pm_d7_seed0/` — PF+PM-only KD, 2.57%
- `evals/` — JSON eval results + summary files
  - `ensemble_triad_warminit_d7.json` — warm-init 3-seed eval
  - `ensemble_triad_h512_d7.json` — H=512 3-seed eval
  - `ensemble_triad_megaensemble_d7.json` — 7-ckpt mega eval (best result)
- `training_logs/` — full per-seed training logs (LER trajectories)
- `scripts/` — training + eval Python scripts + bash queue/watcher scripts
- `archive.tar.gz` — original tarball pulled from pod (94 MB, can delete after verifying)

## Key training scripts

- `scripts/train_distill_triad.py` — main soft Triad-distill (3-teacher KD)
- `scripts/train_distill_triad_hardlabel.py` — hardlabel variant
- `scripts/train_distill_pf_pm.py` — Lange-dropped variant (PF+PM only)
- `scripts/train_seeded_*.py` — seeded wrappers for each
- `scripts/h512_*queue.sh`, `warminit_seeds_12_queue.sh` — sequential training queues
- `scripts/auto_continue_triad_v3.sh` — branching auto-watcher (3 phases)

## Cost spent on this arc
~$110 of GPU credits ($4/hr H200 SXM on RunPod).

## Outstanding follow-up options (if resumed later)
1. Train PF+PM seeds 1+2 → 3-seed avg ($16) — would marginally improve mega-ensemble
2. Try multi-noise mixture training at d=5/d=7 to break Triad ties at lower noise rates
3. Try a richer student architecture (residual head, attention) — may be the only way to fundamentally close the per-shot gap
4. Accept the result and document Triad's architectural superiority as a finding
