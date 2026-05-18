# `bench/results/` directory layout

This file documents the structure of the results tree so reviewers and
reproducers can navigate it directly. Path references in `paper/pathfinder.md`
follow this layout.

## Top-level layout

```
bench/results/
├── STRUCTURE.md                ← (this file)
├── 100k_eval.txt               ← raw 100K-shot run logs from the original
├── comprehensive_eval.json       Table-1 evaluation (§5.1)
├── final_eval.json
├── h200_bench*.log             ← H200 SXM latency runs (§5.3)
├── h200_final_benchmark.json
├── h200_latency_*.json
├── h200_maxspeed*.{log,json}
├── pymatching_latency_m4.txt   ← Apple M4 PyMatching baseline (§5.3 Table 3c)
│
├── h200_lange_headtohead_{low,high}_p.json  ← §5.11 head-to-head data
├── h200_lange_headtohead.log
│
├── h200_session2/              ← initial H200 work (Lange head-to-head v1,
│   ├── train_distill_lange.py     Pathfinder-KD §5.13)
│   ├── run_lange_v3.py
│   └── ...
│
└── h200_main/                  ← consolidated H200 follow-up work
    ├── phase{2,3,4,5}/            sequential development chapters; phase2 =
    │                              §5.12 Triad data; phase3 = §5.13 distill-
    │                              as-fine-tune (negative); phase4 = α_kl
    │                              sweep (negative); phase5 = d=9 from-scratch
    │                              failures
    ├── tuned/                      tuned `finetune_d{3,5,7}` ckpts for §5.11
    │                               head-to-head with Lange
    ├── distill/                    §5.13 Pathfinder-KD distillation ckpts
    ├── hybrid/                     §5.14 modern-primitives hybrid (negative)
    ├── checkpoints/                §6.2 Muon ablation ckpts (d=3, d=7 AdamW)
    ├── tierA/, tierB/, tierBC/     audit-trail subdirectories preserving the
    │                               iterative development order — tierBC has
    │                               the §6.3 d=9 warm-init chain
    │                               (`distill_d9_p003_lowlr`, `distill_d9_p005_ft`,
    │                               `distill_d9_p007_ft`)
    ├── tierC1/                     headline PFWL3S ckpts (§5.13), PFWL3S-H256-d9
    │                               ckpts (§6.3), Triton-at-H=384 audit data
    │                               (M10), H=256 ensemble redo (M7), fine-tuned
    │                               Lange ckpt + eval (C2)
    └── triad_distill/              archived §5.13 Triad-distillation arc
                                    (~$110 follow-up compute) — 6 recipe
                                    variants + 7-ckpt mega-ensemble. Has its
                                    own SUMMARY.md.
```

## Conventions

- Each subdirectory typically contains `*_<seed>/best_model.pt` and `*_<seed>/final_model.pt` checkpoints. `best_model.pt` is the checkpoint saved when in-training eval reached its lowest LER; `final_model.pt` is the end-of-training state.
- Raw JSON eval results are named `ensemble_*.json` or `<analysis>_<distance>_<noise>.json`. The first key of each JSON typically describes (d, p) pairs as `d{d}_p{p}` with nested `pf_ler`, `lange_ler`, `pm_ler`, `majority_ler`, `*_ci` (Wilson 95%) entries.
- Training logs `*.log` are stdout captures from the corresponding training scripts; per-epoch eval LER lines are filtered with `grep "EVAL LER"`.
- File paths in `paper/pathfinder.md` reference this tree relative to the repo root.

## History note

The `h200_main/` directory was originally named `h200_session3/` during development; it was renamed in audit-pass-3 (commit after `becca08`) to remove internal-development jargon from the public release. All references in the paper, README, audit doc, and `figures/make_figures.py` were updated accordingly.
