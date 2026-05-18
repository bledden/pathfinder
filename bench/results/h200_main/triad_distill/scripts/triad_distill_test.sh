#!/usr/bin/env bash
# Single-seed cheap test: PFWL3S student distilled from Triad teacher
# at d=7 H=384 160K steps p=0.007. ~4 hr.
set -e
LOG=/workspace/triad_distill_test.log
echo "=== Triad-distill seed 0 START $(date -u) ===" | tee $LOG
python /workspace/train_seeded_wide_long_triad.py --seed 0 \
  --distance 7 --hidden_dim 384 --steps 160000 \
  --batch 128 --noise_rate 0.007 \
  --alpha_kl 0.7 --alpha_bce 0.3 \
  --ckpt /workspace/persist/checkpoints/pathfinder_triad_distill_d7_seed0 \
  --pf_teacher_ckpts \
    /workspace/persist/checkpoints/pathfinder_wide_long_d7/best_model.pt \
    /workspace/persist/checkpoints/pathfinder_wide_long_d7_seed1/best_model.pt \
    /workspace/persist/checkpoints/pathfinder_wide_long_d7_seed2/best_model.pt \
  --eval_interval 10000 --log_interval 1000 \
  2>&1 | tee -a $LOG
echo "=== Triad-distill seed 0 DONE $(date -u) ===" | tee -a $LOG
