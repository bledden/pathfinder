#!/usr/bin/env bash
set -e
LOG=/workspace/warminit_seeds_12.log
PF_TEACHERS="/workspace/persist/checkpoints/pathfinder_wide_long_d7/best_model.pt /workspace/persist/checkpoints/pathfinder_wide_long_d7_seed1/best_model.pt /workspace/persist/checkpoints/pathfinder_wide_long_d7_seed2/best_model.pt"
echo "=== warm-init seeds 1+2 START $(date -u) ===" | tee -a $LOG
for SEED in 1 2; do
  CKPT=/workspace/persist/checkpoints/pathfinder_triad_warminit_d7_seed${SEED}
  INIT=/workspace/persist/checkpoints/pathfinder_wide_long_d7_seed${SEED}/best_model.pt
  echo "=== seed=$SEED init=$INIT START $(date -u) ===" | tee -a $LOG
  python3 /workspace/train_seeded_wide_long_triad.py --seed $SEED \
    --distance 7 --hidden_dim 384 --steps 80000 \
    --batch 128 --noise_rate 0.007 \
    --alpha_kl 0.7 --alpha_bce 0.3 \
    --init $INIT \
    --ckpt $CKPT \
    --pf_teacher_ckpts $PF_TEACHERS \
    --eval_interval 5000 --log_interval 1000 \
    2>&1 | tee /workspace/triad_warminit_seed${SEED}.log | tail -50 >> $LOG
  echo "=== seed=$SEED DONE $(date -u) ===" | tee -a $LOG
done
echo "=== ALL DONE $(date -u) ===" | tee -a $LOG
