#!/usr/bin/env bash
set -e
LOG=/workspace/h512_seeds_12.log
PF_TEACHERS="/workspace/persist/checkpoints/pathfinder_wide_long_d7/best_model.pt /workspace/persist/checkpoints/pathfinder_wide_long_d7_seed1/best_model.pt /workspace/persist/checkpoints/pathfinder_wide_long_d7_seed2/best_model.pt"
echo "=== H=512 seeds 1+2 START $(date -u) ===" | tee -a $LOG
for SEED in 1 2; do
  CKPT=/workspace/persist/checkpoints/pathfinder_triad_h512_d7_seed${SEED}
  echo "=== seed=$SEED START $(date -u) ===" | tee -a $LOG
  python3 /workspace/train_seeded_wide_long_triad.py --seed $SEED \
    --distance 7 --hidden_dim 512 --steps 160000 \
    --batch 96 --noise_rate 0.007 \
    --muon_lr 0.002 --alpha_kl 0.7 --alpha_bce 0.3 \
    --ckpt $CKPT \
    --pf_teacher_ckpts $PF_TEACHERS \
    --eval_interval 10000 --log_interval 1000 \
    2>&1 | tee /workspace/h512_triad_seed${SEED}.log | tail -50 >> $LOG
  echo "=== seed=$SEED DONE $(date -u) ===" | tee -a $LOG
done
echo "=== ALL DONE $(date -u), running eval ===" | tee -a $LOG
python3 /workspace/eval_triad_h512.py 2>&1 | tee /workspace/eval_triad_h512.log | tee -a $LOG
echo "=== FINAL DONE $(date -u) ===" | tee -a $LOG
