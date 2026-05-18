#!/usr/bin/env bash
# Phase 2 G: Hybrid CNN+GNN — 3 seeds × 160K steps at d=7 H=384
# Same recipe as PFWL3S (Lange-teacher distillation at p=0.007)
set -e
LOG=/workspace/hybrid_queue.log
echo "=== Hybrid queue START $(date -u) ===" | tee $LOG
for SEED in 0 1 2; do
  echo "" | tee -a $LOG
  echo "=== Hybrid d=7 seed=$SEED START $(date -u) ===" | tee -a $LOG
  python3 /workspace/train_seeded_hybrid.py --seed $SEED \
    --distance 7 --hidden_dim 384 --steps 160000 \
    --batch 128 --noise_rate 0.007 \
    --alpha_kl 0.7 --alpha_bce 0.3 \
    --ckpt /workspace/persist/checkpoints/hybrid_d7_seed${SEED} \
    --eval_interval 10000 --log_interval 1000 \
    2>&1 | tee /workspace/hybrid_seed${SEED}.log | tail -50 >> $LOG
done
echo "=== Hybrid queue DONE $(date -u) ===" | tee -a $LOG
