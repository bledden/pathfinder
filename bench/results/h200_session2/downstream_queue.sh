#!/bin/bash
cd /workspace
echo "$(date): waiting for transformer_d7 training"
while pgrep -f "train_transformer" > /dev/null 2>&1; do sleep 60; done
echo "$(date): transformer_d7 done"

echo "$(date): starting fixed_d7 baseline"
python3 -u train_fixed_noise.py --distance 7 --hidden_dim 256 --steps 80000 --noise_rate 0.007 --ckpt /workspace/pathfinder/train/checkpoints/fixed_d7 > train_fixed_d7.log 2>&1 || echo "$(date): fixed_d7 failed"
echo "$(date): fixed_d7 done"

echo "$(date): starting distill_d5"
python3 -u train_distill_lange.py --distance 5 --hidden_dim 256 --steps 80000 --batch 256 --noise_rate 0.003 --ckpt /workspace/pathfinder/train/checkpoints/distill_d5 > train_distill_d5.log 2>&1 || echo "$(date): distill_d5 failed"
echo "$(date): distill_d5 done"

echo "$(date): starting distill_d7"
python3 -u train_distill_lange.py --distance 7 --hidden_dim 256 --steps 80000 --batch 256 --noise_rate 0.003 --ckpt /workspace/pathfinder/train/checkpoints/distill_d7 > train_distill_d7.log 2>&1 || echo "$(date): distill_d7 failed"
echo "$(date): distill_d7 done"

echo "$(date): final comparison"
python3 -u final_comparison.py > final_comparison.log 2>&1 || echo "$(date): final_comparison failed"
python3 -u bench_lange_latency.py > lange_latency.log 2>&1 || echo "$(date): lange_latency failed"
echo "$(date): ALL DONE"
