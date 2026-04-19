#!/bin/bash
# Full queue: fixed_d5 → fixed_d7 → distill_d5 → distill_d7

echo "=== waiting for d=5 fixed noise ==="
while pgrep -f 'train_fixed_noise.*distance 5' > /dev/null; do sleep 30; done
echo "=== d=5 fixed done ==="

echo "=== starting d=7 fixed ==="
cd /workspace && python3 -u train_fixed_noise.py --distance 7 --hidden_dim 256 --steps 80000 --noise_rate 0.007 --ckpt /workspace/pathfinder/train/checkpoints/fixed_d7 > train_fixed_d7.log 2>&1
echo "=== d=7 fixed done ==="

echo "=== starting distill d=5 ==="
python3 -u train_distill_lange.py --distance 5 --hidden_dim 256 --steps 80000 --batch 256 --noise_rate 0.003 --ckpt /workspace/pathfinder/train/checkpoints/distill_d5 > train_distill_d5.log 2>&1
echo "=== distill d=5 done ==="

echo "=== starting distill d=7 ==="
python3 -u train_distill_lange.py --distance 7 --hidden_dim 256 --steps 80000 --batch 256 --noise_rate 0.003 --ckpt /workspace/pathfinder/train/checkpoints/distill_d7 > train_distill_d7.log 2>&1
echo "=== all done ==="
