"""
Launch training of narrower d=7 model (H=128 instead of 256) for latency/accuracy Pareto.
Shims Muon into torch.optim (torch 2.6 doesn't bundle it).
Writes checkpoint to /workspace/pathfinder/train/checkpoints/d7_narrow/
"""
import sys, torch, torch.optim
# Monkey-patch: add Muon to torch.optim so train.py's `from torch.optim import Muon` works
from muon import SingleDeviceMuon
torch.optim.Muon = SingleDeviceMuon

sys.path.insert(0, "/workspace/pathfinder/train")
from train import train
import argparse

class Args:
    distance = 7
    hidden_dim = 128     # narrower - half of 256
    steps = 60000        # slightly shorter than full run - we're probing the Pareto, not aiming for best
    batch_size = 512
    muon_lr = 0.02
    adam_lr = 1e-3
    noise_rate = 0.007
    log_interval = 500
    eval_interval = 5000
    eval_shots = 10000
    checkpoint_dir = "/workspace/pathfinder/train/checkpoints/d7_narrow"

train(Args())
