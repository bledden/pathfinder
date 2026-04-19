import sys, time
sys.path.insert(0, "/workspace/pathfinder/train")
import torch
from model import NeuralDecoder
from data import SyndromeDataset, DataConfig

ck = torch.load("/workspace/pathfinder/train/checkpoints/d7_narrow/best_model.pt", weights_only=False, map_location="cuda")
m = NeuralDecoder(ck["config"]).cuda().eval()
m.load_state_dict(ck["model_state_dict"])

print(f"narrow d=7 H=128 params={sum(p.numel() for p in m.parameters()):,}")
print(f"{'p':>8} {'narrow_LER':>12} {'full_LER':>12} {'PM_LER':>12} {'beats_PM':>10}")

paper_nums = {
    0.001: (0.00007, 0.00009),
    0.002: (0.00005, 0.00007),
    0.003: (0.00032, 0.00057),
    0.005: (0.00253, 0.00442),
    0.007: (0.01041, 0.01489),
    0.010: (0.04104, 0.05161),
    0.015: (0.15843, 0.17045),
}

for p, (full_ler, pm_ler) in paper_nums.items():
    ds = SyndromeDataset(DataConfig(distance=7, rounds=7, physical_error_rate=p, batch_size=500))
    errors = 0
    total = 0
    n_shots = 20000
    batch = 500
    for _ in range(n_shots // batch):
        synd, labels = ds.sample(batch)
        synd = synd.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            logits = m(synd)
            preds = (logits > 0).float()
        errors += (preds != labels).any(dim=1).sum().item()
        total += batch
    ler = errors / total
    beats = "YES" if ler < pm_ler else "no"
    print(f"{p:>8.4f} {ler:>12.5f} {full_ler:>12.5f} {pm_ler:>12.5f} {beats:>10}")
