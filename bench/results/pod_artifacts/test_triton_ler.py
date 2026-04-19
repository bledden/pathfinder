"""Verify Triton swap doesn't change LER — use fixed seed for identical samples."""
import sys, copy, torch
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace")
from model import NeuralDecoder
from data import SyndromeDataset, DataConfig
from triton_directional import swap_to_triton

ck = torch.load("/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt", weights_only=False, map_location="cuda")
m_orig = NeuralDecoder(ck["config"]).cuda().eval()
m_orig.load_state_dict(ck["model_state_dict"])
m_triton = copy.deepcopy(m_orig)
swap_to_triton(m_triton)
m_triton.eval()

for p in [0.003, 0.007, 0.010]:
    ds = SyndromeDataset(DataConfig(distance=7, rounds=7, physical_error_rate=p))
    n = 10000
    batch = 1000
    errors_orig = 0
    errors_triton = 0
    disagreements = 0
    total = 0
    for i in range(n // batch):
        torch.manual_seed(1000 + i)
        synd, lab = ds.sample(batch)
        synd, lab = synd.cuda(), lab.cuda()
        with torch.no_grad():
            p_orig = (m_orig(synd) > 0).float()
            p_triton = (m_triton(synd) > 0).float()
        errors_orig += (p_orig != lab).any(dim=1).sum().item()
        errors_triton += (p_triton != lab).any(dim=1).sum().item()
        disagreements += (p_orig != p_triton).any(dim=1).sum().item()
        total += batch
    print(f"p={p:.3f}:  orig_LER={errors_orig/total:.5f}  triton_LER={errors_triton/total:.5f}  "
          f"disagree={disagreements}/{total} ({100*disagreements/total:.3f}%)")
