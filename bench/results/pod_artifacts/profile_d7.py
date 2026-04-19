"""Profile d=7 B=1024 to find the actual latency bottleneck."""
import sys
sys.path.insert(0, "/workspace/pathfinder/train")
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from model import NeuralDecoder

torch.set_float32_matmul_precision("high")

ck = torch.load("/workspace/pathfinder/train/checkpoints/d7_final/best_model.pt", weights_only=False, map_location="cuda")
m = NeuralDecoder(ck["config"]).cuda().eval()
m.load_state_dict(ck["model_state_dict"])
m = m.half()
mc = torch.compile(m, mode="max-autotune")

x = torch.randint(0, 2, (1024, 1, 7, 7, 7), dtype=torch.float16, device="cuda")

# Warmup
with torch.no_grad():
    for _ in range(100): _ = mc(x)
torch.cuda.synchronize()

# Profile
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    with record_function("forward"):
        with torch.no_grad():
            for _ in range(20):
                _ = mc(x)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
