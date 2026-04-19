"""Benchmark Lange's GNN inference latency on H200 — apples-to-apples with Pathfinder."""
import sys, time
sys.path.insert(0, "/workspace/GNN_decoder")
sys.path.insert(0, "/workspace/pathfinder/train")
import numpy as np, torch, stim
from torch_geometric.nn import knn_graph
from src.gnn_models import GNN_7

device = torch.device("cuda")


def build_lange(d, d_t):
    m = GNN_7(
        hidden_channels_GCN=[32, 128, 256, 512, 512, 256, 256],
        hidden_channels_MLP=[256, 128, 64], num_classes=1).to(device)
    ck = torch.load(f"/workspace/GNN_decoder/models/circuit_level_noise/d{d}/d{d}_d_t_{d_t}.pt",
                    weights_only=False, map_location=device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def sample_graphs(d, p, B):
    """Generate B syndromes and build Lange-style graph batch."""
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)
    s = c.compile_detector_sampler()
    det, _ = s.sample(B, separate_observables=True)
    det = det.astype(np.uint8)

    coords = c.get_detector_coordinates()
    dc = np.array(list(coords.values()))
    dc[:, :2] = dc[:, :2] / 2
    dc = dc.astype(np.uint8)
    sz = d + 1
    sx = np.zeros((sz, sz), dtype=np.uint8)
    sx[::2, 1:sz-1:2] = 1; sx[1::2, 2::2] = 1
    smz = np.rot90(sx) * 3
    mask = np.dstack([sx + smz] * (d + 1))

    mask_full = np.repeat(mask[None, ...], B, 0)
    s3d = np.zeros_like(mask_full)
    s3d[:, dc[:, 1], dc[:, 0], dc[:, 2]] = det
    s3d[np.nonzero(s3d)] = mask_full[np.nonzero(s3d)]
    s3d = s3d.astype(np.float32)

    inds = np.nonzero(s3d)
    defs = s3d[inds]
    inds = np.transpose(np.array(inds))
    xd = defs == 1; zd = defs == 3
    nf = np.zeros((defs.shape[0], 6), dtype=np.float32)
    nf[xd, 0] = 1; nf[xd, 2:] = inds[xd, ...]
    nf[zd, 1] = 1; nf[zd, 2:] = inds[zd, ...]
    x_cols = [0, 1, 3, 4, 5]
    x = torch.tensor(nf[:, x_cols]).to(device)
    batch = torch.tensor(nf[:, 2]).long().to(device)
    pos = x[:, 2:]
    ei = knn_graph(pos, 10, batch=batch)
    dist = torch.sqrt(((pos[ei[0],:] - pos[ei[1],:])**2).sum(dim=1, keepdim=True))
    ea = 1.0 / (dist ** 2 + 1e-8)
    return x, ei, batch, ea, B


def bench(m, x, ei, batch, ea, B, iters=200, warmup=30):
    with torch.no_grad():
        for _ in range(warmup):
            _ = m(x, ei, batch, ea)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = m(x, ei, batch, ea)
    torch.cuda.synchronize()
    ms_per_call = (time.perf_counter() - t0) * 1000 / iters
    us_per_syn = ms_per_call * 1000 / B
    return ms_per_call, us_per_syn


def main():
    print("Lange GNN latency on H200")
    print(f"{'d':>2} {'p':>6} {'B':>5} {'ms/call':>10} {'us/syn':>10}")
    for d in [3, 5, 7]:
        m = build_lange(d, d)
        for p in [0.003, 0.007]:
            for B in [64, 256, 1024]:
                try:
                    x, ei, bt, ea, B_nt = sample_graphs(d, p, B)
                    ms, us = bench(m, x, ei, bt, ea, B_nt)
                    print(f"{d:>2} {p:>6.3f} {B:>5d} {ms:>10.3f} {us:>10.2f}")
                except Exception as e:
                    print(f"{d:>2} {p:>6.3f} {B:>5d} FAIL: {type(e).__name__}")


if __name__ == "__main__":
    main()
