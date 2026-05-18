"""Measure Lange's GNN decoder latency on H200."""
import sys, os, time, json
sys.path.insert(0, "/workspace/pathfinder/train")
sys.path.insert(0, "/workspace/GNN_decoder")
import numpy as np, torch, stim
from torch_geometric.nn import knn_graph
from src.gnn_models import GNN_7

device = torch.device("cuda")


class LangeWrapper:
    def __init__(self, d, d_t, m=10, power=2):
        self.d = d; self.d_t = d_t; self.m = m; self.power = power
        self.model = GNN_7(hidden_channels_GCN=[32,128,256,512,512,256,256],
                           hidden_channels_MLP=[256,128,64], num_classes=1).to(device).eval()
        ck = torch.load(f"/workspace/GNN_decoder/models/circuit_level_noise/d{d}/d{d}_d_t_{d_t}.pt",
                        weights_only=False, map_location=device)
        self.model.load_state_dict(ck["model"])
        self.sig = torch.nn.Sigmoid()

    def init_from_circuit(self, circuit):
        coords = circuit.get_detector_coordinates()
        dc = np.array(list(coords.values())); dc[:, :2] = dc[:, :2] / 2
        self.dc = dc.astype(np.uint8)
        sz = self.d + 1
        sx = np.zeros((sz, sz), dtype=np.uint8); sx[::2, 1:sz-1:2] = 1; sx[1::2, 2::2] = 1
        smz = np.rot90(sx) * 3
        self.smask = np.dstack([sx + smz] * (self.d_t + 1))

    def predict_batch(self, det):
        B = det.shape[0]
        any_flip = np.sum(det, axis=1) != 0
        if not np.any(any_flip): return np.zeros((B,1), dtype=np.uint8)
        det_nt = det[any_flip]
        mask = np.repeat(self.smask[None, ...], det_nt.shape[0], 0)
        s3d = np.zeros_like(mask)
        s3d[:, self.dc[:,1], self.dc[:,0], self.dc[:,2]] = det_nt
        s3d[np.nonzero(s3d)] = mask[np.nonzero(s3d)]
        s3d = s3d.astype(np.float32)
        inds = np.nonzero(s3d); defs = s3d[inds]
        inds_t = np.transpose(np.array(inds))
        xd = defs == 1; zd = defs == 3
        nf = np.zeros((defs.shape[0], 6), dtype=np.float32)
        nf[xd, 0] = 1; nf[xd, 2:] = inds_t[xd]
        nf[zd, 1] = 1; nf[zd, 2:] = inds_t[zd]
        x = torch.tensor(nf[:, [0,1,3,4,5]]).to(device)
        batch = torch.tensor(nf[:,2]).long().to(device)
        pos = x[:,2:]
        ei = knn_graph(pos, self.m, batch=batch)
        dist = torch.sqrt(((pos[ei[0]] - pos[ei[1]])**2).sum(dim=1, keepdim=True))
        ea = 1.0 / (dist ** self.power)
        with torch.no_grad():
            out = self.model(x, ei, batch, ea)
        return (self.sig(out) > 0.5).cpu().numpy().astype(np.uint8)


def time_lange(d, p, B, n_reps=5, warmup=2):
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)
    sampler = circuit.compile_detector_sampler()
    lw = LangeWrapper(d, d)
    lw.init_from_circuit(circuit)

    # Warm up
    for _ in range(warmup):
        det, _ = sampler.sample(B, separate_observables=True)
        det = det.astype(np.uint8)
        _ = lw.predict_batch(det)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_reps):
        det, _ = sampler.sample(B, separate_observables=True)
        det = det.astype(np.uint8)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = lw.predict_batch(det)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times)//2]  # median


def main():
    results = {}
    for d in [3, 5, 7]:
        for p in [0.003, 0.007, 0.010]:
            key = f"d{d}_p{p}"
            print(f"=== {key} ===", flush=True)
            for B in [1, 64, 1024]:
                t = time_lange(d, p, B, n_reps=5)
                us_per_syn = t * 1e6 / B
                print(f"  B={B:>4}: {t*1e3:.2f} ms ({us_per_syn:.2f} us/syn)", flush=True)
                results[f"{key}_B{B}"] = {"total_ms": t*1e3, "us_per_syn": us_per_syn}
    with open("/workspace/persist/results/lange_latency.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved", flush=True)


if __name__ == "__main__":
    main()
