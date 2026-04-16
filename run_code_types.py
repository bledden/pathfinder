"""Train and evaluate neural decoders on different code types.

Item 17: demonstrate generalization beyond rotated surface code.
Tests: color code, unrotated surface code, rotated surface code (X basis).
"""
import sys, os, time, math, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, "train")
from model import NeuralDecoder, DecoderConfig
from muon import SingleDeviceMuonWithAuxAdam
import stim
import pymatching

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CodeTypeDataset:
    def __init__(self, code_type, distance, rounds, physical_error_rate, batch_size=512):
        self.circuit = stim.Circuit.generated(
            code_type, distance=distance, rounds=rounds,
            after_clifford_depolarization=physical_error_rate,
            before_measure_flip_probability=physical_error_rate,
            after_reset_flip_probability=physical_error_rate,
        )
        self.sampler = self.circuit.compile_detector_sampler()
        self.n_detectors = self.circuit.num_detectors
        self.n_observables = self.circuit.num_observables
        self.batch_size = batch_size
        coords = self.circuit.get_detector_coordinates()
        all_c = np.array([coords[i] for i in range(self.n_detectors)])
        spatial = all_c[:, :-1]
        temporal = all_c[:, -1]
        t_u = np.sort(np.unique(temporal))
        x_u = np.sort(np.unique(spatial[:, 0]))
        y_u = np.sort(np.unique(spatial[:, 1])) if spatial.shape[1] > 1 else np.array([0])
        t_m = {v: i for i, v in enumerate(t_u)}
        x_m = {v: i for i, v in enumerate(x_u)}
        y_m = {v: i for i, v in enumerate(y_u)}
        self.grid_shape = (len(t_u), len(y_u), len(x_u))
        self.det_to_grid = {}
        for did in range(self.n_detectors):
            c = coords[did]
            self.det_to_grid[did] = (t_m[c[-1]], y_m.get(c[1], 0) if len(c) > 2 else 0, x_m[c[0]])

    def sample(self, batch_size=None):
        bs = batch_size or self.batch_size
        det, obs = self.sampler.sample(shots=bs, separate_observables=True)
        T, H, W = self.grid_shape
        tensor = torch.zeros(bs, 1, T, H, W, dtype=torch.float32)
        for did, (gi, gj, gk) in self.det_to_grid.items():
            if gi < T and gj < H and gk < W and did < det.shape[1]:
                tensor[:, 0, gi, gj, gk] = torch.from_numpy(det[:, did].astype(np.float32))
        return tensor, torch.from_numpy(obs.astype(np.float32))


def train_and_compare(code_type, d, steps=40000, target_p=0.007):
    print("\n" + "=" * 60, flush=True)
    print("CODE TYPE: {} d={}".format(code_type, d), flush=True)
    print("=" * 60, flush=True)

    ds = CodeTypeDataset(code_type, d, d, target_p)
    T, H_g, W_g = ds.grid_shape
    print("Grid: {}x{}x{}, {} det, {} obs".format(T, H_g, W_g, ds.n_detectors, ds.n_observables), flush=True)

    config = DecoderConfig(distance=d, rounds=d, hidden_dim=128, n_observables=ds.n_observables)
    model = NeuralDecoder(config).to(device)
    print("{} params".format(sum(p.numel() for p in model.parameters())), flush=True)

    w2d = [p for p in model.parameters() if p.ndim == 2 and p.requires_grad]
    other = [p for p in model.parameters() if p.ndim != 2 and p.requires_grad]
    opt = SingleDeviceMuonWithAuxAdam([
        dict(params=w2d, lr=0.02, momentum=0.95, use_muon=True),
        dict(params=other, lr=3e-3, betas=(0.9, 0.999), eps=1e-8, use_muon=False),
    ])

    # Curriculum
    def get_rate(step):
        f = step / max(steps, 1)
        if f < 0.1: return target_p * 0.3
        elif f < 0.4: return target_p * (0.3 + 0.4 * (f - 0.1) / 0.3)
        else: return target_p * (0.7 + 0.3 * (f - 0.4) / 0.6)

    rates = sorted(set(round(get_rate(s), 4) for s in range(0, steps, steps // 30)) | {target_p})
    samplers = {p: CodeTypeDataset(code_type, d, d, max(p, 1e-6), batch_size=512) for p in rates}

    model.train()
    best_ler = 1.0
    t0 = time.time()

    for step in range(steps):
        p = get_rate(step)
        ds_t = samplers[min(samplers.keys(), key=lambda k: abs(k - p))]
        syn, lab = ds_t.sample()
        syn, lab = syn.to(device), lab.to(device)
        loss = F.binary_cross_entropy_with_logits(model(syn), lab)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sc = (step / 1000 if step < 1000 else 0.5 * (1 + math.cos(math.pi * (step - 1000) / (steps - 1000))))
        for pg in opt.param_groups:
            pg["lr"] = (0.02 if pg["use_muon"] else 3e-3) * sc

        if step % 10000 == 0:
            print("  step {:>6}/{} loss={:.4f}".format(step, steps, loss.item()), flush=True)
        if step > 0 and step % 10000 == 0:
            model.eval_mode = True
            evds = CodeTypeDataset(code_type, d, d, target_p, batch_size=512)
            te, tt = 0, 0
            for _ in range(20):
                se, le = evds.sample(500)
                se, le = se.to(device), le.to(device)
                with torch.no_grad():
                    te += ((model(se) > 0).float() != le).any(dim=1).sum().item()
                    tt += 500
            ler = te / tt
            best_ler = min(best_ler, ler)
            print("  EVAL LER={:.6f} best={:.6f}".format(ler, best_ler), flush=True)
            model.eval_mode = False

    # Final comparison
    model.eval()
    n_ev = 20000
    evds = CodeTypeDataset(code_type, d, d, target_p, batch_size=1000)
    te, tt = 0, 0
    for _ in range(n_ev // 1000):
        se, le = evds.sample(1000)
        se, le = se.to(device), le.to(device)
        with torch.no_grad():
            te += ((model(se) > 0).float() != le).any(dim=1).sum().item()
            tt += 1000
    neural_ler = te / tt

    circ = stim.Circuit.generated(code_type, distance=d, rounds=d,
        after_clifford_depolarization=target_p,
        before_measure_flip_probability=target_p, after_reset_flip_probability=target_p)
    det, obs = circ.compile_detector_sampler().sample(shots=n_ev, separate_observables=True)
    dem = circ.detector_error_model(decompose_errors=True)
    pm_ler = np.sum(np.any(pymatching.Matching.from_detector_error_model(dem).decode_batch(det) != obs, axis=1)) / n_ev

    winner = "NEURAL" if neural_ler < pm_ler else "PM"
    print("  Neural={:.6f} PM={:.6f} ratio={:.2f}x {}  time={:.0f}s".format(
        neural_ler, pm_ler, neural_ler / pm_ler if pm_ler > 0 else 999, winner, time.time() - t0), flush=True)
    return neural_ler, pm_ler


results = {}
for code_type, d in [("color_code:memory_xyz", 3),
                      ("surface_code:unrotated_memory_z", 5),
                      ("surface_code:rotated_memory_x", 5)]:
    n, p = train_and_compare(code_type, d, steps=40000)
    results[code_type] = {"neural": n, "pm": p}

print("\n" + "=" * 60, flush=True)
print("CODE TYPE RESULTS", flush=True)
print("=" * 60, flush=True)
for code, r in results.items():
    ratio = r["neural"] / r["pm"] if r["pm"] > 0 else 999
    w = "NEURAL" if r["neural"] < r["pm"] else "PM"
    print("  {}: N={:.6f} PM={:.6f} ({:.2f}x) {}".format(code, r["neural"], r["pm"], ratio, w), flush=True)
