"""Shared core for the Section 5.7 phenomenological-noise generalization eval.

Phenomenological noise here = `before_round_data_depolarization` only: data-qubit
depolarizing errors each round with perfect syndrome measurement (no measurement
or reset flips, no gate-level depolarizing). This is the "phenom" model the
Section 5.7 generalization test uses, evaluating a Pathfinder checkpoint per
distance against PyMatching across d in {3,5,7} x p in {0.003..0.015}, 60K shots
per point, and writing a JSON matching bench/results/h200_main/tierB/.

Used by eval_phenomenological.py (canonical fine-tune ckpts) and
eval_phenom_table1.py (original Table-1 ckpts, per run_final_eval.py).

Reconstruction note: the original pod-side driver scripts were not preserved in
the repo; this faithfully regenerates the committed phenom_eval*.json within
shot-noise from the documented noise model + checkpoints. The committed JSONs
remain the canonical recorded H200 runs.
"""
import os, sys, json, math, argparse
import numpy as np, torch, stim, pymatching

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "train"))
from model import NeuralDecoder

# NOTE: the Apple MPS backend mis-evaluates DirectionalConv3d at large batch
# (e.g. d=5 gives 38% instead of 3% at batch=2000, though it is correct at
# batch<=500). We therefore use CUDA or CPU only, which are correct at any batch.
DEV = "cuda" if torch.cuda.is_available() else "cpu"
RATES = [0.003, 0.005, 0.007, 0.01, 0.015]
DISTS = [3, 5, 7]


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), c + h]


def phenom_circuit(d, p):
    # phenom: data-qubit depolarizing only, perfect measurement
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=d,
        before_round_data_depolarization=p)


class DetectorMapper:
    """Maps a (B, num_detectors) syndrome array onto the (B,1,T,H,W) grid the
    3D-CNN expects, using the circuit's detector coordinates. Same mapping the
    training/eval harness uses; grid is determined by detector coords only, so it
    is identical across noise models at a fixed (distance, rounds)."""
    def __init__(self, circ):
        nd = circ.num_detectors
        co = circ.get_detector_coordinates()
        ac = np.array([co[i] for i in range(nd)])
        sp, tm = ac[:, :-1], ac[:, -1]
        tu, xu, yu = np.sort(np.unique(tm)), np.sort(np.unique(sp[:, 0])), np.sort(np.unique(sp[:, 1]))
        self.grid = (len(tu), len(yu), len(xu))  # (T, H, W)
        tmm = {v: i for i, v in enumerate(tu)}
        xm = {v: i for i, v in enumerate(xu)}
        ym = {v: i for i, v in enumerate(yu)}
        self.di = np.zeros((nd, 3), dtype=np.int64)
        for k in range(nd):
            cc = co[k]
            self.di[k] = [tmm[cc[-1]], ym.get(cc[1], 0), xm[cc[0]]]
        self.nd = nd

    def to_tensor(self, det):
        B = det.shape[0]
        T, H, W = self.grid
        t = torch.zeros(B, 1, T, H, W)
        d = torch.from_numpy(det.astype(np.float32))
        for i in range(self.nd):
            t[:, 0, self.di[i, 0], self.di[i, 1], self.di[i, 2]] = d[:, i]
        return t


def eval_point(model, d, p, n, batch=2000):
    circ = phenom_circuit(d, p)
    mapper = DetectorMapper(circ)
    det, obs = circ.compile_detector_sampler().sample(n, separate_observables=True)
    det, obs = det.astype(np.uint8), obs.astype(np.uint8)
    pred = np.zeros_like(obs)
    for i in range(0, n, batch):
        with torch.no_grad():
            lg = model(mapper.to_tensor(det[i:i + batch]).to(DEV))
        pred[i:i + batch] = (lg > 0).cpu().numpy().astype(np.uint8)
    pf_wrong = int(np.any(pred != obs, axis=1).sum())
    pm = pymatching.Matching.from_detector_error_model(
        circ.detector_error_model(decompose_errors=True))
    pm_wrong = int(np.any(pm.decode_batch(det) != obs, axis=1).sum())
    return pf_wrong, pm_wrong


def run(ckpts, out_path, note, shots=60000, dists=None, rates=None):
    dists = dists or DISTS
    rates = rates or RATES
    print(f"device={DEV}  shots/point={shots}", flush=True)
    out = {"_note": note}
    for d in dists:
        ck = torch.load(ckpts[d], weights_only=False, map_location="cpu")
        m = NeuralDecoder(ck["config"])
        m.load_state_dict(ck["model_state_dict"])
        m.to(DEV).eval()
        for p in rates:
            pf, pm = eval_point(m, d, p, shots)
            key = f"d{d}_p{p}"
            out[key] = {"d": d, "p": p, "n": shots,
                        "pf_ler": pf / shots, "pf_ci": wilson(pf, shots),
                        "pm_ler": pm / shots}
            print(f"  {key}: PF {pf/shots:.6f}  PM {pm/shots:.6f}", flush=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print("wrote", out_path, flush=True)


def cli(default_ckpts, default_out, note):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--shots", type=int, default=60000)
    ap.add_argument("--distances", type=int, nargs="*", default=None)
    ap.add_argument("--rates", type=float, nargs="*", default=None)
    a = ap.parse_args()
    run(default_ckpts, a.out, note, a.shots, a.distances, a.rates)
