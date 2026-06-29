"""Leak-free d=7 Table-1 re-evaluation (resolves the selection-on-test issue).

The original run_final_eval.py reports, for d=7 at each noise rate, the MIN LER over
4 candidate checkpoints + their ensemble, computed on the same 100K shots it reports
(selection on the test set). This script removes that bias:

  - selection is done on a VALIDATION sample (seed=1),
  - the selected option is then reported ONCE on a DISJOINT TEST sample (seed=2),
  - it also reports the single deployable checkpoint (d7_final) on the test set
    (the honest "what you would actually ship"),
  - and every candidate's test LER, for transparency.

3-parameter circuit-level noise, matching Table 1 / run_final_eval's CDS.
Run on CUDA (or CPU). Output: bench/results/h200_main/clean_d7_eval.json
"""
import sys, os, json, math
import numpy as np, torch, stim, pymatching
sys.path.insert(0, "train")
from model import NeuralDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"
N_VAL, N_TEST = 100000, 100000
PVALS = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015]
CANDS = [
    ("d7_final", "train/checkpoints/d7_final/best_model.pt"),   # the deployable single ckpt
    ("d7_p01",   "train/checkpoints/d7_p01/best_model.pt"),
    ("d7_mixed", "train/checkpoints/d7_mixed/best_model.pt"),
    ("d7_p015",  "train/checkpoints/d7_p015/best_model.pt"),
]


def wilson(ne, nt, z=1.96):
    if nt == 0: return [0.0, 0.0]
    p = ne / nt; d = 1 + z * z / nt
    c = (p + z * z / (2 * nt)) / d
    h = z * math.sqrt(p * (1 - p) / nt + z * z / (4 * nt * nt)) / d
    return [max(0.0, c - h), c + h]


class CDS:  # verbatim mapping from run_final_eval.py (same input format the model expects)
    def __init__(self, d, r, p):
        self.circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=d, rounds=r,
            after_clifford_depolarization=p, before_measure_flip_probability=p,
            after_reset_flip_probability=p)
        self.nd = self.circ.num_detectors
        coords = self.circ.get_detector_coordinates()
        ac = np.array([coords[i] for i in range(self.nd)])
        sp, tm = ac[:, :-1], ac[:, -1]
        tu = np.sort(np.unique(tm)); xu = np.sort(np.unique(sp[:, 0]))
        yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
        self.grid = (len(tu), len(yu), len(xu))
        tm_m = {v: i for i, v in enumerate(tu)}
        xm = {v: i for i, v in enumerate(xu)}; ym = {v: i for i, v in enumerate(yu)}
        self.d2g = {}
        for did in range(self.nd):
            c = coords[did]
            self.d2g[did] = (tm_m[c[-1]], ym.get(c[1], 0) if len(c) > 2 else 0, xm[c[0]])

    def tensor(self, det):
        B = det.shape[0]; T, H, W = self.grid
        t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
        for did, (gi, gj, gk) in self.d2g.items():
            if gi < T and gj < H and gk < W and did < det.shape[1]:
                t[:, 0, gi, gj, gk] = torch.from_numpy(det[:, did].astype(np.float32))
        return t

    def sample(self, n, seed):
        return self.circ.compile_detector_sampler(seed=seed).sample(n, separate_observables=True)


def logits_of(model, ds, det, n, bs=2000):
    parts = []
    for i in range(0, n, bs):
        with torch.no_grad():
            parts.append(model(ds.tensor(det[i:i + bs]).to(device)).cpu())
    return torch.cat(parts, 0)


def ler_from_logits(lg, obs):
    lab = torch.from_numpy(obs.astype(np.float32))
    return ((lg > 0).float() != lab).any(dim=1).sum().item() / len(obs)


def main():
    print(f"device={device}  N_val={N_VAL} N_test={N_TEST}", flush=True)
    models = {}
    for nm, pt in CANDS:
        ck = torch.load(pt, weights_only=False, map_location="cpu")
        m = NeuralDecoder(ck["config"]).to(device); m.load_state_dict(ck["model_state_dict"]); m.eval()
        models[nm] = m
        print(f"  loaded {nm}", flush=True)

    out = {"_note": "Leak-free d=7: select on val (seed=1), report on disjoint test (seed=2). "
                    "3-param noise. 'selected'=val-chosen option reported on test; "
                    "'deployable'=d7_final on test.", "rows": []}
    for p in PVALS:
        ds = CDS(7, 7, p)
        dv, ov = ds.sample(N_VAL, seed=1)
        dt, ot = ds.sample(N_TEST, seed=2)
        # per-candidate logits on val + test
        val_lg = {nm: logits_of(models[nm], ds, dv, N_VAL) for nm, _ in CANDS}
        test_lg = {nm: logits_of(models[nm], ds, dt, N_TEST) for nm, _ in CANDS}
        val_ler = {nm: ler_from_logits(val_lg[nm], ov) for nm, _ in CANDS}
        test_ler = {nm: ler_from_logits(test_lg[nm], ot) for nm, _ in CANDS}
        # ensemble (logit mean)
        ens_val = ler_from_logits(sum(val_lg.values()) / len(CANDS), ov)
        ens_test = ler_from_logits(sum(test_lg.values()) / len(CANDS), ot)
        val_ler["ensemble"] = ens_val; test_ler["ensemble"] = ens_test
        # SELECT on val, REPORT on test
        sel = min(val_ler, key=val_ler.get)
        sel_test = test_ler[sel]
        dep_test = test_ler["d7_final"]
        # PM on test
        pm = pymatching.Matching.from_detector_error_model(
            ds.circ.detector_error_model(decompose_errors=True))
        pm_test = int(np.any(pm.decode_batch(dt) != ot, axis=1).sum()) / N_TEST
        row = {"p": p,
               "selected_on_val": sel,
               "selected_test_ler": sel_test, "selected_test_ci": wilson(round(sel_test * N_TEST), N_TEST),
               "deployable_d7final_test_ler": dep_test, "deployable_test_ci": wilson(round(dep_test * N_TEST), N_TEST),
               "pm_test_ler": pm_test, "pm_test_ci": wilson(round(pm_test * N_TEST), N_TEST),
               "all_test_ler": test_ler, "all_val_ler": val_ler}
        out["rows"].append(row)
        def verdict(x):
            return "WIN" if x < pm_test else ("TIE" if abs(x - pm_test) < 1e-9 else "LOSS")
        print(f"  p={p}: selected({sel})={sel_test:.5f} [{verdict(sel_test)}]  "
              f"deployable(d7_final)={dep_test:.5f} [{verdict(dep_test)}]  PM={pm_test:.5f}", flush=True)

    json.dump(out, open("bench/results/h200_main/clean_d7_eval.json", "w"), indent=2)
    print("wrote bench/results/h200_main/clean_d7_eval.json", flush=True)


if __name__ == "__main__":
    main()
