"""Take an IBM SamplerV2 result, convert measurements to Stim format, decode."""
from __future__ import annotations
import json, math, sys
import numpy as np
import stim
import pymatching


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    spread = (z / denom) * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return p, max(0.0, center - spread), min(1.0, center + spread)


def bitarray_packed_to_bools(packed: np.ndarray, n_shots: int, n_bits: int,
                             bit_order: str = "lsb_first") -> np.ndarray:
    """Convert SamplerV2's packed BitArray uint8s into (n_shots, n_bits) bool.

    Qiskit's BitArray packs bytes MSB-first (byte at index 0 contains the
    HIGH bits of the classical register), and bits within each byte are
    little-endian (bit 0 of byte k holds creg bit k*8). So we reverse byte
    order first, then unpackbits little-endian, then keep the first n_bits.
    """
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.ndim == 1:
        packed = packed[None, :]
    # Reverse byte order along axis=1, then unpack little-endian within bytes
    rev = packed[:, ::-1]
    out = np.unpackbits(rev, axis=1, bitorder='little')
    return out[:, :n_bits].astype(bool)


def run_decoders(circ: stim.Circuit, measurements: np.ndarray,
                 pathfinder_ckpt: str | None = None,
                 dem_noise_p: float = 0.005,
                 distance: int = 5, rounds: int = 5) -> dict:
    m2d = circ.compile_m2d_converter()
    detectors, observables = m2d.convert(measurements=measurements, separate_observables=True)
    n_shots = detectors.shape[0]
    dem_circ = stim.Circuit.generated(
        'surface_code:rotated_memory_z', distance=distance, rounds=rounds,
        after_clifford_depolarization=dem_noise_p,
        before_measure_flip_probability=dem_noise_p,
        after_reset_flip_probability=dem_noise_p,
        before_round_data_depolarization=dem_noise_p,
    )
    dem = dem_circ.detector_error_model(decompose_errors=True)
    pm = pymatching.Matching.from_detector_error_model(dem)
    pm_pred = pm.decode_batch(detectors).astype(np.uint8)
    pm_wrong = np.any(pm_pred != observables, axis=1)
    pm_k = int(pm_wrong.sum())
    pm_ler, pm_lo, pm_hi = wilson_ci(pm_k, n_shots)
    result = {
        "n_shots": n_shots,
        "n_detectors": int(detectors.shape[1]),
        "detector_flip_rate": float(detectors.mean()),
        "observable_flip_rate": float(observables.mean()),
        "pymatching": {"ler": pm_ler, "ci": [pm_lo, pm_hi], "errors": pm_k},
    }
    if pathfinder_ckpt:
        import torch
        sys.path.insert(0, 'train')
        from model import NeuralDecoder

        class PathfinderMapper:
            def __init__(self, circuit):
                nd = circuit.num_detectors
                coords = circuit.get_detector_coordinates()
                ac = np.array([coords[i] for i in range(nd)])
                sp, tm = ac[:, :-1], ac[:, -1]
                tu = np.sort(np.unique(tm))
                xu = np.sort(np.unique(sp[:, 0]))
                yu = np.sort(np.unique(sp[:, 1])) if sp.shape[1] > 1 else np.array([0.0])
                self.grid = (len(tu), len(yu), len(xu))
                tm_m = {v: i for i, v in enumerate(tu)}
                xm = {v: i for i, v in enumerate(xu)}
                ym = {v: i for i, v in enumerate(yu)}
                di = np.zeros((nd, 3), dtype=np.int64)
                for did in range(nd):
                    c = coords[did]
                    di[did] = [tm_m[c[-1]], ym.get(c[1], 0) if len(c) > 2 else 0, xm[c[0]]]
                self.det_idx = di
                self.nd = nd
            def to_tensor(self, det):
                B = det.shape[0]
                T, H, W = self.grid
                t = torch.zeros(B, 1, T, H, W, dtype=torch.float32)
                d = torch.from_numpy(det.astype(np.float32))
                for i in range(self.nd):
                    t[:, 0, self.det_idx[i, 0], self.det_idx[i, 1], self.det_idx[i, 2]] = d[:, i]
                return t

        device = torch.device('cpu')
        pfm = PathfinderMapper(circ)
        ck = torch.load(pathfinder_ckpt, weights_only=False, map_location=device)
        model = NeuralDecoder(ck['config']).to(device)
        model.load_state_dict(ck['model_state_dict'])
        model.train(False)  # inference mode
        det_u8 = detectors.astype(np.uint8)
        pf_pred = np.zeros_like(observables)
        chunk = 200
        with torch.no_grad():
            for i in range(0, n_shots, chunk):
                end = min(i+chunk, n_shots)
                t = pfm.to_tensor(det_u8[i:end]).to(device)
                logits = model(t).cpu().numpy()
                pf_pred[i:end] = (logits > 0).astype(np.uint8)
        pf_wrong = np.any(pf_pred != observables, axis=1)
        pf_k = int(pf_wrong.sum())
        pf_ler, pf_lo, pf_hi = wilson_ci(pf_k, n_shots)
        result["pathfinder"] = {
            "ckpt": pathfinder_ckpt, "ler": pf_ler,
            "ci": [pf_lo, pf_hi], "errors": pf_k,
        }
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibm-result", required=True,
                    help="JSON saved by submit_ibm.py --poll (has shots_array_packed)")
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--pathfinder", help="Path to Pathfinder checkpoint .pt file")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    obj = json.load(open(args.ibm_result))
    r = obj["result"]
    packed = np.array(r["shots_array_packed"], dtype=np.uint8)
    measurements = bitarray_packed_to_bools(packed, r["n_shots"], r["n_bits"])
    print(f"IBM measurements: shape={measurements.shape} dtype={measurements.dtype}")
    print(f"Per-bit flip rate (mean): {measurements.mean(axis=0)[:10]}")
    circ = stim.Circuit.generated('surface_code:rotated_memory_z',
                                  distance=args.distance, rounds=args.rounds)
    out = run_decoders(circ, measurements,
                       pathfinder_ckpt=args.pathfinder,
                       distance=args.distance, rounds=args.rounds)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
