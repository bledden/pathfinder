"""Isolated subprocess worker for the qLDPC SlidingWindowDecoder.

WHY A SUBPROCESS
----------------
The streaming sliding-window decoder lives in the PyPI/GitHub package
``qldpcOrg/qLDPC`` whose *import name is also* ``qldpc`` — byte-identical to
THIS repository's local ``qldpc`` package. Because the repo deliberately ships
``qldpc`` as a namespace package (no ``qldpc/__init__.py``), installing the PyPI
``qldpc`` into the venv's site-packages makes the regular installed package WIN
over the local namespace package for every ``import qldpc.*`` — which silently
breaks the whole repo (``qldpc.foundation``, ``qldpc.zoo`` stop importing). The
two cannot coexist on one ``sys.path``.

The fix: install the PyPI ``qldpc`` into an ISOLATED directory
(``QLDPC_EXT_DIR``, default ``~/.venvs/braket/qldpc_ext_clean``) that is NOT on
the repo's import path, and run the sliding-window decode HERE, in a subprocess
whose ``sys.path`` is the venv site-packages + that isolated dir, with every
``pathfinder`` entry stripped. In this process ``import qldpc`` cleanly resolves
to the installed package; the repo is never imported.

G1 (DEM identity) PROVENANCE
----------------------------
The parent passes the shared DEM as its canonical text (``str(dem)``). This
worker reconstructs ``stim.DetectorErrorModel(text)`` — a faithful, byte-exact
round-trip under the SAME stim version the parent uses (the isolated dir carries
NO stim of its own, so the venv's stim is used in both processes). The decode
therefore runs against the byte-identical DEM. The parent adapter still holds
the ORIGINAL shared DEM object as ``.dem``, so the harness' G1 hash-identity
check (which inspects ``adapter.dem``) sees the real shared object: provenance is
the same-DEM text, never a weakened gate.

PROTOCOL (stdin JSON -> stdout JSON)
------------------------------------
stdin : {"dem_text": str, "window_size": int, "stride": int, "rounds": int,
         "dets_npy": path, "decoder_kwargs": {...}}
stdout: {"pred_npy": path, "n_windows": int, "n_obs": int}
"""
import json
import os
import sys

import numpy as np


def main():
    req = json.loads(sys.stdin.read())

    ext_dir = os.environ.get(
        "QLDPC_EXT_DIR",
        os.path.expanduser("~/.venvs/braket/qldpc_ext_clean"),
    )
    # Strip every repo path; add the isolated installed-qldpc dir. In this
    # process `import qldpc` resolves to the PyPI package, not the repo.
    sys.path = [p for p in sys.path if "Documents/pathfinder" not in p]
    if ext_dir not in sys.path:
        sys.path.append(ext_dir)

    import stim
    from qldpc.decoders import SlidingWindowDecoder

    dem = stim.DetectorErrorModel(req["dem_text"])
    dets = np.load(req["dets_npy"]).astype(np.uint8)

    rounds = int(req["rounds"])
    n_det = dem.num_detectors
    # The repo's BB memory lays detectors out as (rounds+1) equal time layers of
    # N = n_det/(rounds+1) detectors each (R syndrome rounds + 1 final-readout
    # layer), in time order. So detector d -> time d // N. The repo DEM attaches
    # no DETECTOR coordinates, hence the default coord-based time map cannot be
    # used; supply this explicit, structure-derived mapping.
    n_layers = rounds + 1
    if n_det % n_layers != 0:
        raise ValueError(
            f"sliding-window worker: n_det={n_det} not divisible by (rounds+1)="
            f"{n_layers}; cannot derive a per-round detector->time map."
        )
    per_layer = n_det // n_layers

    def detector_to_time(d):
        return d // per_layer

    swd = SlidingWindowDecoder(
        window_size=int(req["window_size"]),
        stride=int(req["stride"]),
        detector_to_time=detector_to_time,
        **req.get("decoder_kwargs", {}),
    )
    compiled = swd.compile_decoder_for_dem(dem)
    pred = np.asarray(compiled.decode_shots(dets)) % 2
    pred = pred.astype(bool)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    out_path = req["pred_npy"]
    np.save(out_path, pred)
    sys.stdout.write(json.dumps({
        "pred_npy": out_path,
        "n_windows": len(compiled.window_decoders),
        "n_obs": int(pred.shape[1]),
    }))


if __name__ == "__main__":
    main()
