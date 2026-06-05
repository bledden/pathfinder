import numpy as np
from bb_code import BBCode
from canon_dem import extract
from qldpc.foundation.circuits import build_memory

def _block_ler_bposd(ex, n=4000, seed=3):
    from ldpc import BpOsdDecoder
    rng = np.random.default_rng(seed)
    pri = ex["priors"]; H = ex["H"]; Lo = ex["Lo"].toarray().astype(np.int64)
    err = (rng.random((n, ex["n_err"])) < pri[None, :]).astype(np.int8)
    synd = (err @ H.T.toarray().astype(np.int64)) % 2
    obs = (err @ Lo.T) % 2
    dec = BpOsdDecoder(H, error_channel=list(np.clip(pri, 1e-6, 1-1e-6)), max_iter=30,
                       bp_method="ms", osd_method="osd_cs", osd_order=10)
    f = 0
    for i in range(n):
        c = dec.decode(synd[i].astype(np.uint8)).astype(np.int64)
        if not np.array_equal((Lo @ c) % 2, obs[i]): f += 1
    return f / n

def test_both_bases_build_and_decode_sanely():
    bb = BBCode()
    for basis in ("Z", "X"):
        c = build_memory(bb, rounds=6, p=0.003, basis=basis, noise="si1000")
        ex = extract(c.detector_error_model(decompose_errors=False))
        assert ex["n_det"] == 6 * 36 + 36, f"{basis}: unexpected detector count {ex['n_det']}"
        assert _block_ler_bposd(ex) <= 0.02, f"{basis}: BP-OSD-10 sanity LER too high"

def test_si1000_rates_differ_from_uniform():
    bb = BBCode()
    cu = str(build_memory(bb, 2, 0.01, basis="Z", noise="uniform"))
    cs = str(build_memory(bb, 2, 0.01, basis="Z", noise="si1000"))
    assert "0.05" in cs and cs != cu  # measurement flip 5p = 0.05 appears under SI1000

def test_z_uniform_matches_existing():
    # (Z, uniform) must be the EXACT existing build_z_memory (regression safety)
    from bb_circuit import build_z_memory
    bb = BBCode()
    assert str(build_memory(bb, 4, 0.005, basis="Z", noise="uniform")) == str(build_z_memory(bb, 4, 0.005))
