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


# --- Adversarial X-basis correctness (these FAIL on the pre-fix transpose bug) -----------------

def test_x_basis_observable_commutes_with_measured_checks():
    """Algebraic invariant: the X-memory measures the TRUE H_X=[A|B], whose stabilizers commute
    with the Z-logical observable. (The pre-fix code measured the transpose [A^T|B^T], which does
    NOT commute with logicals_Z, so a memory built on it protected a relabeled non-CSS check set.)"""
    bb = BBCode()
    HX = np.asarray(bb.HX) % 2
    LZ = np.asarray(bb.logicals_Z()) % 2
    assert np.all((HX @ LZ.T) % 2 == 0)
    # the WRONG (transposed) check set demonstrably anti-commutes with the observable:
    A, B = bb.A % 2, bb.B % 2
    HX_transpose = np.concatenate([A.T, B.T], axis=1) % 2
    assert not np.all((HX_transpose @ LZ.T) % 2 == 0), \
        "sanity: the transpose check set should NOT commute with logicals_Z (else test is blind)"


def _measured_check_matrix(bb, basis):
    """Operationally recover the per-round measured stabilizer support from the COMPILED circuit by
    propagating a deterministic single-qubit error on each data qubit through the noiseless circuit
    and reading which round-0 detectors fire. For an X-memory (detects Z errors) inject Z; for a
    Z-memory inject X. Returns M[check, data_qubit] over the 2N data qubits."""
    import stim
    from qldpc.foundation.circuits import build_memory
    N, n_data = bb.N, 2 * bb.N
    base = build_memory(bb, rounds=1, p=0.0, basis=basis, noise="uniform")
    err_op = "Z_ERROR" if basis == "X" else "X_ERROR"
    instrs = list(base)
    M = np.zeros((N, n_data), dtype=np.int8)
    for q in range(n_data):
        circ = stim.Circuit()
        circ.append(instrs[0])               # initial data reset
        circ.append(err_op, [q], 1.0)        # deterministic flip on data qubit q
        for ins in instrs[1:]:
            circ.append(ins)
        det = circ.compile_detector_sampler(seed=0).sample(1)[0]
        for cc in range(N):
            if det[cc]:                      # round-0 detectors are the first N
                M[cc, q] ^= 1
    return M


def test_x_memory_measures_true_HX_not_transpose():
    """The compiled X-memory's per-round measured check matrix must equal the TRUE H_X=[A|B].
    This is the direct operational discriminator: it FAILS on the pre-fix code (which measured the
    transpose [A^T|B^T]) and PASSES on the fix. (Preferred over the 'undetected-error => stabilizer'
    probe, which is vacuous here: a distance-6 code essentially never yields a fully-silent shot
    under random sampling, so it cannot distinguish the two check sets.)"""
    bb = BBCode()
    M = _measured_check_matrix(bb, "X")
    HX = np.asarray(bb.HX) % 2
    assert np.array_equal(M, HX), "X-memory measures the wrong check set (transpose bug)"
    # also confirm it is NOT the transpose (would catch a partial/relabeled fix)
    A, B = bb.A % 2, bb.B % 2
    assert not np.array_equal(M, np.concatenate([A.T, B.T], axis=1) % 2)


def test_z_memory_measures_true_HZ_unchanged():
    """Regression: the Z-memory still measures H_Z=[B^T|A^T] exactly (the Z path is untouched)."""
    bb = BBCode()
    M = _measured_check_matrix(bb, "Z")
    HZ = np.asarray(bb.HZ) % 2
    assert np.array_equal(M, HZ)


def test_build_memory_commute_guard_fires_on_wrong_observable():
    """The build-time commute guard must reject a circuit whose observable does not commute with
    the measured stabilizers (the structural defense against this whole bug class). Feeding the
    X-builder the WRONG observable (logicals_X, which anti-commutes with H_X) must raise."""
    bb = BBCode()
    orig = bb.logicals_Z
    bb.logicals_Z = bb.logicals_X            # wrong observable for an X-memory
    try:
        import pytest
        with pytest.raises(AssertionError):
            build_memory(bb, rounds=2, p=0.0, basis="X", noise="uniform")
    finally:
        bb.logicals_Z = orig
