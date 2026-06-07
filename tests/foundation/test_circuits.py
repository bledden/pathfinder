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
    # SI1000 now emits idle noise (idle p/10 per CNOT layer + idle-during-measure 2p), which
    # roughly doubles the per-mechanism error mass and raises the BP-OSD-10 LER from the no-idle
    # regime (~0.006 Z / ~0.005 X) up to the measured AFTER values below. Band is re-centered on
    # the new correct values (fixed seed=3, n=4000 -> deterministic): lower bound confirms idle is
    # present (LER materially above the no-idle ~0.006), upper bound confirms a modest increase,
    # NOT a blow-up that would signal mis-placed idle noise.
    bb = BBCode()
    expected = {"Z": 0.01375, "X": 0.01475}   # measured with idle, seed=3, n=4000
    for basis in ("Z", "X"):
        c = build_memory(bb, rounds=6, p=0.003, basis=basis, noise="si1000")
        ex = extract(c.detector_error_model(decompose_errors=False))
        assert ex["n_det"] == 6 * 36 + 36, f"{basis}: unexpected detector count {ex['n_det']}"
        ler = _block_ler_bposd(ex)
        assert 0.009 <= ler <= 0.020, (
            f"{basis}: BP-OSD-10 LER {ler:.5f} outside the with-idle band "
            f"[0.009, 0.020] (expected ~{expected[basis]}); idle noise raised it from the "
            f"no-idle ~0.006 regime — too low means idle dropped, too high means mis-placed")

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


# --- SI1000 idle-noise completion (Gidney): idle p/10 per CNOT layer + idle-during-measure 2p ----

_IDLE_BB = BBCode()  # default [[72,12,6]] (N=36); monomial_perms/X_ORDER/Z_ORDER bound to it


def test_si1000_emits_idle_noise():
    """Idle noise must change the DEM by ADDING error mass at the same (bb, rounds, p).

    FINDING: for this single-basis BB CSS memory, the idle DEPOLARIZE1 lands on data qubits whose
    error signatures already coincide with the existing two-qubit-gate error mechanisms, so Stim
    MERGES them — the DEM mechanism COUNT (num_errors) is unchanged. The real, faithful signal that
    idle noise is emitted is that the merged mechanisms' PRIORS rise: total DEM error mass strictly
    increases (si1000-with-idle > uniform). Pre-fix the two DEMs are identical -> this FAILS."""
    bb = _IDLE_BB
    for basis in ("Z", "X"):
        eu = extract(build_memory(bb, 2, 0.003, basis=basis, noise="uniform")
                     .detector_error_model(decompose_errors=False))
        es = extract(build_memory(bb, 2, 0.003, basis=basis, noise="si1000")
                     .detector_error_model(decompose_errors=False))
        mass_uni = float(eu["priors"].sum())
        mass_si = float(es["priors"].sum())
        # Idle noise must measurably raise the total error mass (here it ~doubles), proving the
        # DEPOLARIZE1 idle is wired into the DEM and not silently dropped.
        assert mass_si > mass_uni * 1.2, (
            f"{basis}: si1000 total error mass ({mass_si:.4f}) must exceed uniform "
            f"({mass_uni:.4f}) once idle noise is emitted")


def _depolarize1_rates(circ):
    """All DEPOLARIZE1 probabilities in a circuit, parsed NUMERICALLY (robust to float repr)."""
    rates = []
    for inst in circ:
        if inst.name == "DEPOLARIZE1":
            rates.extend(inst.gate_args_copy())
    return rates


def test_si1000_idle_rates_present():
    """The built si1000 circuit must contain DEPOLARIZE1 at the p/10 (per-CNOT-layer idle) and
    2p (idle-during-measure) rates; uniform must contain NO DEPOLARIZE1 idle.

    Rates are parsed numerically from the circuit (not matched against float repr) so this holds
    at any p -- including p=0.003 (the spec reference), where repr(p/10)=0.0003000...3 != Stim's
    printed 0.0003 would break a string match."""
    bb = _IDLE_BB
    for p in (0.01, 0.003):
        idle = p / 10.0
        idle_meas = 2.0 * p
        for basis in ("Z", "X"):
            rates = _depolarize1_rates(build_memory(bb, 2, p, basis=basis, noise="si1000"))
            assert rates, f"{basis} p={p}: si1000 has no DEPOLARIZE1 idle"
            assert any(np.isclose(r, idle) for r in rates), \
                f"{basis} p={p}: si1000 missing per-layer idle DEPOLARIZE1(~{idle})"
            assert any(np.isclose(r, idle_meas) for r in rates), \
                f"{basis} p={p}: si1000 missing idle-during-measure DEPOLARIZE1(~{idle_meas})"
            # every DEPOLARIZE1 rate must be one of the two idle rates (no stray rates)
            assert all(np.isclose(r, idle) or np.isclose(r, idle_meas) for r in rates), \
                f"{basis} p={p}: unexpected DEPOLARIZE1 rate in {sorted(set(rates))}"
            cu = build_memory(bb, 2, p, basis=basis, noise="uniform")
            assert not _depolarize1_rates(cu), f"{basis} p={p}: uniform must emit NO DEPOLARIZE1"


def test_uniform_unchanged_no_idle():
    """Regression: (X, uniform) via _build_general and (Z, uniform via build_z_memory) emit NO
    DEPOLARIZE1 idle noise, and (Z, uniform) stays bit-identical to build_z_memory."""
    from bb_circuit import build_z_memory
    bb = _IDLE_BB
    cx_uni = str(build_memory(bb, 3, 0.005, basis="X", noise="uniform"))
    assert "DEPOLARIZE1" not in cx_uni, "(X, uniform) must not emit idle DEPOLARIZE1"
    cz_uni = build_memory(bb, 3, 0.005, basis="Z", noise="uniform")
    assert "DEPOLARIZE1" not in str(cz_uni), "(Z, uniform) must not emit idle DEPOLARIZE1"
    assert str(cz_uni) == str(build_z_memory(bb, 3, 0.005)), "(Z, uniform) not bit-identical"


def test_dem_sanity_si1000():
    """Coda build-step-1 gate: dem_sanity must pass (a) decoder-DEM == Stim-DEM bit-identical,
    (b) noiseless num_errors == 0, (c) fixed-seed reproducibility — for a small si1000 memory,
    both bases."""
    from qldpc.foundation.circuits import dem_sanity
    bb = _IDLE_BB
    for basis in ("Z", "X"):
        # (b) noiseless determinism: zero error mechanisms
        c0 = build_memory(bb, 2, 0.0, basis=basis, noise="si1000")
        assert c0.detector_error_model(decompose_errors=False).num_errors == 0, \
            f"{basis}: noiseless si1000 DEM must have zero error mechanisms"
        # (a) + (c) via the gate on a noisy circuit
        c = build_memory(bb, 2, 0.003, basis=basis, noise="si1000")
        rep = dem_sanity(c, seed=7)
        assert rep["dem_faithful"], f"{basis}: decoder DEM != Stim DEM: {rep}"
        assert rep["reproducible"], f"{basis}: fixed-seed sampling not reproducible: {rep}"
        assert rep["stim_num_errors"] == rep["decoder_num_errors"], f"{basis}: mechanism count mismatch: {rep}"


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
