"""TDD for the BB circulant-block layout (qldpc/kernel/bb_layout.py).

The BB check matrices are block-circulant by construction: A, B are sums of shift
monomials x^a y^b = Sl^a (x) Sm^b over the (l, m) torus, and H_Z = [B^T | A^T],
H_X = [A | B] are 1x2 grids of those N=l*m circulant blocks. The GPU "circulant-
unroll" kernel (T7b) will iterate over each block's compile-time-known shift set
(the analogue of the 7 lattice directions in bench/triton_directional.py), so this
layout must:

  * recover each block's circulant shift set as plain integers (a, b) on Z_l x Z_m,
  * reconstruct H LOSSLESSLY from that block/shift decomposition (== bb.HZ % 2),
  * expose a plain Tanner-graph CSR/CSC adjacency (check->bit, bit->check) derived
    from the SAME H, so the BP baseline and the future kernel share one layout.
"""
import numpy as np

from bb_code import BBCode
from qldpc.kernel.bb_layout import bb_circulant_blocks


def _mono(bb, a, b):
    """Shift monomial Sl^a (x) Sm^b as an N x N GF2 permutation matrix."""
    Sl = np.linalg.matrix_power(bb.Sl, a) % 2
    Sm = np.linalg.matrix_power(bb.Sm, b) % 2
    return np.kron(Sl, Sm) % 2


def test_returns_block_grid_for_HZ():
    bb = BBCode()
    layout = bb_circulant_blocks(bb)
    assert "HZ" in layout
    hz = layout["HZ"]
    # H_Z = [B^T | A^T]: a 1 (check-block-row) x 2 (bit-block-col) grid of NxN blocks.
    assert hz["block_rows"] == 1
    assert hz["block_cols"] == 2
    assert hz["N"] == bb.N
    assert hz["l"] == bb.l and hz["m"] == bb.m
    blocks = hz["blocks"]
    assert len(blocks) == 2  # one circulant block per bit-block-column


def test_block_shifts_are_compile_time_integers():
    """Each block's shifts must be plain (int, int) tuples on Z_l x Z_m -- the
    compile-time offsets the GPU kernel unrolls over (no runtime data)."""
    bb = BBCode()
    layout = bb_circulant_blocks(bb)
    for code in ("HZ", "HX"):
        for blk in layout[code]["blocks"]:
            for (a, b) in blk["shifts"]:
                assert isinstance(a, int) and isinstance(b, int)
                assert 0 <= a < bb.l and 0 <= b < bb.m


def test_HZ_block_shifts_match_known_monomials():
    """H_Z = [B^T | A^T]. B^T has shifts = negated B shifts, A^T = negated A shifts.
    Default A = x^3 + y^1 + y^2 -> {(3,0),(0,1),(0,2)}; B = y^3 + x^1 + x^2 ->
    {(0,3),(1,0),(2,0)}. Transpose negates each shift mod (l, m)."""
    bb = BBCode()
    layout = bb_circulant_blocks(bb)
    blocks = layout["HZ"]["blocks"]
    s0 = set(blocks[0]["shifts"])  # B^T block
    s1 = set(blocks[1]["shifts"])  # A^T block
    l, m = bb.l, bb.m
    bT = {((-a) % l, (-b) % m) for (a, b) in [(0, 3), (1, 0), (2, 0)]}
    aT = {((-a) % l, (-b) % m) for (a, b) in [(3, 0), (0, 1), (0, 2)]}
    assert s0 == bT, f"B^T shifts {s0} != {bT}"
    assert s1 == aT, f"A^T shifts {s1} != {aT}"


def test_HZ_reconstruction_is_lossless():
    """Reconstruct H_Z from the circulant-block representation; must equal bb.HZ % 2
    exactly. The block/shift decomposition is lossless."""
    bb = BBCode()
    layout = bb_circulant_blocks(bb)
    hz = layout["HZ"]
    N = hz["N"]
    H_rec = np.zeros((N, hz["block_cols"] * N), dtype=np.uint8)
    for blk in hz["blocks"]:
        bc = blk["block_col"]
        M = np.zeros((N, N), dtype=np.uint8)
        for (a, b) in blk["shifts"]:
            M = (M + _mono(bb, a, b)) % 2
        H_rec[:, bc * N:(bc + 1) * N] = M
    assert np.array_equal(H_rec, np.asarray(bb.HZ, dtype=np.uint8) % 2)
    # And the layout exposes the same reconstruction directly.
    assert np.array_equal(np.asarray(hz["H_dense"], dtype=np.uint8) % 2,
                          np.asarray(bb.HZ, dtype=np.uint8) % 2)


def test_HX_reconstruction_is_lossless():
    bb = BBCode()
    layout = bb_circulant_blocks(bb)
    hx = layout["HX"]
    assert np.array_equal(np.asarray(hx["H_dense"], dtype=np.uint8) % 2,
                          np.asarray(bb.HX, dtype=np.uint8) % 2)


def test_csr_csc_adjacency_matches_H_nonzero_structure():
    """The Tanner-graph CSR (check->bit) and CSC (bit->check) adjacency must match
    H's nonzero structure exactly. This is the SHARED gather/scatter layout the BP
    baseline and the GPU kernel both consume."""
    bb = BBCode()
    layout = bb_circulant_blocks(bb)
    H = np.asarray(bb.HZ, dtype=np.uint8) % 2
    tg = layout["HZ"]["tanner"]

    n_checks, n_bits = H.shape
    assert tg["n_checks"] == n_checks
    assert tg["n_bits"] == n_bits

    # CSR check->bit: row pointers + bit indices.
    chk_ptr = np.asarray(tg["check_to_bit_ptr"])
    chk_idx = np.asarray(tg["check_to_bit_idx"])
    assert chk_ptr.shape == (n_checks + 1,)
    assert chk_ptr[-1] == int(H.sum())
    for c in range(n_checks):
        got = set(int(b) for b in chk_idx[chk_ptr[c]:chk_ptr[c + 1]])
        want = set(int(b) for b in np.nonzero(H[c])[0])
        assert got == want, f"check {c}: {got} != {want}"

    # CSC bit->check: col pointers + check indices.
    bit_ptr = np.asarray(tg["bit_to_check_ptr"])
    bit_idx = np.asarray(tg["bit_to_check_idx"])
    assert bit_ptr.shape == (n_bits + 1,)
    assert bit_ptr[-1] == int(H.sum())
    for v in range(n_bits):
        got = set(int(c) for c in bit_idx[bit_ptr[v]:bit_ptr[v + 1]])
        want = set(int(c) for c in np.nonzero(H[:, v])[0])
        assert got == want, f"bit {v}: {got} != {want}"


def test_edge_lists_are_consistent_between_csr_and_csc():
    """CSR and CSC must enumerate the SAME set of (check, bit) edges (same total),
    so a single edge-indexed message array can be shared by gather (CSR) and
    scatter (CSC) -- the GPU-portable invariant."""
    bb = BBCode()
    layout = bb_circulant_blocks(bb)
    tg = layout["HZ"]["tanner"]
    chk_ptr = np.asarray(tg["check_to_bit_ptr"])
    chk_idx = np.asarray(tg["check_to_bit_idx"])
    bit_ptr = np.asarray(tg["bit_to_check_ptr"])
    bit_idx = np.asarray(tg["bit_to_check_idx"])

    csr_edges = set()
    for c in range(tg["n_checks"]):
        for b in chk_idx[chk_ptr[c]:chk_ptr[c + 1]]:
            csr_edges.add((c, int(b)))
    csc_edges = set()
    for v in range(tg["n_bits"]):
        for c in bit_idx[bit_ptr[v]:bit_ptr[v + 1]]:
            csc_edges.add((int(c), v))
    assert csr_edges == csc_edges
    assert len(csr_edges) == int((np.asarray(bb.HZ) % 2).sum())
