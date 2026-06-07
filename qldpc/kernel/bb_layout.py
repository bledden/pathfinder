"""BB -> circulant-block representation + shared Tanner-graph CSR/CSC layout.

Stage-(a) of the kernel lane. The BB check matrices are *block-circulant by
construction*: ``A`` and ``B`` are GF2 sums of shift monomials
``x^a y^b = Sl^a (x) Sm^b`` over the (l, m) torus, and the CSS check matrices are
1x2 grids of those ``N = l*m`` circulant blocks::

    H_X = [A | B]        H_Z = [B^T | A^T]

This module expresses ``H_Z`` (and ``H_X``) as that grid of circulant blocks and
extracts, per block, its set of compile-time-known shifts ``(a, b)`` on
``Z_l x Z_m``. Those shifts are exactly what the later Triton "circulant-unroll"
kernel (T7b) will iterate over -- the algebraic analogue of the 7 fixed lattice
directions unrolled in ``bench/triton_directional.py``. Reconstructing ``H`` from
the block/shift decomposition is *lossless* (asserted in tests against
``bb.HZ % 2``), so the representation throws nothing away.

It also exposes a plain Tanner-graph edge layout derived from the SAME ``H``:
CSR (check->bit) and CSC (bit->check) adjacency. The portable BP baseline
(``bp_baseline.py``) and the future GPU kernel consume this one shared layout, so
their gather/scatter index sets are identical by construction.

Pure numpy. No triton/torch/cuda. The block-circulant *structure* (fixed shift
offsets + edge-list gather/scatter) is the GPU-portable part; this CPU module is
the reference the kernel matches.
"""
import numpy as np


def _recover_circulant_shifts(M, l, m):
    """Recover the shift set of an N x N 2D-circulant block over Z_l x Z_m.

    A block that is a GF2 sum of monomials ``Sl^a (x) Sm^b`` maps basis position
    ``(i, j)`` (flat index ``i*m + j``, row-major) to ``(i+a mod l, j+b mod m)``.
    Row 0 (position ``(0,0)``) therefore has a 1 in exactly the columns
    ``a*m + b`` for each monomial in the block, so its nonzeros read off the shift
    set directly. Returns a sorted list of plain ``(int, int)`` tuples.
    """
    M = np.asarray(M, dtype=np.uint8) % 2
    cols = np.nonzero(M[0])[0]
    shifts = sorted((int(c) // m, int(c) % m) for c in cols)
    return shifts


def _mono(Sl, Sm, a, b):
    """Shift monomial Sl^a (x) Sm^b as an N x N GF2 permutation matrix."""
    Pl = np.linalg.matrix_power(np.asarray(Sl, dtype=np.int64), a) % 2
    Pm = np.linalg.matrix_power(np.asarray(Sm, dtype=np.int64), b) % 2
    return np.kron(Pl, Pm).astype(np.uint8) % 2


def _circulant_grid(H_dense, N, l, m, Sl, Sm):
    """Decompose a (N x (block_cols*N)) block-circulant matrix into its grid of
    circulant blocks + per-block shift sets, and verify lossless reconstruction.
    """
    H_dense = np.asarray(H_dense, dtype=np.uint8) % 2
    n_rows, n_cols = H_dense.shape
    assert n_rows == N, f"expected {N} check rows, got {n_rows}"
    assert n_cols % N == 0, f"{n_cols} not a multiple of N={N}"
    block_cols = n_cols // N
    block_rows = 1  # BB CSS check matrices are a single block-row of N circulants

    blocks = []
    H_rec = np.zeros_like(H_dense)
    for bc in range(block_cols):
        sub = H_dense[:, bc * N:(bc + 1) * N]
        shifts = _recover_circulant_shifts(sub, l, m)
        # Lossless guard: rebuild this block from its recovered shifts.
        rebuilt = np.zeros((N, N), dtype=np.uint8)
        for (a, b) in shifts:
            rebuilt = (rebuilt + _mono(Sl, Sm, a, b)) % 2
        assert np.array_equal(rebuilt, sub), (
            f"block_col {bc}: shift decomposition not lossless")
        H_rec[:, bc * N:(bc + 1) * N] = rebuilt
        blocks.append(dict(block_row=0, block_col=bc, shifts=shifts))

    assert np.array_equal(H_rec, H_dense), "block grid reconstruction != H"
    return dict(block_rows=block_rows, block_cols=block_cols, N=N, l=l, m=m,
                blocks=blocks, H_dense=H_dense)


def tanner_adjacency(H):
    """CSR (check->bit) + CSC (bit->check) adjacency for a GF2 check matrix H.

    This is the SHARED gather/scatter layout: a single set of (check, bit) edges,
    addressable both check-major (CSR, the gather of a check's incident bit
    messages) and bit-major (CSC, the scatter back to bits). Both index the SAME
    edge set, so an edge-indexed message array ports directly to a GPU
    gather/scatter kernel.

    Returns a dict with:
      n_checks, n_bits,
      check_to_bit_ptr (n_checks+1), check_to_bit_idx (nnz)  -- CSR,
      bit_to_check_ptr (n_bits+1),   bit_to_check_idx (nnz)  -- CSC.
    """
    H = np.asarray(H, dtype=np.uint8) % 2
    n_checks, n_bits = H.shape

    # CSR check->bit (row-major nonzeros, columns ascending within each row).
    rows, cols = np.nonzero(H)
    order = np.lexsort((cols, rows))           # sort by (row, then col)
    rows, cols = rows[order], cols[order]
    chk_ptr = np.zeros(n_checks + 1, dtype=np.int64)
    np.add.at(chk_ptr, rows + 1, 1)
    chk_ptr = np.cumsum(chk_ptr)
    chk_idx = cols.astype(np.int64)

    # CSC bit->check (col-major nonzeros, rows ascending within each col).
    order2 = np.lexsort((rows, cols))          # sort by (col, then row)
    rows2, cols2 = rows[order2], cols[order2]
    bit_ptr = np.zeros(n_bits + 1, dtype=np.int64)
    np.add.at(bit_ptr, cols2 + 1, 1)
    bit_ptr = np.cumsum(bit_ptr)
    bit_idx = rows2.astype(np.int64)

    return dict(
        n_checks=int(n_checks), n_bits=int(n_bits),
        check_to_bit_ptr=chk_ptr, check_to_bit_idx=chk_idx,
        bit_to_check_ptr=bit_ptr, bit_to_check_idx=bit_idx,
    )


def bb_circulant_blocks(bb, which=("HZ", "HX")):
    """Express the BB check matrices as grids of circulant blocks + shifts.

    Args:
        bb: a ``BBCode`` instance (exposes ``.A, .B, .HZ, .HX, .Sl, .Sm, l, m, N``).
        which: which check matrices to decompose ("HZ" and/or "HX").

    Returns a dict keyed by check-matrix name. Each entry holds:
        block_rows, block_cols, N, l, m,
        blocks: list of {block_row, block_col, shifts:[(a,b)...]},
        H_dense: the (losslessly) reconstructed dense GF2 matrix,
        tanner: CSR/CSC Tanner-graph adjacency (see ``tanner_adjacency``).

    The ``shifts`` are plain ``(int, int)`` offsets on ``Z_l x Z_m`` -- the
    compile-time directions the GPU circulant-unroll kernel iterates over. The
    ``tanner`` block is the SAME-H adjacency the BP baseline consumes.
    """
    l, m, N = int(bb.l), int(bb.m), int(bb.N)
    Sl, Sm = bb.Sl, bb.Sm
    out = {}
    for code in which:
        H_dense = np.asarray(getattr(bb, code), dtype=np.uint8) % 2
        entry = _circulant_grid(H_dense, N, l, m, Sl, Sm)
        entry["tanner"] = tanner_adjacency(H_dense)
        out[code] = entry
    return out
