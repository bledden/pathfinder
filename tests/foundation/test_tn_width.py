import numpy as np
from qldpc.foundation.tn_width import contraction_width

def test_tiny_repetition_code_is_low_width():
    # 3-bit repetition code: 2 checks (wt2), 1 logical (wt3). Treewidth ~1 -> width small.
    H = np.array([[1,1,0],[0,1,1]], dtype=np.int64)
    L = np.array([[1,1,1]], dtype=np.int64)
    w = contraction_width(H, L, max_time=10, max_repeats=50)
    assert w <= 6, f"rep-code width {w} unexpectedly large"

def test_bb_codecap_is_intractable():
    # [[72,12,6]] code-capacity must be intractable (pilot measured ~77).
    from bb_code import BBCode
    bb = BBCode()
    w = contraction_width(np.asarray(bb.HZ) % 2, np.asarray(bb.logicals_X()) % 2,
                          max_time=60, max_repeats=200)
    assert w > 32, f"expected intractable (>32), got {w}"
