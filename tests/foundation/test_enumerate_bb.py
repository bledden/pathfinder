from qldpc.foundation.enumerate_bb import candidate_bb_params, evaluate_candidate

def test_candidates_are_valid_css_small():
    cands = candidate_bb_params(max_lm=4, max_n=60)
    assert len(cands) >= 3
    for c in cands:
        assert c["n"] == 2 * c["l"] * c["m"]

def test_evaluate_returns_width_and_k():
    # l=2,m=2 (n=8) BB codes are all degenerate (k=0); use the smallest CSS-valid k>=1
    # candidate (l=2,m=3, n=12, k=4) so the width+k contract is actually exercised.
    c = {"l": 2, "m": 3, "A_terms": (("x",0),("y",1),("y",2)), "B_terms": (("x",1),("y",1),("y",2))}
    r = evaluate_candidate(c, width_time=20, width_repeats=40)
    assert r["css_valid"] is True
    assert isinstance(r["width"], float) and r["width"] > 0
    assert r["k"] >= 1
