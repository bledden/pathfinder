"""TDD for the LER-grid driver (qldpc/zoo/run_grid.py).

``run_grid(prereg_path, out_path, ...)`` is the matched cross-decoder LER-grid
driver. For each (p, basis) cell of the (optionally subsetted) pre-registered
grid (``qldpc/zoo/prereg.json``) it:

  1. builds the BB memory circuit + the shared DEM (decompose_errors=False),
  2. G1 pre-commit check: asserts dem_hash(dem) == the prereg's recorded
     dem_hash for that cell (fail-fast on DEM drift — ties the grid to the
     pre-registration),
  3. builds the full available decoder zoo from that shared DEM,
  4. runs the matched protocol (run_matched, keep_per_shot=True),
  5. per decoder records name/config/tie_break/fails/ler/ler_ci/lambda/decode_s,
     gap-to-MLE (the locked per-shot paired bootstrap vs the Tesseract anchor)
     for every non-anchor decoder, and Gate-A (BP-OSD-10 vs Tesseract),
  6. after all cells: a multiplicity-annotated pairwise comparison list
     (Holm + BH) and the LER-accuracy frontier (per decoder: lambda vs p and
     gap-to-MLE vs p).

The returned dict (and the emitted JSON) is strictly JSON-serializable: the
numpy per-shot fail masks are stripped, only scalars / CIs survive.

The public signature is ``run_grid(prereg_path, out_path, p_subset=None,
shots_override=None, bases=None, include_sliding_window=True, seed=0,
dry_scale=None)`` — ``prereg_path`` is a PATH to the locked prereg JSON,
``out_path`` (may be None) is where the JSON-serializable result is written.

These tests run a REDUCED grid (tiny shots, p_subset=[0.005], one basis, core
decoders only — sliding-window OFF) to stay fast; the full pre-registered
budget grid is gated and NOT exercised here.
"""
import copy
import json
import os

import pytest

from qldpc.zoo.adapters import APPROVED_TIE_BREAKS
from qldpc.zoo.run_grid import load_prereg, run_grid

_PREREG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "qldpc", "zoo", "prereg.json",
)


@pytest.fixture(scope="module")
def prereg():
    return load_prereg(_PREREG_PATH)


@pytest.fixture(scope="module")
def reduced_grid():
    """A small, fast reduced grid: p=0.005 (cheapest), Z basis only, ~300 shots,
    core decoders only (sliding-window OFF) to keep the test quick. Driven from
    the prereg PATH (the public signature)."""
    return run_grid(
        _PREREG_PATH,
        out_path=None,
        p_subset=[0.005],
        shots_override=300,
        bases=["Z"],
        include_sliding_window=False,
        seed=0,
    )


# --- per-cell records -------------------------------------------------------
def test_grid_has_cells_with_decoder_records(reduced_grid):
    g = reduced_grid
    assert "cells" in g and len(g["cells"]) == 1  # one (p, basis) cell
    cell = g["cells"][0]
    assert cell["p"] == 0.005
    assert cell["basis"] == "Z"
    assert cell["shots"] == 300
    # G1 provenance recorded on the cell
    assert isinstance(cell["dem_hash"], str) and len(cell["dem_hash"]) == 64
    assert cell["g1_match"] is True

    decs = cell["decoders"]
    names = {d["name"] for d in decs}
    # core zoo present (sliding-window not requested)
    assert {"BPOSD-0", "BPOSD-10", "BPLSD", "BP", "Tesseract"} <= names

    for d in decs:
        assert isinstance(d["config"], dict) and d["config"]
        assert d["tie_break"] in APPROVED_TIE_BREAKS
        assert 0.0 <= d["ler"] <= 1.0
        lo, hi = d["ler_ci"]
        assert 0.0 <= lo <= d["ler"] <= hi <= 1.0
        assert 0 <= d["fails"] <= cell["shots"]
        assert 0.0 <= d["lambda_per_round"] <= 1.0
        assert d["decode_s"] >= 0.0
        # no numpy fail-mask leaked into the record
        assert "fail_mask" not in d


# --- gap-to-MLE present for non-anchor; Tesseract is the anchor -------------
def test_gap_to_mle_for_non_tesseract_and_anchor(reduced_grid):
    cell = reduced_grid["cells"][0]
    by = {d["name"]: d for d in cell["decoders"]}

    # Tesseract is the anchor: ratio 1.0 vs itself (and explicitly flagged).
    tess = by["Tesseract"]
    assert tess.get("is_anchor") is True
    g = tess["gap_to_mle"]
    assert g["ratio"] == pytest.approx(1.0)

    # every non-anchor decoder has a populated gap-to-MLE block
    for name, d in by.items():
        if name == "Tesseract":
            continue
        assert d.get("is_anchor") in (False, None)
        gm = d["gap_to_mle"]
        assert set(gm) >= {"ratio", "lo", "hi"}
        assert gm["ratio"] >= 0.0
        assert gm["lo"] <= gm["hi"]
        # gap-to-MLE values are plain python floats (JSON-serializable)
        for k in ("ratio", "lo", "hi"):
            assert isinstance(gm[k], float)


# --- Gate-A populated -------------------------------------------------------
def test_gate_a_populated(reduced_grid):
    cell = reduced_grid["cells"][0]
    ga = cell["gate_a"]
    assert ga is not None
    assert ga["n"] == cell["shots"]
    # 2x2 disagreement counts must partition all shots
    total = (ga["both_correct"] + ga["both_wrong"]
             + ga["only_a_right"] + ga["only_b_right"])
    assert total == cell["shots"]
    assert 0.0 <= ga["oracle_ler_bound"] <= 1.0


# --- multiplicity-annotated comparison list ---------------------------------
def test_multiplicity_comparison_list(reduced_grid):
    g = reduced_grid
    comps = g["comparisons"]
    assert isinstance(comps, list) and len(comps) >= 1
    for c in comps:
        # identity passthrough
        assert "decoder" in c and "p" in c and "basis" in c
        assert {"fails_a", "n_a", "fails_b", "n_b"} <= set(c)
        # multiplicity annotation present
        assert "holm_sig" in c and isinstance(c["holm_sig"], bool)
        assert "bh_sig" in c and isinstance(c["bh_sig"], bool)
        assert "pvalue" in c and 0.0 <= c["pvalue"] <= 1.0
        # physical p preserved alongside the raw p-value
        assert c["p"] == 0.005


# --- frontier ---------------------------------------------------------------
def test_frontier_present(reduced_grid):
    g = reduced_grid
    fr = g["frontier"]
    assert isinstance(fr, dict) and fr
    # per decoder: lambda vs p and gap-to-MLE vs p
    for name, series in fr.items():
        assert "lambda_vs_p" in series
        assert "gap_to_mle_vs_p" in series
        for pt in series["lambda_vs_p"]:
            assert "p" in pt and "basis" in pt and "lambda_per_round" in pt
        # the anchor has no gap-to-MLE-vs-p points (or all 1.0)
        if name == "Tesseract":
            assert series["gap_to_mle_vs_p"] == [] or all(
                pt["ratio"] == 1.0 for pt in series["gap_to_mle_vs_p"])


# --- JSON-serializable (no numpy) -------------------------------------------
def test_grid_json_serializable(reduced_grid):
    s = json.dumps(reduced_grid)
    assert isinstance(s, str) and len(s) > 0
    back = json.loads(s)
    assert back["cells"][0]["p"] == 0.005


# --- provenance: prereg git-hash + env + per-cell dem_hash ------------------
def test_grid_provenance(reduced_grid, prereg):
    g = reduced_grid
    assert g["prereg_git_head"] == prereg["git_head"]
    assert "env" in g and isinstance(g["env"], dict)
    assert g["cells"][0]["dem_hash"]
    # sliding-window cost-probe rule must be recorded even when SW is off
    assert "sliding_window" in g
    assert "down_scope_rule" in g["sliding_window"]


# --- dry_scale scales the prereg shots/cell ---------------------------------
def test_dry_scale_scales_prereg_shots(prereg):
    """dry_scale=f runs each cell at ~f * its prereg shots/cell (the pilot knob
    when shots_override is not given). p=0.005 prereg shots=3261 -> *0.05 ~= 163."""
    g = run_grid(
        _PREREG_PATH,
        out_path=None,
        p_subset=[0.005],
        bases=["Z"],
        dry_scale=0.05,
        include_sliding_window=False,
        seed=0,
    )
    prereg_shots = int(prereg["shots_per_cell"]["0.005"]["shots"])
    expected = max(1, int(round(prereg_shots * 0.05)))
    assert g["cells"][0]["shots"] == expected
    assert g["dry_scale"] == 0.05


# --- G1 drift fail-fast -----------------------------------------------------
def test_g1_drift_failfast(prereg, tmp_path):
    """Tamper the prereg's recorded dem_hash for the p=0.005 Z cell, write it to
    a temp prereg PATH -> the G1 pre-commit check must raise (the circuit's DEM
    no longer matches the pre-registered one)."""
    tampered = copy.deepcopy(prereg)
    for cell in tampered["dem_hashes"]["cells"]:
        if cell["p"] == 0.005 and cell["basis"] == "Z":
            cell["dem_hash"] = "0" * 64  # bogus
    bad = tmp_path / "prereg_tampered.json"
    bad.write_text(json.dumps(tampered))
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        run_grid(
            str(bad),
            out_path=None,
            p_subset=[0.005],
            shots_override=100,
            bases=["Z"],
            include_sliding_window=False,
            seed=0,
        )


# --- out_path writes JSON-serializable file ---------------------------------
def test_out_path_writes_file(tmp_path):
    out = tmp_path / "zoo_grid_test.json"
    g = run_grid(
        _PREREG_PATH,
        out_path=str(out),
        p_subset=[0.005],
        shots_override=100,
        bases=["Z"],
        include_sliding_window=False,
        seed=0,
    )
    assert out.exists()
    loaded = json.load(open(out))
    assert loaded["cells"][0]["p"] == 0.005
    # the in-memory return value equals what was written
    assert loaded["cells"][0]["dem_hash"] == g["cells"][0]["dem_hash"]
