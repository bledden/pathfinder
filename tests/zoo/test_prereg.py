"""TDD for the pre-registration artifact (qldpc/zoo/prereg.py).

A pre-registration commits (git-hashed) the experimental plan BEFORE any LER-grid
number is seen, so the grid (T6) cannot be stop-on-favorable / cherry-picked. The
build must be CONCRETE (no placeholders): pinned decoder configs, a frozen p-grid,
a frozen per-cell failure target + shots/cell, the gap-to-MLE statistic definition,
endpoints, multiplicity, a separate eval-compute cap, and an environment manifest.

Binding constraints checked here:
  * R6: the BP-OSD/LSD/BP classical bar uses normalized min-sum ms_scaling_factor
    = 0.625, with an explicit note that this differs from canon_dem.decode_bposd's
    legacy default (1.0) so the grid's classical bar is cited correctly.
  * R3: per-cell failure target = 300 (200-500 band); shots/cell are FROZEN (pilot-
    derived, upsize-once rule).
  * R2: the primary gap statistic is the per-shot PAIRED ratio with bootstrap CIs
    (NOT an aggregate-ratio with a single Wilson CI).
  * stim MUST be 1.15.0 (recorded + asserted in the env manifest).
"""
import json

import pytest

from qldpc.zoo.prereg import build_prereg, write_prereg


@pytest.fixture(scope="module")
def prereg():
    # No pilot here (keep the test fast + deterministic): pass explicit pilot LERs
    # so build_prereg can size shots/cell without sampling. The shots/cell are still
    # FROZEN in the returned artifact; only the source is "supplied" vs "pilot".
    return build_prereg(run_pilot=False)


# --- required keys ----------------------------------------------------------
def test_all_required_keys(prereg):
    required = {
        "decoders", "p_grid", "bases", "rounds", "failure_target",
        "shots_per_cell", "gap_statistic", "endpoints", "multiplicity",
        "env", "git_head", "eval_compute_cap",
    }
    assert required <= set(prereg), f"missing keys: {required - set(prereg)}"


# --- decoders + pinned configs + tie-breaks ---------------------------------
def test_decoders_have_name_config_tiebreak(prereg):
    decs = prereg["decoders"]
    assert isinstance(decs, list) and decs
    for d in decs:
        assert isinstance(d["name"], str) and d["name"]
        assert isinstance(d["config"], dict) and d["config"]
        # tie_break is present for every available (non-conditional) decoder.
        if d.get("availability", "available") == "available":
            assert isinstance(d["tie_break"], str) and d["tie_break"]


def test_core_decoders_present(prereg):
    names = {d["name"] for d in prereg["decoders"]}
    # The core zoo (always available in this env) must all be pre-committed.
    assert {"BPOSD-0", "BPOSD-10", "BPLSD", "BP", "Tesseract"} <= names


def test_R6_ms_scaling_note(prereg):
    """R6 (BINDING): the BP-family classical bar pins ms_scaling_factor=0.625 and
    the prereg explicitly flags this differs from canon_dem.decode_bposd's legacy
    default (1.0)."""
    # Every BP-family adapter's pinned config records ms_scaling_factor = 0.625.
    bp_family = [d for d in prereg["decoders"]
                 if d["name"] in {"BPOSD-0", "BPOSD-10", "BPLSD", "BP"}]
    assert bp_family
    for d in bp_family:
        assert d["config"].get("ms_scaling_factor") == 0.625, (
            f"{d['name']} must pin ms_scaling_factor=0.625")

    # An explicit, machine-readable R6 note that flags the legacy 1.0 difference.
    note = prereg.get("ms_scaling_note")
    assert isinstance(note, dict)
    assert note["ms_scaling_factor"] == 0.625
    assert note["legacy_canon_dem_default"] == 1.0
    assert "decode_bposd" in note["note"]
    text = note["note"].lower()
    assert "0.625" in note["note"]
    assert "1.0" in note["note"] or "1.0" in str(note["legacy_canon_dem_default"])
    assert "legacy" in text or "differ" in text


def test_relay_and_sliding_window_recorded(prereg):
    """relay-bp + sliding-window must be recorded (sliding-window may be
    conditional/env-dependent)."""
    names = {d["name"] for d in prereg["decoders"]}
    assert "RelayBP" in names
    assert "SlidingWindow" in names
    sw = next(d for d in prereg["decoders"] if d["name"] == "SlidingWindow")
    # availability is one of the documented states.
    assert sw["availability"] in {"available", "conditional (env-dependent)"}


# --- code + p/d grid --------------------------------------------------------
def test_p_grid_and_grid_params(prereg):
    assert prereg["p_grid"] == [0.001, 0.002, 0.003, 0.005]
    assert prereg["bases"] == ["X", "Z"]
    assert prereg["rounds"] == 6
    code = prereg["code"]
    assert code["name"] == "BBCode"
    assert code["params"]["n"] == 72
    assert code["params"]["k"] == 12
    assert code["params"]["d"] == 6
    # p=0.003 is the spec reference point.
    assert prereg["p_reference"] == 0.003
    assert prereg["p_reference"] in prereg["p_grid"]


# --- R3 shot budget ---------------------------------------------------------
def test_failure_target_in_band(prereg):
    assert prereg["failure_target"] == 300
    assert 200 <= prereg["failure_target"] <= 500


def test_shots_per_cell_frozen_and_sized(prereg):
    spc = prereg["shots_per_cell"]
    assert isinstance(spc, dict)
    # one frozen shots/cell per p in the grid.
    assert {str(p) for p in prereg["p_grid"]} == set(spc.keys())
    for p in prereg["p_grid"]:
        cell = spc[str(p)]
        assert cell["shots"] >= prereg["failure_target"]  # >= one shot per failure
        assert cell["pilot_ler"] > 0.0
        # shots ~ ceil(failure_target / pilot_ler)
        import math
        exp = int(math.ceil(prereg["failure_target"] / cell["pilot_ler"]))
        assert cell["shots"] == exp
        assert cell["source"] in {"pilot-derived", "supplied-estimate"}
    # No stop-on-favorable: explicit frozen + upsize-once rules.
    rule = prereg["shot_budget_rule"]
    assert rule["frozen"] is True
    assert rule["stop_on_favorable"] is False
    assert "upsize" in rule["upsize_rule"].lower()
    assert "once" in rule["upsize_rule"].lower()
    # Tesseract low-p down-scope rule pre-committed (not a post-hoc cut).
    assert "tesseract" in rule["tesseract_low_p_rule"].lower()


# --- R2 gap-to-MLE statistic ------------------------------------------------
def test_gap_statistic_is_per_shot_bootstrap(prereg):
    assert prereg["gap_statistic"] == "per_shot_ratio_bootstrap"
    gd = prereg["gap_statistic_def"]
    text = (gd["definition"] + " " + gd["summary"]).lower()
    assert "per-shot" in text or "per shot" in text
    assert "paired" in text
    assert "bootstrap" in text
    assert "tesseract" in text
    # Must NOT be the aggregate-ratio-with-Wilson form.
    assert gd["aggregate_ratio_single_wilson"] is False
    assert "aggregate" in text  # explicitly contrasts against the aggregate form


# --- endpoints + multiplicity ----------------------------------------------
def test_endpoints(prereg):
    ep = prereg["endpoints"]
    assert "lambda" in ep["primary"].lower() and "round" in ep["primary"].lower()
    sec = " ".join(ep["secondary"]).lower()
    assert "block" in sec and "ler" in sec
    assert "gap" in sec and "mle" in sec


def test_multiplicity(prereg):
    mu = prereg["multiplicity"]
    assert "holm" in mu["primary"].lower()
    assert "bh" in mu["secondary"].lower() or "fdr" in mu["secondary"].lower()
    assert mu["report_full_grid_including_losses"] is True
    # multiplicity applied across (decoder, p, basis) cells.
    span = mu["applied_across"].lower()
    assert "decoder" in span and "basis" in span


# --- separate eval-compute cap ----------------------------------------------
def test_eval_compute_cap(prereg):
    cap = prereg["eval_compute_cap"]
    assert cap["total_shots"] > 0
    assert cap["distinct_from_dev"] is True


# --- environment manifest ---------------------------------------------------
def test_env_records_stim_1_15_0(prereg):
    env = prereg["env"]
    assert env["stim"] == "1.15.0", "stim MUST be 1.15.0 (recorded + asserted)"
    assert env["python"]
    assert env["numpy"]
    assert env["ldpc"]
    assert env["tesseract_decoder"]
    # relay-bp present in this env; recorded.
    assert "relay_bp" in env
    # sliding-window qLDPC version + isolation note recorded.
    assert "qldpc_sliding_window" in env
    swenv = env["qldpc_sliding_window"]
    assert "isolat" in (swenv.get("isolation", "") + str(swenv)).lower()


def test_git_head_recorded(prereg):
    gh = prereg["git_head"]
    # full 40-hex sha (or None if unavailable, but in-repo it should be present).
    assert gh is None or (len(gh) == 40 and all(c in "0123456789abcdef" for c in gh))


# --- DEM hashes (per-cell, optional but ideal) ------------------------------
def test_dem_hashes_recorded(prereg):
    dh = prereg["dem_hashes"]
    # Either concrete per-cell hashes, OR a regeneration method (both acceptable).
    assert "method" in dh
    if dh.get("cells"):
        for cell in dh["cells"]:
            assert cell["p"] in prereg["p_grid"]
            assert cell["basis"] in prereg["bases"]
            h = cell["dem_hash"]
            assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


# --- write / reload round-trip ----------------------------------------------
def test_write_reload_roundtrip(prereg, tmp_path):
    path = tmp_path / "prereg.json"
    write_prereg(str(path), prereg)
    reloaded = json.load(open(path))
    # JSON round-trips object keys to str; compare the load-bearing fields.
    assert reloaded["p_grid"] == prereg["p_grid"]
    assert reloaded["failure_target"] == prereg["failure_target"]
    assert reloaded["gap_statistic"] == prereg["gap_statistic"]
    assert reloaded["env"]["stim"] == "1.15.0"
    assert reloaded["ms_scaling_note"]["ms_scaling_factor"] == 0.625
    assert {d["name"] for d in reloaded["decoders"]} == {
        d["name"] for d in prereg["decoders"]}
