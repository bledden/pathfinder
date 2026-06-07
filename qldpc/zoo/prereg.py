"""Pre-registration artifact for the matched cross-decoder LER grid (gate §10).

This module BUILDS and WRITES a concrete pre-registration (``prereg.json``) that
is committed to git BEFORE any LER-grid number is observed. A real pre-registration
makes the grid (T6) impossible to stop-on-favorable / cherry-pick: the experimental
plan (decoders + pinned configs, p-grid, frozen shot budget, the gap-to-MLE
statistic, endpoints, multiplicity correction, eval-compute cap, and a reproducible
environment manifest) is frozen and git-hashed in advance.

Everything here is CONCRETE — pre-registration means no placeholders.

Binding constraints encoded here
--------------------------------
* R6 (classical bar): every BP-family adapter (BP, BP-OSD-0, BP-OSD-10, BP-LSD)
  pins normalized min-sum ``ms_scaling_factor = 0.625``. The artifact carries an
  explicit ``ms_scaling_note`` flagging that this DIFFERS from the legacy default
  used by ``canon_dem.decode_bposd`` (which sets ``bp_method='ms'`` but leaves
  ``ms_scaling_factor`` at ldpc's default of 1.0), so the grid's classical bar is
  cited correctly.
* R3 (shot budget): per-cell failure target = 300 (in the feasible 200-500 band).
  Shots/cell are sized from a tiny pilot (BP-OSD-10) via
  ``failures_to_shots(target, pilot_ler)`` and then FROZEN — no stop-on-favorable.
  An "upsize ONCE after pilot, document, no re-tune" rule and a pre-committed
  "Tesseract at low p may be cost-probed and down-scoped in T6" rule are recorded.
* R2 (gap-to-MLE statistic): the PRIMARY gap statistic is the per-shot PAIRED ratio
  (decoder failures vs Tesseract-MLE failures on the SAME sampled shots) summarized
  with bootstrap CIs — NOT an aggregate-ratio with a single Wilson CI. The
  definition is written into the artifact; the computation lives in T5.

The committed ``prereg.json`` (written by ``__main__``) IS the pre-registration.
"""
import importlib.metadata as _md
import math
import os
import subprocess
import sys

# conftest.py puts repo root + qldpc/ on sys.path; ensure both when run directly.
_QLDPC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_QLDPC)
for _p in (_ROOT, _QLDPC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# foundation.stats: prefer the qldpc-qualified import (unambiguous when the full
# suite is collected), fall back to flat (qldpc/ on path, run directly).
try:
    from qldpc.foundation import stats
except ImportError:  # pragma: no cover - flat fallback
    from foundation import stats

try:
    from qldpc.zoo import adapters
except ImportError:  # pragma: no cover - flat fallback
    import adapters


# --------------------------------------------------------------------------- #
# Frozen plan constants (pre-committed; do NOT tune after seeing the grid).     #
# --------------------------------------------------------------------------- #
# Primary code: the [[72,12,6]] bivariate-bicycle code (BBCode default).
P_GRID = [0.001, 0.002, 0.003, 0.005]   # sub-threshold -> near-threshold
P_REFERENCE = 0.003                      # spec reference point (p=0.003)
BASES = ["X", "Z"]
ROUNDS = 6                               # R = d = 6
FAILURE_TARGET = 300                     # per-cell failure target (200-500 band)
PILOT_SHOTS = 2000                       # tiny pilot per p to estimate LER
PILOT_SEED = 12345
# Conservative ceiling on shots/cell (also the per-cell eval-compute guard). Used
# both as the eval-compute accounting unit and to cap cells whose pilot LER is so
# low that a target of 300 failures is infeasible (see _size_shots).
MAX_SHOTS_PER_CELL = 5_000_000


def _version(pkg):
    try:
        return _md.version(pkg)
    except Exception:
        return None


def _git_head():
    """Full 40-hex current git HEAD; None if unavailable."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Decoder + config commitments                                                 #
# --------------------------------------------------------------------------- #
def _bp_family_config(name):
    """The pinned BP-family config (mirrors adapters._BP_* constants). R6: every
    BP-family bar pins ms_scaling_factor=0.625."""
    base = dict(
        bp_method=adapters._BP_METHOD,
        ms_scaling_factor=adapters._BP_MS_SCALING,   # R6: 0.625
        max_iter=adapters._BP_MAX_ITER,
        schedule=adapters._BP_SCHEDULE,
    )
    if name == "BP":
        base.update(decoder="BpDecoder")
    elif name == "BPOSD-0":
        base.update(decoder="BpOsdDecoder", osd_method="osd_0", osd_order=0)
    elif name == "BPOSD-10":
        base.update(decoder="BpOsdDecoder", osd_method="osd_cs", osd_order=10)
    elif name == "BPLSD":
        base.update(decoder="BpLsdDecoder", lsd_method="lsd_cs", lsd_order=10)
    return base


def _decoder_commitments():
    """One commitment record per zoo decoder: name, pinned config, tie-break, and
    availability. The BP-family records are derived from adapters' pinned constants
    WITHOUT building the decoders (no DEM needed); Relay-BP + sliding-window record
    their pinned configs and availability (sliding-window is env-dependent)."""
    recs = []

    # Core ldpc-family + Tesseract (always available in this env).
    bp_tie = {
        "BP": "min_sum_parallel_hard_decision",
        "BPOSD-0": "osd0_reliability_order",
        "BPOSD-10": "osd_cs_order10",
        "BPLSD": "lsd_cs_order10",
    }
    for name in ("BPOSD-0", "BPOSD-10", "BPLSD", "BP"):
        recs.append(dict(name=name, config=_bp_family_config(name),
                         tie_break=bp_tie[name], availability="available"))

    recs.append(dict(
        name="Tesseract",
        config=dict(decoder="Tesseract", det_beam=64),
        tie_break="astar_beam64_lowest_cost",
        availability="available",
    ))

    # Relay-BP (in-process; available iff importable).
    relay_avail = adapters.relay_bp_available()
    relay_cfg = dict(decoder="RelayBP", relay_bp_version=_version("relay-bp"),
                     **adapters._RELAY_BP_DEFAULTS)
    # tuple -> list for JSON.
    relay_cfg["gamma_dist_interval"] = list(relay_cfg["gamma_dist_interval"])
    recs.append(dict(
        name="RelayBP",
        config=relay_cfg,
        tie_break="relay_bp_nconv_disjoint_ensemble",
        availability="available" if relay_avail else "conditional (env-dependent)",
    ))

    # Sliding-window (qLDPCOrg/qLDPC streaming decoder; isolated subprocess).
    sw_avail = False
    try:
        sw_avail = adapters.sliding_window_available()
    except Exception:
        sw_avail = False
    sw_cfg = dict(
        decoder="SlidingWindow",
        qldpc_version="0.2.9",
        window=3,
        stride=1,
        commit=1,
        rounds=ROUNDS,
        inner=dict(adapters._SWD_DECODER_KWARGS),
    )
    recs.append(dict(
        name="SlidingWindow",
        config=sw_cfg,
        tie_break="sliding_window_bposd_cs_commit",
        availability="available" if sw_avail else "conditional (env-dependent)",
    ))
    return recs


def _ms_scaling_note():
    """R6 (BINDING): record that the BP/LSD/OSD bar uses ms_scaling_factor=0.625 and
    that this DIFFERS from canon_dem.decode_bposd's legacy default (1.0)."""
    return dict(
        ms_scaling_factor=adapters._BP_MS_SCALING,         # 0.625
        legacy_canon_dem_default=1.0,
        note=(
            "R6 (BINDING): the BP-OSD / BP-LSD / BP classical bar pins normalized "
            "min-sum ms_scaling_factor=0.625. This DIFFERS from the legacy default "
            "in canon_dem.decode_bposd, which sets bp_method='ms' but leaves "
            "ms_scaling_factor at ldpc's default (1.0). The grid MUST cite its "
            "classical bar as normalized min-sum ms=0.625, not the legacy ms=1.0 "
            "BP-OSD; the two are different decoders."
        ),
    )


# --------------------------------------------------------------------------- #
# R2 gap-to-MLE statistic definition                                           #
# --------------------------------------------------------------------------- #
def _gap_statistic_def():
    return dict(
        definition=(
            "Primary gap-to-MLE statistic = the per-shot PAIRED comparison of each "
            "decoder against the Tesseract MLE anchor on the SAME sampled shots "
            "(identical DEM, identical seed -> identical detectors/observables). "
            "For each shot we record (decoder_wrong, mle_wrong); the gap is the "
            "paired excess-failure rate / per-shot failure ratio of the decoder "
            "relative to Tesseract-MLE."
        ),
        summary=(
            "Summarized with BOOTSTRAP confidence intervals over the paired "
            "per-shot indicators (resampling shots), NOT an aggregate-ratio of two "
            "marginal failure counts with a single Wilson CI. The paired bootstrap "
            "respects the shot-level correlation between the decoder and the MLE "
            "anchor (both decode the same shots), which the aggregate Wilson form "
            "ignores."
        ),
        aggregate_ratio_single_wilson=False,   # explicitly NOT this form
        anchor="Tesseract",
        computation_location="T5 (per-shot decomposition); this prereg only "
                             "freezes the definition.",
    )


# --------------------------------------------------------------------------- #
# R3 shot-budget sizing (pilot-derived, frozen)                                #
# --------------------------------------------------------------------------- #
def _size_shots(pilot_ler, failure_target=FAILURE_TARGET, cap=MAX_SHOTS_PER_CELL):
    """shots/cell = ceil(failure_target / pilot_ler), capped at ``cap``.

    ``pilot_ler`` MUST be > 0 (a zero-failure pilot is uninformative for sizing;
    callers replace a 0 estimate with a documented conservative floor before
    calling here)."""
    shots = stats.failures_to_shots(failure_target, pilot_ler)
    return min(shots, cap)


def _pilot_ler(p, basis="Z", shots=PILOT_SHOTS, seed=PILOT_SEED):
    """Tiny BP-OSD-10 pilot to estimate the LER at (p, basis). Returns
    (ler, fails, shots). A real pilot — recorded as pilot-derived in the artifact."""
    from bb_code import BBCode

    try:
        from qldpc.foundation.circuits import build_memory
    except ImportError:  # pragma: no cover
        from foundation.circuits import build_memory
    from canon_dem import decode_bposd, extract

    c = build_memory(BBCode(), rounds=ROUNDS, p=p, basis=basis, noise="si1000")
    dem = c.detector_error_model(decompose_errors=False)
    ex = extract(dem)
    det, obs = c.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    (ler, _lo, _hi), fails, n = decode_bposd(ex, det, obs, max_iter=30, osd_order=10)
    return ler, fails, n


# Supplied fallback LER estimates (used when run_pilot=False, e.g. in tests, or as
# a documented conservative floor when a pilot sees zero failures). These come from
# a 2000-shot BP-OSD-10 pilot at R=6 SI1000 (seed=12345):
#   p=0.001 -> 0 failures  (uninformative; floored to the rule-of-three 95% upper
#              bound 3/2000 = 0.0015 so sizing is finite & not anti-conservative);
#   p=0.002 -> ~0.0015;  p=0.003 -> ~0.018;  p=0.005 -> ~0.092.
_SUPPLIED_PILOT_LER = {
    0.001: 0.0015,
    0.002: 0.0015,
    0.003: 0.018,
    0.005: 0.092,
}
# Floor for a zero-failure pilot: rule-of-three 95% upper CI for 0/N (3/N). Using
# this finite estimate keeps shots/cell finite; the cap + the documented low-p rule
# (cells may not reach 300 failures and instead bound the LER) cover the residual.
def _zero_failure_floor(shots):
    return 3.0 / shots


def _shots_per_cell(run_pilot, pilot_basis="Z"):
    """Frozen shots/cell per p. If run_pilot, run a real 2000-shot BP-OSD-10 pilot
    at each p (recorded pilot-derived); else use the supplied 2000-shot estimates
    (recorded supplied-estimate). Either way the result is FROZEN in the artifact."""
    out = {}
    for p in P_GRID:
        if run_pilot:
            ler, fails, n = _pilot_ler(p, basis=pilot_basis)
            source = "pilot-derived"
            note = None
            if ler <= 0.0:
                ler = _zero_failure_floor(n)
                note = (f"pilot saw 0/{n} failures (uninformative); LER floored to "
                        f"the rule-of-three 95% upper bound 3/{n}={ler:.6g} for "
                        f"finite sizing — this cell may not reach the 300-failure "
                        f"target and instead BOUNDS the LER (see tesseract_low_p_rule).")
            cell = dict(pilot_ler=ler, pilot_shots=n, pilot_fails=fails,
                        shots=_size_shots(ler), source=source)
            if note:
                cell["note"] = note
        else:
            ler = _SUPPLIED_PILOT_LER[p]
            cell = dict(pilot_ler=ler, pilot_shots=PILOT_SHOTS,
                        shots=_size_shots(ler), source="supplied-estimate")
        out[str(p)] = cell
    return out


def _shot_budget_rule():
    return dict(
        frozen=True,
        stop_on_favorable=False,
        upsize_rule=(
            "Upsize shots/cell at most ONCE after the pilot (e.g. if a cell's "
            "observed failure count lands outside the 200-500 band); the upsize "
            "must be DOCUMENTED with its reason and the new frozen shots/cell. No "
            "re-tuning, no per-cell adaptive growth, no stop-on-favorable."
        ),
        tesseract_low_p_rule=(
            "Tesseract (MLE anchor) at low p is expensive and the LER is tiny, so "
            "at low p Tesseract MAY be cost-probed and DOWN-SCOPED in T6 (e.g. "
            "fewer shots than the ldpc bars, or reported as an LER upper bound "
            "rather than a 300-failure point). This is a PRE-COMMITTED rule, not a "
            "post-hoc cut: any down-scope is recorded with its measured cost."
        ),
        max_shots_per_cell=MAX_SHOTS_PER_CELL,
    )


# --------------------------------------------------------------------------- #
# Per-cell DEM hashes (G1 ties the grid to these)                              #
# --------------------------------------------------------------------------- #
def _dem_hashes(record_hashes=True):
    """Per-(p, basis) DEM hash so the grid's DEMs are pre-committed (G1). Always
    records the construction params + a regeneration method; if ``record_hashes``
    also builds each circuit and records its dem_hash."""
    method = dict(
        builder="qldpc.foundation.circuits.build_memory",
        code="BBCode()  # [[72,12,6]]",
        rounds=ROUNDS,
        noise="si1000",
        decompose_errors=False,
        dem_hash="qldpc.zoo.harness.dem_hash = sha256(str(dem.flattened()).encode())",
        regenerate=(
            "for p in p_grid, basis in bases: build_memory(BBCode(), rounds=6, p=p, "
            "basis=basis, noise='si1000').detector_error_model(decompose_errors="
            "False); then dem_hash(dem)."
        ),
    )
    out = dict(method=method, cells=[])
    if not record_hashes:
        return out

    from bb_code import BBCode

    try:
        from qldpc.foundation.circuits import build_memory
    except ImportError:  # pragma: no cover
        from foundation.circuits import build_memory
    try:
        from qldpc.zoo.harness import dem_hash
    except ImportError:  # pragma: no cover
        from harness import dem_hash

    for p in P_GRID:
        for basis in BASES:
            c = build_memory(BBCode(), rounds=ROUNDS, p=p, basis=basis,
                             noise="si1000")
            dem = c.detector_error_model(decompose_errors=False)
            out["cells"].append(dict(
                p=p, basis=basis, rounds=ROUNDS, noise="si1000",
                n_det=dem.num_detectors, n_obs=dem.num_observables,
                dem_hash=dem_hash(dem),
            ))
    return out


# --------------------------------------------------------------------------- #
# Environment manifest                                                          #
# --------------------------------------------------------------------------- #
def _env_manifest():
    stim_ver = _version("stim")
    # stim MUST be 1.15.0 (BINDING): assert + record.
    assert stim_ver == "1.15.0", (
        f"prereg env: stim MUST be 1.15.0, found {stim_ver!r}. The pre-registered "
        f"DEM/decoder behavior is pinned to stim 1.15.0.")
    env = dict(
        python=sys.version.split()[0],
        numpy=_version("numpy"),
        stim=stim_ver,
        ldpc=_version("ldpc"),
        tesseract_decoder=_version("tesseract_decoder"),
        relay_bp=_version("relay-bp"),
    )
    # qLDPCOrg/qLDPC sliding-window decoder: version + the isolation note (it runs
    # from an isolated dir in a subprocess because its import name collides with
    # this repo's local ``qldpc`` namespace package).
    try:
        sw_avail = adapters.sliding_window_available()
    except Exception:
        sw_avail = False
    env["qldpc_sliding_window"] = dict(
        version="0.2.9",
        available=sw_avail,
        isolation=(
            "env-dependent: installed into an ISOLATED dir and driven from "
            "qldpc/zoo/_swd_worker.py in a SUBPROCESS whose sys.path excludes the "
            "repo (its PyPI import name 'qldpc' collides with this repo's local "
            "qldpc namespace package). Default ext dir: "
            "~/.venvs/braket/qldpc_ext_clean (override via QLDPC_EXT_DIR)."
        ),
    )
    return env


# --------------------------------------------------------------------------- #
# Build + write                                                                 #
# --------------------------------------------------------------------------- #
def build_prereg(run_pilot=True, record_dem_hashes=True, pilot_basis="Z"):
    """Build the concrete pre-registration dict.

    Args:
        run_pilot: if True (default, used for the committed artifact), run a real
            2000-shot BP-OSD-10 pilot at each p to size shots/cell (recorded
            pilot-derived). If False (used in tests), use the supplied 2000-shot
            estimates (recorded supplied-estimate). Either way shots/cell are FROZEN.
        record_dem_hashes: if True, build each (p, basis) circuit and record its
            DEM hash (G1 pre-commit). If False, record only the regeneration method.
        pilot_basis: basis used for the LER-sizing pilot (Z by default).

    Returns: a JSON-serializable dict with ALL pre-registered fields.
    """
    decoders = _decoder_commitments()
    spc = _shots_per_cell(run_pilot, pilot_basis=pilot_basis)

    # Eval-compute cap (DISTINCT from dev compute): the total pre-committed shot
    # budget = sum over cells (p x basis) of the frozen shots/cell.
    total = sum(spc[str(p)]["shots"] for p in P_GRID) * len(BASES)

    return dict(
        kind="pre-registration",
        version=1,
        purpose=(
            "Frozen experimental plan for the matched cross-decoder LER grid (T6), "
            "committed (git-hashed) BEFORE any grid number is observed so the grid "
            "cannot be stop-on-favorable / cherry-picked."
        ),

        # 1. decoders + pinned configs + tie-breaks (+ R6 note)
        decoders=decoders,
        ms_scaling_note=_ms_scaling_note(),

        # 2. code + p/d grid
        code=dict(name="BBCode", builder="qldpc.bb_code.BBCode()",
                  params=dict(n=72, k=12, d=6, l=6, m=6)),
        p_grid=P_GRID,
        p_reference=P_REFERENCE,
        bases=BASES,
        rounds=ROUNDS,
        noise="si1000",

        # 3. R3 shot budget
        failure_target=FAILURE_TARGET,
        shots_per_cell=spc,
        shot_budget_rule=_shot_budget_rule(),

        # 4. R2 gap-to-MLE statistic
        gap_statistic="per_shot_ratio_bootstrap",
        gap_statistic_def=_gap_statistic_def(),

        # 5. endpoints
        endpoints=dict(
            primary="per-round logical error rate lambda = 1 - (1-LER)^(1/rounds)",
            secondary=[
                "block-LER (per-shot logical failure rate)",
                "gap-to-MLE (per-shot paired ratio vs Tesseract, bootstrap CI)",
            ],
        ),

        # 6. multiplicity
        multiplicity=dict(
            primary="Holm-Bonferroni (FWER, alpha=0.05) on the primary endpoint",
            secondary="BH-FDR (q=0.05) on the secondary endpoints",
            applied_across="all (decoder, p, basis) cells",
            report_full_grid_including_losses=True,
            note=("Commit to reporting the FULL grid including losses (cells where "
                  "a decoder does NOT beat / match its bar): no selective reporting."),
        ),

        # 7. separate eval-compute cap (distinct from dev)
        eval_compute_cap=dict(
            total_shots=total,
            distinct_from_dev=True,
            note=("Total pre-committed EVAL shot budget = sum over (p x basis) cells "
                  "of the frozen shots/cell. This is SEPARATE from dev/iteration "
                  "compute; dev shots do not draw from this cap."),
            per_cell_max=MAX_SHOTS_PER_CELL,
        ),

        # 8. environment manifest (reproducibility)
        env=_env_manifest(),
        git_head=_git_head(),

        # 9. per-cell DEM hashes (G1 pre-commit)
        dem_hashes=_dem_hashes(record_hashes=record_dem_hashes),
    )


def write_prereg(path, prereg):
    """Write the pre-registration dict to ``path`` as indented JSON (round-trips
    via json.load). Reuses foundation.stats.write_prereg (which stamps written_unix
    and dumps indented JSON)."""
    return stats.write_prereg(path, **prereg)


_PREREG_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "prereg.json")


if __name__ == "__main__":
    # Build the REAL artifact (run the pilot + record per-cell DEM hashes) and
    # write the committed prereg.json. This file, once committed, IS the
    # pre-registration.
    pr = build_prereg(run_pilot=True, record_dem_hashes=True)
    write_prereg(_PREREG_JSON, pr)
    print(f"wrote {_PREREG_JSON}")
    print(f"  stim={pr['env']['stim']} git_head={pr['git_head']}")
    print(f"  p_grid={pr['p_grid']} failure_target={pr['failure_target']}")
    for p in pr["p_grid"]:
        cell = pr["shots_per_cell"][str(p)]
        print(f"  p={p}: pilot_ler={cell['pilot_ler']:.6g} "
              f"shots/cell={cell['shots']} ({cell['source']})")
    print(f"  eval_compute_cap.total_shots={pr['eval_compute_cap']['total_shots']}")
    print(f"  gap_statistic={pr['gap_statistic']}")
    print(f"  dem_hash cells={len(pr['dem_hashes']['cells'])}")
