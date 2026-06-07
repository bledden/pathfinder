"""Matched cross-decoder LER-grid driver (T6).

``run_grid`` walks the (optionally subsetted) pre-registered grid of (p, basis)
cells (``qldpc/zoo/prereg.json``) and, for each cell, runs the matched protocol
(ONE shared DEM, ONE sampled shot set, every decoder decoding the byte-identical
DEM) and assembles the defensible per-cell verdict layer:

  1. ``circ = build_memory(BBCode(), rounds, p, basis, "si1000")``; the shared DEM
     is ``circ.detector_error_model(decompose_errors=False)``.
  2. **G1 pre-commit check (fail-fast).** ``dem_hash(dem)`` MUST equal the
     prereg's recorded ``dem_hash`` for this (p, basis) cell. A mismatch means
     the circuit's DEM drifted from the pre-registered one (a different
     code/rounds/p/basis/noise, or a builder change) — the grid is no longer the
     pre-registered experiment, so we raise immediately. This ties every grid
     number to the committed pre-registration.
  3. ``decoders = build_decoders(dem, include_external=..., rounds=rounds, ...)``
     — the full available zoo (BP, BP-OSD-0/10, BP-LSD, Tesseract, Relay-BP, and
     the sliding-window streaming decoder when available).
  4. ``run_matched(circ, decoders, shots=<cell shots>, ..., keep_per_shot=True)``
     -> per-decoder fail masks aligned to the shared shots.
  5. Per decoder: name, config, tie_break, fails, ler, ler_ci (Wilson),
     lambda_per_round, decode_s. **gap-to-MLE**: for each non-Tesseract decoder,
     ``gap_to_mle_bootstrap(decoder_mask, tesseract_mask)`` (R2 paired bootstrap,
     ratio + CI). **Gate-A**: ``gate_a(bposd10_mask, tesseract_mask)``.
  6. After all cells: the FULL pairwise comparison list across (decoder, p,
     basis) annotated by ``multiplicity_grid`` (Holm + BH); the LER-accuracy
     **frontier** (per decoder: lambda vs p, and gap-to-MLE vs p).

The result (and the optional ``out_path`` JSON) is strictly JSON-serializable:
the numpy per-shot fail masks are stripped, only scalars / CIs survive. It also
carries provenance: the prereg git-hash, the per-cell dem_hash, and the env.

Honest reporting (prereg ``report_full_grid_including_losses = true``): ALL cells
and ALL decoders are recorded, including losses. A low-p cell that falls short of
the prereg failure target is marked ``ler_is_bound = True`` (its LER is an upper
bound, not a 300-failure point estimate; see the prereg's tesseract_low_p_rule).

Sliding-window cost handling (BINDING)
--------------------------------------
The sliding-window decoder runs per-window OSD in an isolated subprocess. Its
subprocess STARTUP cost (~1s, fixed) dominates a tiny probe batch, but its
AMORTIZED per-shot cost on a real batch is ~10x cheaper (one subprocess decodes
the whole batch). At the prereg's full shots sliding-window is therefore feasible
(~tens of seconds for a 16k-shot cell, a few minutes for a 200k-shot cell), so it
runs at the FULL prereg shots/cell like every other decoder.

To measure the realistic cost (not the startup-dominated tiny probe), ``run_grid``
cost-probes on a SMALL-BATCH-AMORTIZED basis: it times the streaming decoder on a
modest probe batch on the FIRST cell, then fits an affine cost model
(``total ~= startup + per_shot * n``) using the known fixed startup so the
projection reflects the marginal per-shot rate, NOT the startup-inflated
per-probe-shot number. The projection only ever triggers a CAP as a generous
backstop: if a cell's projected wall-time exceeds ``sliding_window_time_cap_s``
(default 20 min — NOT expected to fire at the real amortized rate), that cell's
sliding-window is down-scoped to the **failure-target shot count** for that p
(enough shots for ~``failure_target`` failures at the cell's pilot LER, floored to
a sane minimum), recorded as ``sliding_window_shots`` + a note — never to the
1-5-shot garbage the old startup-dominated probe produced. The probe, the cap,
the rule, and the actual measured cost are recorded under ``sliding_window`` and
per cell. No other decoder is ever down-scoped.

NOT the full grid: this module is the driver. The full pre-registered budget grid
is run only after the gated sign-off; a small pilot validation (``p_subset=[0.005]``,
both bases, ``shots_override=2000``) proves the pipeline end-to-end.
"""
import importlib.metadata as _md
import json
import os
import subprocess
import sys
import time

import numpy as np

# conftest.py puts repo root + qldpc/ on sys.path; ensure both when imported
# directly (e.g. a gated full-run script).
_QLDPC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_QLDPC)
for _p in (_ROOT, _QLDPC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from qldpc.foundation.circuits import build_memory
except ImportError:  # pragma: no cover - flat fallback
    from foundation.circuits import build_memory

try:
    from qldpc.zoo import adapters
    from qldpc.zoo.adapters import build_decoders
    from qldpc.zoo.analysis import gap_to_mle_bootstrap, gate_a, multiplicity_grid
    from qldpc.zoo.harness import dem_hash, run_matched
except ImportError:  # pragma: no cover - flat fallback (qldpc/ on path)
    import adapters
    from adapters import build_decoders
    from analysis import gap_to_mle_bootstrap, gate_a, multiplicity_grid
    from harness import dem_hash, run_matched


# Tesseract MLE anchor: gap-to-MLE is measured against it; every other decoder's
# gap is its paired per-shot failure ratio vs this anchor.
_ANCHOR = "Tesseract"
# Gate-A pits the strong classical bar (BP-OSD-10) against the MLE anchor.
_GATE_A_DECODER = "BPOSD-10"
# Fixed sampler seed for reproducibility (the grid is fixed-seed end to end).
_DEFAULT_SEED = 0

# --- sliding-window cost-probe defaults (BINDING; see module docstring) ------
# Probe the streaming decoder on this many shots on the first cell, fit the affine
# cost model (startup + per_shot * n) so the projection reflects the AMORTIZED
# marginal per-shot rate (the subprocess startup is fixed, not per-shot). At that
# rate sliding-window runs FULL prereg shots/cell; the time cap is a generous
# backstop that is not expected to fire. If it ever does, the cell's SW is
# down-scoped to the failure-target shot count for that p (a meaningful set), NOT
# to a few shots.
_SW_PROBE_SHOTS = 400           # amortized probe batch (past the startup hump)
_SW_TIME_CAP_S = 1200.0         # 20-min/cell backstop (not expected to trigger)
_SW_STARTUP_S = 1.0             # fixed subprocess-startup cost subtracted in the
                                # affine fit so the per-shot rate is amortized
_SW_FLOOR_SHOTS = 1000          # never down-scope SW below this many shots


def load_prereg(path):
    """Load the pre-registration JSON (the locked grid plan)."""
    with open(path) as fh:
        return json.load(fh)


def _version(pkg):
    try:
        return _md.version(pkg)
    except Exception:
        return None


def _git_head():
    """Full current git HEAD (40-hex); None if unavailable."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _env_manifest():
    """Reproducibility manifest (mirrors prereg env keys)."""
    return dict(
        python=sys.version.split()[0],
        numpy=_version("numpy"),
        stim=_version("stim"),
        ldpc=_version("ldpc"),
        tesseract_decoder=_version("tesseract_decoder"),
        relay_bp=_version("relay-bp"),
    )


def _prereg_dem_hash(prereg, p, basis):
    """The prereg's recorded dem_hash for (p, basis).

    The prereg nests per-cell hashes under ``dem_hashes.cells`` as a list of
    ``{p, basis, ..., dem_hash}`` records; also tolerate a possible nested
    ``dem_hashes[basis][p]`` form. Returns None if the cell is not pre-registered
    (the G1 check then has nothing to compare against -> treated as a hard fail by
    the caller).
    """
    dh = prereg.get("dem_hashes", {})
    cells = dh.get("cells")
    if isinstance(cells, list):
        for c in cells:
            if c.get("p") == p and c.get("basis") == basis:
                return c.get("dem_hash")
        return None
    # tolerate dem_hashes[basis][p] (str-key p) or dem_hashes[p][basis]
    by_basis = dh.get(basis)
    if isinstance(by_basis, dict):
        return by_basis.get(str(p)) or by_basis.get(p)
    by_p = dh.get(str(p)) or dh.get(p)
    if isinstance(by_p, dict):
        return by_p.get(basis)
    return None


def _cell_shots(prereg, p, shots_override, dry_scale=None):
    """Shots/cell for p, in precedence order:

      1. ``shots_override`` (if set): use this exact count for EVERY cell
         (the explicit pilot / test path).
      2. ``dry_scale`` (if set): use ``round(dry_scale * frozen_shots)`` —
         a fractional down-scale of the prereg's frozen shots/cell for a pilot
         that preserves the per-p shot RATIO (cheap cells stay cheaper). Floored
         to >= 1 so a cell never collapses to zero shots.
      3. otherwise the frozen prereg ``shots_per_cell[str(p)]['shots']`` (the
         full pre-registered budget).
    """
    if shots_override is not None:
        return int(shots_override)
    spc = prereg["shots_per_cell"][str(p)]
    frozen = int(spc["shots"])
    if dry_scale is not None:
        return max(1, int(round(frozen * float(dry_scale))))
    return frozen


def _strip_mask(rec):
    """Return a JSON-safe copy of a run_matched decoder record (drop fail_mask)."""
    return {k: v for k, v in rec.items() if k != "fail_mask"}


def _sliding_window_cost_probe(circ, dem, rounds, seed, probe_shots,
                               startup_s=_SW_STARTUP_S):
    """Time the sliding-window streaming decoder on a probe batch and return its
    AMORTIZED marginal per-shot rate.

    The sliding-window subprocess pays a fixed STARTUP cost (~1s) once per
    ``decode_batch`` call regardless of batch size; the marginal per-shot decode
    cost is ~10x cheaper. Projecting a full cell from a startup-dominated tiny
    probe massively over-estimates the cost (the old defect). Here we fit the
    affine model ``total = startup + per_shot * n`` by subtracting the (known,
    fixed) ``startup_s`` from the measured probe total, so the projection uses the
    realistic MARGINAL per-shot rate.

    Returns ``(amortized_seconds_per_shot, probe_shots, probe_total_s)`` or None
    if the sliding-window decoder is unavailable, or ``{"error": ...}`` on failure.
    """
    try:
        if not adapters.sliding_window_available():
            return None
        sw = adapters.make_sliding_window(dem, rounds=rounds)
        dets, _obs = circ.compile_detector_sampler(seed=seed).sample(
            probe_shots, separate_observables=True)
        dets = np.asarray(dets, dtype=bool)
        t0 = time.perf_counter()
        sw.decode_batch(dets)
        dt = time.perf_counter() - t0
        if not probe_shots:
            return (float("inf"), probe_shots, dt)
        # Affine fit: marginal per-shot = (total - fixed startup) / probe_shots,
        # clamped to >= 0 (a tiny/fast probe can dip below the nominal startup).
        marginal = max(0.0, dt - startup_s) / probe_shots
        return (marginal, probe_shots, dt)
    except Exception as exc:  # never let the probe stall / break the grid
        return {"error": repr(exc)}


def _sw_failure_target_shots(prereg, p, failure_target, floor_shots):
    """Shots needed for ~``failure_target`` failures at p (the SANE backstop floor
    when the time cap fires): ``failure_target / pilot_ler`` for that p, floored to
    ``floor_shots`` and never above the cell's full budget (set by the caller)."""
    spc = prereg.get("shots_per_cell", {}).get(str(p), {})
    pilot_ler = spc.get("pilot_ler")
    if pilot_ler and pilot_ler > 0:
        target = int(round(failure_target / float(pilot_ler)))
    else:
        target = floor_shots
    return max(int(floor_shots), target)


def run_grid(prereg_path, out_path, p_subset=None, shots_override=None,
             bases=None, include_sliding_window=True, seed=_DEFAULT_SEED,
             dry_scale=None, include_external=True, n_boot=10000,
             sliding_window_probe_shots=_SW_PROBE_SHOTS,
             sliding_window_time_cap_s=_SW_TIME_CAP_S,
             sliding_window_floor_shots=_SW_FLOOR_SHOTS):
    """Run the matched cross-decoder LER grid over the pre-registered cells.

    Args:
        prereg_path: PATH to the locked pre-registration JSON (the frozen grid
            plan; see ``load_prereg``). A pre-loaded prereg dict is also accepted
            defensively, but the canonical contract is a path.
        out_path: PATH to write the JSON-serializable result to (or None to skip
            writing and only return the dict).
        p_subset: optional list of p values to restrict to (default: prereg
            ``p_grid``). Each must be a pre-registered p.
        shots_override: if set, use this shots/cell for EVERY cell (the explicit
            pilot / test path).
        bases: optional list of bases to restrict to (default: prereg ``bases``).
        include_sliding_window: gate the sliding-window adapter independently of
            the other external adapters (its subprocess cost is handled below).
        seed: detector-sampler seed (fixed-seed reproducibility).
        dry_scale: optional fractional down-scale of the prereg's frozen
            shots/cell (``round(dry_scale * frozen)``, floored to >= 1) — the
            pilot knob that preserves the per-p shot RATIO. Ignored when
            ``shots_override`` is set (override wins).
        include_external: build the available external adapters (Relay-BP, and
            sliding-window when ``include_sliding_window``). Sliding-window also
            requires ``rounds`` (always supplied here).
        n_boot: bootstrap resamples for the gap-to-MLE paired bootstrap.
        sliding_window_probe_shots / _time_cap_s / _floor_shots: the
            sliding-window cost-probe + backstop parameters (see module docstring).
            Sliding-window runs at the FULL prereg shots/cell; the time cap is a
            generous backstop and, if it ever fires, the cell's SW is down-scoped
            to the failure-target shot count (floored to ``_floor_shots``).

    Returns:
        a JSON-serializable dict:
          * ``cells``: one record per (p, basis) cell — p, basis, shots,
            dem_hash, g1_match, per-decoder records (incl gap_to_mle + is_anchor),
            gate_a, ler_is_bound, sliding_window_shots/note (when down-scoped).
          * ``comparisons``: the FULL multiplicity-annotated pairwise comparison
            list across (decoder, p, basis) (Holm + BH).
          * ``frontier``: per decoder, lambda-vs-p and gap-to-MLE-vs-p series.
          * ``sliding_window``: the cost-probe result + the down-scope rule.
          * provenance: ``prereg_git_head``, ``grid_git_head``, ``env``,
            ``failure_target``, ``seed``.
    """
    # Accept a PATH (canonical contract) or a pre-loaded prereg dict (defensive).
    prereg = prereg_path if isinstance(prereg_path, dict) else load_prereg(prereg_path)

    p_grid = list(p_subset) if p_subset is not None else list(prereg["p_grid"])
    basis_list = list(bases) if bases is not None else list(prereg["bases"])
    rounds = int(prereg["rounds"])
    noise = prereg.get("noise", "si1000")
    failure_target = int(prereg.get("failure_target", 300))

    # Lazy code import (kept local: run_grid is imported even where bb_code's
    # heavy deps are not needed at import time).
    try:
        from bb_code import BBCode
    except ImportError:  # pragma: no cover
        from qldpc.bb_code import BBCode

    # Sliding-window cost-probe state (probed once on the first SW-eligible cell,
    # then applied to every cell). Recorded in the result for transparency.
    want_sw = bool(include_external and include_sliding_window)
    sw_block = {
        "requested": want_sw,
        "available": (adapters.sliding_window_available() if want_sw else False),
        "probe_shots": sliding_window_probe_shots,
        "time_cap_s": sliding_window_time_cap_s,
        "floor_shots": sliding_window_floor_shots,
        "startup_s": _SW_STARTUP_S,
        "down_scope_rule": (
            "Sliding-window runs at the FULL prereg shots/cell, like every other "
            "decoder. To measure realistic cost, cost-probe the streaming decoder "
            f"on {sliding_window_probe_shots} shots on the first cell and fit the "
            f"affine model total = startup(~{_SW_STARTUP_S:.0f}s) + per_shot * n, "
            "so the projection uses the AMORTIZED marginal per-shot rate (NOT the "
            "startup-dominated per-probe-shot number). The time cap "
            f"({sliding_window_time_cap_s:.0f}s/cell) is a generous backstop that "
            "is NOT expected to fire at the amortized rate. If a cell's projected "
            "wall-time ever exceeds the cap, that cell's sliding-window is "
            "down-scoped to the failure-target shot count for its p (enough shots "
            f"for ~{failure_target} failures at the pilot LER, floored to "
            f"{sliding_window_floor_shots}), recorded as sliding_window_shots + a "
            "note. No other decoder is ever down-scoped."
        ),
        "probe": None,            # filled by the first probe
        "sec_per_shot": None,     # AMORTIZED marginal per-shot rate
    }

    cells = []
    for p in p_grid:
        for basis in basis_list:
            # 1. build the circuit + the single shared DEM.
            circ = build_memory(BBCode(), rounds=rounds, p=p, basis=basis,
                                noise=noise)
            dem = circ.detector_error_model(decompose_errors=False)
            cell_hash = dem_hash(dem)

            # 2. G1 pre-commit check: this cell's DEM MUST match the prereg's
            #    recorded dem_hash (fail-fast on drift).
            expected = _prereg_dem_hash(prereg, p, basis)
            if expected is None:
                raise ValueError(
                    f"G1 pre-commit FAIL: prereg has no recorded dem_hash for "
                    f"cell (p={p}, basis={basis!r}); cannot tie this cell to the "
                    f"pre-registration. Add it to prereg.dem_hashes or restrict "
                    f"the grid to pre-registered cells."
                )
            if cell_hash != expected:
                raise ValueError(
                    f"G1 pre-commit FAIL: cell (p={p}, basis={basis!r}) DEM "
                    f"hashes to {cell_hash}, but the pre-registration recorded "
                    f"{expected}. The grid's DEM drifted from the pre-registered "
                    f"one (different code/rounds/p/basis/noise or a builder "
                    f"change) — the grid is no longer the pre-registered "
                    f"experiment. Aborting."
                )

            shots = _cell_shots(prereg, p, shots_override, dry_scale=dry_scale)

            # --- sliding-window cost-probe + per-cell down-scope --------------
            sw_shots = None
            sw_note = None
            cell_wants_sw = want_sw and sw_block["available"]
            if cell_wants_sw:
                if sw_block["probe"] is None:
                    probe = _sliding_window_cost_probe(
                        circ, dem, rounds, seed, sliding_window_probe_shots)
                    sw_block["probe"] = probe
                    if isinstance(probe, tuple):
                        sw_block["sec_per_shot"] = probe[0]
                sps = sw_block.get("sec_per_shot")
                # Backstop floor: shots for ~failure_target failures at this p
                # (a MEANINGFUL set), never below the floor, never above the cell
                # budget. Only used if the (never-expected) time cap fires.
                floor = min(
                    shots,
                    _sw_failure_target_shots(prereg, p, failure_target,
                                             sliding_window_floor_shots),
                )
                if sps is None:
                    # probe failed -> cannot project; down-scope to the
                    # failure-target floor defensively (never stall the grid),
                    # NOT to a few shots.
                    sw_shots = floor
                    sw_note = ("sliding-window cost-probe failed; down-scoped to "
                               f"the failure-target {sw_shots}-shot floor "
                               "defensively (see sliding_window.probe).")
                else:
                    # Amortized affine projection (fixed startup + marginal/shot).
                    projected = _SW_STARTUP_S + sps * shots
                    if projected > sliding_window_time_cap_s and shots > floor:
                        sw_shots = floor
                        sw_note = (
                            f"sliding-window projected ~{projected:.1f}s for "
                            f"{shots} shots (> {sliding_window_time_cap_s:.0f}s "
                            f"backstop at {sps:.5f}s/shot amortized); DOWN-SCOPED "
                            f"to the failure-target {sw_shots}-shot floor for this "
                            f"cell (gap-to-MLE / LER on the floor set, marked "
                            f"conditional).")
                    else:
                        sw_shots = shots  # full budget fits the cap (expected)

            # 3. build the full decoder zoo from the shared DEM. Sliding-window
            #    is built into the matched run ONLY when it is NOT down-scoped
            #    (i.e. its full-shot cost fits the cap); a down-scoped SW is run
            #    separately on the capped shot count so it cannot stall the grid.
            sw_inline = cell_wants_sw and (sw_shots == shots)
            decoders = build_decoders(
                dem,
                include_external=include_external,
                rounds=(rounds if (include_external and include_sliding_window
                                   and sw_inline) else None),
            )
            # If external requested but SW is down-scoped, build_decoders(rounds=None)
            # drops SW (and keeps Relay-BP). The down-scoped SW is decoded below.

            # 4. matched run (one shared DEM, one shot set, keep per-shot masks).
            man = run_matched(circ, decoders, shots=shots, rounds=rounds,
                              seed=seed, label=f"grid:p={p} basis={basis}",
                              keep_per_shot=True)

            by_mask = {r["name"]: r["fail_mask"] for r in man["decoders"]}
            anchor_mask = by_mask.get(_ANCHOR)

            # 5. assemble per-decoder cell records (strip masks; add gap-to-MLE).
            dec_records = []
            for r in man["decoders"]:
                rec = _strip_mask(r)
                name = rec["name"]
                if name == _ANCHOR:
                    rec["is_anchor"] = True
                    # the anchor's gap to itself is 1.0 by definition.
                    rec["gap_to_mle"] = {"ratio": 1.0, "lo": 1.0, "hi": 1.0}
                else:
                    rec["is_anchor"] = False
                    if anchor_mask is not None:
                        rec["gap_to_mle"] = gap_to_mle_bootstrap(
                            r["fail_mask"], anchor_mask, n_boot=n_boot, seed=seed)
                    else:
                        rec["gap_to_mle"] = None  # no anchor in this cell
                dec_records.append(rec)

            # Gate-A: BP-OSD-10 vs Tesseract (over the same shots).
            ga = None
            if _GATE_A_DECODER in by_mask and anchor_mask is not None:
                ga = gate_a(by_mask[_GATE_A_DECODER], anchor_mask)

            # honest low-p handling: a cell short of the failure target reports
            # its LER as a BOUND (per prereg tesseract_low_p_rule), not a point.
            anchor_fails = next(
                (d["fails"] for d in dec_records if d["name"] == _ANCHOR), None)
            max_fails = max((d["fails"] for d in dec_records), default=0)
            ler_is_bound = max_fails < failure_target

            cell_rec = {
                "p": p,
                "basis": basis,
                "rounds": rounds,
                "noise": noise,
                "shots": shots,
                "seed": seed,
                "dem_hash": cell_hash,
                "g1_match": True,
                "n_det": man["n_det"],
                "n_obs": man["n_obs"],
                "failure_target": failure_target,
                "ler_is_bound": bool(ler_is_bound),
                "decoders": dec_records,
                "gate_a": ga,
            }

            # --- down-scoped sliding-window: decode on the capped shot set ----
            if cell_wants_sw and not sw_inline:
                cell_rec["sliding_window_shots"] = sw_shots
                cell_rec["sliding_window_note"] = sw_note
                try:
                    sw = adapters.make_sliding_window(dem, rounds=rounds)
                    dets, obs = circ.compile_detector_sampler(seed=seed).sample(
                        sw_shots, separate_observables=True)
                    dets = np.asarray(dets, dtype=bool)
                    obs = np.asarray(obs, dtype=bool)
                    t0 = time.perf_counter()
                    pred = np.asarray(sw.decode_batch(dets), dtype=bool)
                    sw_decode_s = time.perf_counter() - t0
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, 1)
                    sw_fail_mask = np.any(pred != obs, axis=1)
                    sw_fails = int(sw_fail_mask.sum())
                    sw_ler = sw_fails / sw_shots if sw_shots else 0.0
                    try:
                        from qldpc.foundation import stats as _stats
                    except ImportError:  # pragma: no cover
                        from foundation import stats as _stats
                    lo, hi = _stats.wilson_ci(sw_fails, sw_shots)
                    from qldpc.zoo.harness import _lambda_per_round
                    sw_rec = {
                        "name": sw.name,
                        "config": dict(sw.config),
                        "tie_break": sw.tie_break,
                        "fails": sw_fails,
                        "ler": sw_ler,
                        "ler_ci": [lo, hi],
                        "lambda_per_round": _lambda_per_round(sw_ler, rounds),
                        "decode_s": sw_decode_s,
                        "is_anchor": False,
                        "conditional": True,
                        "conditional_note": sw_note,
                    }
                    # gap-to-MLE for the down-scoped SW needs the anchor on the
                    # SAME capped shots: decode the anchor on those shots too.
                    anch_capped = next(
                        (d for d in decoders if d.name == _ANCHOR), None)
                    if anch_capped is not None:
                        a_pred = np.asarray(
                            anch_capped.decode_batch(dets), dtype=bool)
                        if a_pred.ndim == 1:
                            a_pred = a_pred.reshape(-1, 1)
                        a_mask = np.any(a_pred != obs, axis=1)
                        sw_rec["gap_to_mle"] = gap_to_mle_bootstrap(
                            sw_fail_mask, a_mask, n_boot=n_boot, seed=seed)
                    else:
                        sw_rec["gap_to_mle"] = None
                    cell_rec["decoders"].append(sw_rec)
                except Exception as exc:
                    cell_rec["sliding_window_error"] = repr(exc)

            cells.append(cell_rec)

    # 6a. FULL pairwise comparison list across (decoder, p, basis) vs the anchor,
    #     annotated with multiplicity (Holm primary FWER + BH-FDR secondary).
    raw_comparisons = []
    for cell in cells:
        anchor = next((d for d in cell["decoders"]
                       if d["name"] == _ANCHOR), None)
        if anchor is None:
            continue
        for d in cell["decoders"]:
            if d["name"] == _ANCHOR:
                continue
            raw_comparisons.append({
                "decoder": d["name"],
                "p": cell["p"],
                "basis": cell["basis"],
                "fails_a": d["fails"],
                "n_a": d.get("sliding_window_shots", cell["shots"])
                if d.get("conditional") else cell["shots"],
                "fails_b": anchor["fails"],
                "n_b": cell["shots"],
            })
    annotated = multiplicity_grid(raw_comparisons)
    # multiplicity_grid overwrites the "p" key with the raw p-value; preserve the
    # physical p as "p" and expose the raw p-value as "pvalue" for clarity.
    comparisons = []
    for raw, ann in zip(raw_comparisons, annotated):
        rec = dict(ann)
        rec["p"] = raw["p"]            # physical error rate (restore)
        rec["pvalue"] = ann["p"]      # raw two-proportion p-value
        comparisons.append(rec)

    # 6b. LER-accuracy frontier: per decoder, lambda-vs-p and gap-to-MLE-vs-p.
    frontier = {}
    for cell in cells:
        for d in cell["decoders"]:
            name = d["name"]
            series = frontier.setdefault(
                name, {"lambda_vs_p": [], "gap_to_mle_vs_p": []})
            series["lambda_vs_p"].append({
                "p": cell["p"],
                "basis": cell["basis"],
                "ler": d["ler"],
                "lambda_per_round": d["lambda_per_round"],
                "ler_is_bound": cell["ler_is_bound"],
            })
            gm = d.get("gap_to_mle")
            if name != _ANCHOR and gm is not None:
                series["gap_to_mle_vs_p"].append({
                    "p": cell["p"],
                    "basis": cell["basis"],
                    "ratio": gm["ratio"],
                    "lo": gm["lo"],
                    "hi": gm["hi"],
                })

    result = {
        "kind": "matched-cross-decoder-LER-grid",
        "anchor": _ANCHOR,
        "gap_statistic": prereg.get("gap_statistic"),
        "failure_target": failure_target,
        "seed": seed,
        "n_boot": n_boot,
        "p_grid": p_grid,
        "bases": basis_list,
        "rounds": rounds,
        "noise": noise,
        "shots_override": shots_override,
        "dry_scale": dry_scale,
        "include_external": include_external,
        "include_sliding_window": include_sliding_window,
        "prereg_git_head": prereg.get("git_head"),
        "grid_git_head": _git_head(),
        "env": _env_manifest(),
        "sliding_window": sw_block,
        "cells": cells,
        "comparisons": comparisons,
        "frontier": frontier,
    }

    if out_path is not None:
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)

    return result


_PREREG_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "prereg.json")
_GRID_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "zoo_grid.json")
_PILOT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "zoo_grid_pilot.json")


def _print_summary(result):
    print(f"  cells={len(result['cells'])} "
          f"comparisons={len(result['comparisons'])} "
          f"frontier_decoders={len(result['frontier'])}")
    for cell in result["cells"]:
        print(f"  cell p={cell['p']} basis={cell['basis']} "
              f"shots={cell['shots']} g1_match={cell['g1_match']} "
              f"ler_is_bound={cell['ler_is_bound']}")
        for d in cell["decoders"]:
            gm = d.get("gap_to_mle")
            gtxt = (f"gap={gm['ratio']:.3f}[{gm['lo']:.3f},{gm['hi']:.3f}]"
                    if gm else "gap=NA")
            print(f"    {d['name']:14s} ler={d['ler']:.4f} "
                  f"lam={d['lambda_per_round']:.5f} {gtxt} "
                  f"decode_s={d['decode_s']:.2f}")
    sw = result["sliding_window"]
    print(f"  sliding_window: available={sw['available']} "
          f"sec_per_shot={sw['sec_per_shot']} probe={sw['probe']}")


if __name__ == "__main__":
    # `python3 -m qldpc.zoo.run_grid` runs the FULL pre-registered budget grid
    # (all p x basis cells at their frozen shots/cell, the full available zoo)
    # and writes zoo_grid.json — this is the GATED full run the orchestrator
    # launches after sign-off. The `--pilot` flag instead runs the small pilot
    # validation (p=0.005, both bases, 2000 shots) and writes zoo_grid_pilot.json.
    import argparse

    ap = argparse.ArgumentParser(description="matched cross-decoder LER grid")
    ap.add_argument("--pilot", action="store_true",
                    help="run the small pilot (p=0.005, both bases, 2000 shots) "
                         "-> zoo_grid_pilot.json, NOT the full prereg grid")
    ap.add_argument("--out", default=None, help="override output path")
    ap.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    args = ap.parse_args()

    if args.pilot:
        out = args.out or _PILOT_JSON
        result = run_grid(
            _PREREG_JSON, out_path=out,
            p_subset=[0.005], shots_override=2000, bases=["X", "Z"],
            include_sliding_window=True, seed=args.seed,
        )
        print(f"wrote PILOT {out}")
    else:
        # FULL pre-registered budget grid (every cell at its frozen shots/cell).
        out = args.out or _GRID_JSON
        result = run_grid(
            _PREREG_JSON, out_path=out,
            include_sliding_window=True, seed=args.seed,
        )
        print(f"wrote FULL GRID {out}")
    _print_summary(result)
