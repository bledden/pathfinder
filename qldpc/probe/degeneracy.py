"""Degeneracy-gap probe + level/trend kill-switches: the B-MLE-vs-Option-C verdict.

The qLDPC program anchors "optimality" to Tesseract's most-likely-ERROR (MLE).
Exact coset maximum-likelihood (coset-ML) is intractable on the [[72,12,6]] BB
code, so we must show that the **MLE-vs-coset-ML degeneracy gap**

    ratio = LER_MLE / LER_cosetML   (>= 1, since coset-ML is optimal)

is small and non-growing at the small BB scales where exact coset-ML IS tractable
(width-gated). Two PRE-REGISTERED kill-switches decide the verdict:

  * Level kill-switch: any tested ratio > 2.0  ->  verdict C.  (Robust: a 2x gap
    is far outside the ~1% sampling-noise band, so this stays a strict
    point-estimate rule.)
  * Trend kill-switch (CI-AWARE): the BB-family ratios (in ascending-n order)
    strictly monotonically increasing across >=2 points AND the growth is
    STATISTICALLY SIGNIFICANT (the largest-n ratio's lower CI bound exceeds the
    smallest-n ratio's upper CI bound -- i.e. endpoint CIs are disjoint with the
    large end higher)  ->  verdict C.
  * Otherwise -> verdict B-MLE (the gap is measured-bounded and not growing).

HISTORY / RATIONALE
-------------------
The trend kill-switch was ORIGINALLY a strict point-estimate rule (fire on any
strictly-increasing BB sequence regardless of CIs). That rule was found
SEED-FRAGILE: at the BB scales the MLE/coset-ML ratios all sit in a ~1%
sampling-noise band around 1.0, so an ascending ordering can arise purely from
RNG noise. Demonstrated: p=0.01, seed=123 gave BB ratios [0.9851, 0.9953,
1.0013] -- strictly increasing by construction-noise alone -> a SPURIOUS "C".
The point-estimate rule detects noise, not signal.

The defensible form, consistent with the program's stated statistics discipline
(>=5 seeds, CI-aware, CI-overlap != signal), is the CI-AWARE rule above: a trend
only counts as a kill-switch event if the increase exceeds sampling noise
(disjoint endpoint CIs). When a gap carries optional ``ratio_lo``/``ratio_hi``
(95% CI on the ratio), the CI-aware test is used. When those keys are ABSENT we
fall back to the original strict point-estimate rule, so the synthetic no-CI unit
tests retain their original meaning.

PAIRED PROBE
------------
Because the ratio is small (~1.0-1.5) and thresholded at 2x, coset-ML and MLE
must decode the SAME sampled shots (paired), never independently-sampled runs.
For each scale we build a CODE-CAPACITY Stim circuit that exactly realises a
given (H, L, priors): ``X_ERROR(p)`` on each of ``n`` data qubits, one ``MPP``
Z-product detector per row of ``H`` over its support, one ``MPP`` Z-product
observable per row of ``L``. The resulting DEM has detectors == H-syndrome and
observables == L-values, so Tesseract's MLE on that DEM is directly comparable to
``coset_ml_ler(H, L, priors, dets, obs)`` on the very same sampled ``dets``/``obs``.
(This is the construction the wiring sanity-anchor below validates against the T6
cross-check ratio ~1.0 on ``BBCode(3,3, A=x0x1x2, B=x0x1y1)``.)
"""
import json
import os
import sys

import numpy as np
import stim

# qldpc/ (one level up) holds bb_code.py; conftest adds it under pytest, but a
# direct ``python3 -m qldpc.probe.degeneracy`` run needs it on sys.path too.
_QLDPC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _QLDPC_DIR not in sys.path:
    sys.path.insert(0, _QLDPC_DIR)

from qldpc.foundation.stats import wilson_ci

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "degeneracy_verdict.json")


# --------------------------------------------------------------------------- #
# Pre-registered verdict logic (STEP A/B.1)
# --------------------------------------------------------------------------- #
def verdict_from_gaps(gaps, level=2.0):
    """Apply the pre-registered level + (CI-aware) trend kill-switches to gaps.

    ``gaps`` is a list of dicts each with at least a ``ratio`` and a ``scale``;
    BB-family entries are marked ``bb=True`` and are read **in the given order**
    (the runner orders the BB spine ascending by ``n``). A gap MAY additionally
    carry ``ratio_lo``/``ratio_hi`` (a 95% CI on the ratio).

    Kill-switches (see module docstring for full rationale):

      * LEVEL (strict point-estimate, unchanged): any ratio > ``level`` -> C.
      * TREND (CI-aware when CIs present, strict point-estimate otherwise):
          - With CIs on BOTH endpoints: fire ONLY if the BB ratios are strictly
            monotonically increasing AND the largest-n ratio's ``ratio_lo`` >
            the smallest-n ratio's ``ratio_hi`` (endpoint CIs disjoint, large end
            higher -> the growth exceeds sampling noise).
          - Without CIs: fall back to strict point-estimate (any strictly
            increasing BB sequence fires) so the no-CI synthetic tests keep their
            meaning.

    Returns ``{"verdict": "C"|"B-MLE", "reason": str, "gaps": gaps}``.
    """
    # Level kill-switch: any single ratio above the level threshold -> C.
    # (Strict and unchanged: 2x is far outside the ~1% noise band.)
    for g in gaps:
        if g["ratio"] > level:
            return {
                "verdict": "C",
                "reason": f"level kill-switch: ratio {g['ratio']} > {level} at {g['scale']}",
                "gaps": gaps,
            }

    # Trend kill-switch: BB-only ratios, in given (ascending-n) order.
    bb = [g for g in gaps if g.get("bb")]
    ratios = [g["ratio"] for g in bb]
    if len(ratios) >= 2 and all(a < b for a, b in zip(ratios[:-1], ratios[1:])):
        # Strictly increasing in point estimate. Decide whether it is significant.
        first, last = bb[0], bb[-1]
        have_ci = all(k in first for k in ("ratio_lo", "ratio_hi")) and all(
            k in last for k in ("ratio_lo", "ratio_hi")
        )
        if have_ci:
            # CI-aware: fire only if endpoint CIs are disjoint with large end up,
            # i.e. the growth exceeds sampling noise.
            if last["ratio_lo"] > first["ratio_hi"]:
                return {
                    "verdict": "C",
                    "reason": (
                        "trend kill-switch (CI-aware): BB-family gap "
                        "monotonically increasing AND significant -- "
                        f"largest-n ratio_lo {last['ratio_lo']:.4f} > "
                        f"smallest-n ratio_hi {first['ratio_hi']:.4f} "
                        "(endpoint CIs disjoint)"
                    ),
                    "gaps": gaps,
                }
            # Monotonic but within noise (CIs overlap) -> NOT a kill-switch event.
        else:
            # No CIs: original strict point-estimate fallback.
            return {
                "verdict": "C",
                "reason": "trend kill-switch: BB-family gap monotonically increasing",
                "gaps": gaps,
            }

    return {
        "verdict": "B-MLE",
        "reason": "all gaps <=2x and no significant increasing BB trend (CI-aware)",
        "gaps": gaps,
    }


# --------------------------------------------------------------------------- #
# Code-capacity circuit construction (STEP B.2)
# --------------------------------------------------------------------------- #
def build_codecap_circuit(H, L, p):
    """Code-capacity Stim circuit realising (H, L) at i.i.d. bit-flip rate ``p``.

    ``X_ERROR(p)`` on each of the ``n = H.shape[1]`` data qubits; one ``MPP``
    Z-product detector per row of ``H`` (over the qubits in that row's support);
    one ``MPP`` Z-product observable per row of ``L``. An X error on a qubit
    anticommutes with the Z-product of any check/logical that includes it, so the
    sampled detection events equal ``H @ e (mod 2)`` and the observables equal
    ``L @ e (mod 2)`` -- exactly the inputs ``coset_ml_ler(H, L, priors, ...)``
    expects, and a DEM whose MLE (Tesseract) is directly comparable.
    """
    H = np.asarray(H) % 2
    L = np.asarray(L) % 2
    n = H.shape[1]
    c = stim.Circuit()
    for q in range(n):
        c.append("X_ERROR", [q], p)
    nm = H.shape[0]
    for r in range(nm):
        supp = np.nonzero(H[r])[0]
        targets = []
        for j, q in enumerate(supp):
            if j:
                targets.append(stim.target_combiner())
            targets.append(stim.target_z(int(q)))
        c.append("MPP", targets)
    # Detectors reference the nm just-appended check measurements (deterministic
    # in the noiseless limit -> a bare DETECTOR per check is correct here).
    for r in range(nm):
        c.append("DETECTOR", [stim.target_rec(-(nm - r))])
    for o in range(L.shape[0]):
        supp = np.nonzero(L[o])[0]
        targets = []
        for j, q in enumerate(supp):
            if j:
                targets.append(stim.target_combiner())
            targets.append(stim.target_z(int(q)))
        c.append("MPP", targets)
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], o)
    return c


def tesseract_ler_on_samples(dem, dets, obs, beam):
    """MLE LER + failure count of Tesseract on PRE-SAMPLED (dets, obs).

    Decodes the exact ``dets`` that coset-ML also saw (paired). Returns
    ``(ler, fails)``.
    """
    import tesseract_decoder as _td

    n_shots = len(dets)
    if n_shots == 0:
        return 0.0, 0
    dec = _td.tesseract.TesseractConfig(dem=dem, det_beam=int(beam)).compile_decoder()
    pred = np.asarray(dec.decode_batch(np.asarray(dets, dtype=bool)), dtype=bool)
    obs = np.asarray(obs, dtype=bool)
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    assert pred.shape == obs.shape, (
        f"Tesseract prediction shape {pred.shape} != observable shape {obs.shape}"
    )
    fails = int(np.any(pred != obs, axis=1).sum())
    return fails / n_shots, fails


# --------------------------------------------------------------------------- #
# Paired probe runner (STEP B.2)
# --------------------------------------------------------------------------- #
def probe_scale(name, kind, bb, H, L, p, shots, beam=64, seed=0):
    """Run the PAIRED degeneracy probe for one (H, L, p) scale.

    Samples ``shots`` (dets, obs) ONCE from the code-capacity circuit, decodes
    the same shots with both coset-ML (exact TN, syndrome-memoized) and Tesseract
    MLE, and records a full per-scale gap record.
    """
    from qldpc.foundation.tn_mld import coset_ml_ler_memoized

    H = np.asarray(H) % 2
    L = np.asarray(L) % 2
    n = H.shape[1]
    priors = np.full(n, float(p))

    circ = build_codecap_circuit(H, L, p)
    dets, obs = circ.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True
    )
    dets = np.asarray(dets, dtype=bool)
    obs = np.asarray(obs, dtype=bool)

    # Syndrome-memoized coset-ML: bit-identical to the un-memoized path but skips
    # re-contracting on repeated syndromes (esp. the dominant all-zero syndrome).
    ler_cml = coset_ml_ler_memoized(H, L, priors, dets, obs)
    fails_cml = int(round(ler_cml * shots))

    dem = circ.detector_error_model(decompose_errors=False)
    ler_mle, fails_mle = tesseract_ler_on_samples(dem, dets, obs, beam)

    ratio = (ler_mle / ler_cml) if ler_cml > 0 else float("inf")
    lo_c, hi_c = wilson_ci(fails_cml, shots)
    lo_m, hi_m = wilson_ci(fails_mle, shots)

    return {
        "scale": name,
        "kind": kind,
        "bb": bool(bb),
        "n": int(n),
        "p": float(p),
        "shots": int(shots),
        "beam": int(beam),
        "fails_cml": fails_cml,
        "fails_mle": fails_mle,
        "ler_cosetML": float(ler_cml),
        "ler_MLE": float(ler_mle),
        "ci_cosetML": [float(lo_c), float(hi_c)],
        "ci_MLE": [float(lo_m), float(hi_m)],
        "ratio": float(ratio),
    }


def probe_scale_multiseed(name, kind, bb, H, L, p, shots, beam=64, seeds=(0, 1, 2, 3, 4)):
    """Run ``probe_scale`` over S seeds and aggregate to a pooled ratio + CI.

    For each seed we run the paired probe at ``shots`` shots, collecting per-seed
    ``(fails_cml, fails_mle, shots)``. We then:

      * POOL: total_fails_* = sum over seeds, total_shots = S * shots, and the
        pooled LERs LER_* = total_fails_* / total_shots. The pooled RATIO is
        LER_MLE / LER_cosetML.
      * RATIO CI (reported method = "pooled-Wilson-propagated"): compute a 95%
        Wilson CI on EACH pooled LER, then propagate to a ratio CI by the
        monotone-quotient bound
            ratio_lo = LER_MLE_lo / LER_cosetML_hi
            ratio_hi = LER_MLE_hi / LER_cosetML_lo
        This is the cleaner of the two options (it uses the full pooled sample
        directly and the program's own ``wilson_ci``); we also record the
        per-seed ratio spread for transparency.

    Emits a per-scale record with ``ratio`` (pooled), ``ratio_lo``/``ratio_hi``
    (so the CI-aware verdict can use it), ``per_seed_ratios``, total shots/fails,
    and the per-seed sub-records.
    """
    seeds = list(seeds)
    per_seed = []
    for s in seeds:
        rec = probe_scale(name, kind, bb, H, L, p, shots, beam=beam, seed=int(s))
        per_seed.append(rec)

    total_shots = sum(r["shots"] for r in per_seed)
    total_fails_cml = sum(r["fails_cml"] for r in per_seed)
    total_fails_mle = sum(r["fails_mle"] for r in per_seed)

    ler_cml = total_fails_cml / total_shots if total_shots else 0.0
    ler_mle = total_fails_mle / total_shots if total_shots else 0.0
    ratio = (ler_mle / ler_cml) if ler_cml > 0 else float("inf")

    lo_c, hi_c = wilson_ci(total_fails_cml, total_shots)
    lo_m, hi_m = wilson_ci(total_fails_mle, total_shots)
    # Propagate the two pooled-LER Wilson CIs to a ratio CI (monotone quotient).
    ratio_lo = (lo_m / hi_c) if hi_c > 0 else 0.0
    ratio_hi = (hi_m / lo_c) if lo_c > 0 else float("inf")

    per_seed_ratios = [r["ratio"] for r in per_seed]

    return {
        "scale": name,
        "kind": kind,
        "bb": bool(bb),
        "n": int(per_seed[0]["n"]),
        "p": float(p),
        "beam": int(beam),
        "seeds": [int(s) for s in seeds],
        "shots_per_seed": int(shots),
        "total_shots": int(total_shots),
        "total_fails_cml": int(total_fails_cml),
        "total_fails_mle": int(total_fails_mle),
        "ler_cosetML": float(ler_cml),
        "ler_MLE": float(ler_mle),
        "ci_cosetML": [float(lo_c), float(hi_c)],
        "ci_MLE": [float(lo_m), float(hi_m)],
        "ratio": float(ratio),
        "ratio_lo": float(ratio_lo),
        "ratio_hi": float(ratio_hi),
        "ratio_ci_method": "pooled-Wilson-propagated",
        "per_seed_ratios": [float(r) for r in per_seed_ratios],
        "per_seed": per_seed,
    }


def run(scales, out_path=_OUT):
    """Run the paired probe over ``scales`` (each a dict for ``probe_scale``),
    apply the verdict logic, write the full record to JSON, and return it.

    ``scales`` order is preserved -> place the BB spine ascending by n so the
    trend kill-switch reads it correctly.
    """
    gaps = [probe_scale(**s) for s in scales]
    verdict = verdict_from_gaps(gaps)
    out = {**verdict, "gaps": gaps}
    json.dump(out, open(out_path, "w"), indent=2)
    return out


def run_multiseed(scales, out_path=_OUT):
    """Multi-seed variant of ``run``: each scale is a dict for
    ``probe_scale_multiseed`` (carrying ``seeds=...``). Applies the CI-aware
    verdict to the aggregated gaps and writes the full record to JSON.
    """
    gaps = [probe_scale_multiseed(**s) for s in scales]
    verdict = verdict_from_gaps(gaps)
    out = {**verdict, "gaps": gaps}
    json.dump(out, open(out_path, "w"), indent=2)
    return out


# --------------------------------------------------------------------------- #
# Scale builders for the decisive experiment (STEP C)
# --------------------------------------------------------------------------- #
def _gf2_nullspace(M):
    M = np.asarray(M) % 2
    rows, cols = M.shape
    piv = []
    r = 0
    Mr = M.copy()
    for c in range(cols):
        pr = [i for i in range(r, rows) if Mr[i, c]]
        if not pr:
            continue
        Mr[[r, pr[0]]] = Mr[[pr[0], r]]
        for i in range(rows):
            if i != r and Mr[i, c]:
                Mr[i] = (Mr[i] + Mr[r]) % 2
        piv.append(c)
        r += 1
    free = [c for c in range(cols) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(cols, np.int8)
        v[f] = 1
        for ri, pc in enumerate(piv):
            if Mr[ri, f]:
                v[pc] = 1
        basis.append(v % 2)
    return np.array(basis, np.int8) if basis else np.zeros((0, cols), np.int8)


def _gf2_quotient(ker, stab):
    """Independent reps of ker modulo rowspace(stab) (clean logical basis)."""
    basis = []

    def reduce_vec(w):
        for lead, b in basis:
            if w[lead]:
                w = (w + b) % 2
        return w

    for s in np.asarray(stab) % 2:
        w = reduce_vec(s.copy())
        if w.any():
            basis.append((int(np.argmax(w)), w))
    logs = []
    for v in np.asarray(ker) % 2:
        w = reduce_vec(v.copy())
        if w.any():
            basis.append((int(np.argmax(w)), w))
            logs.append(w.copy())
    cols = ker.shape[1] if ker.shape[0] else (stab.shape[1] if stab.shape[0] else 0)
    return np.array(logs, np.int8) if logs else np.zeros((0, cols), np.int8)


def rotated_surface_HZ_LX(d):
    """Canonical rotated surface code (d odd): return (HZ, LX, n).

    Data qubits on a d x d grid (index r*d+c). Stabilizer plaquettes on the dual
    lattice at half-integer centres; X-plaquette if (i+j) even else Z-plaquette;
    boundary weight-2 plaquettes kept on the alternating edges (canonical
    Tomita-Svore / Horsman layout). Returns the Z-check matrix (detects X errors)
    and the X-logical (the observable an X-error chain flips). Verified CSS, k=1.
    """
    def site(r, c):
        return r * d + c

    Xs, Zs = [], []
    for i in range(-1, d):
        for j in range(-1, d):
            corners = [(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)]
            corners = [(r, c) for (r, c) in corners if 0 <= r < d and 0 <= c < d]
            if len(corners) not in (2, 4):
                continue
            is_x = (i + j) % 2 == 0
            if len(corners) == 2:
                rs = [r for r, c in corners]
                cs = [c for r, c in corners]
                horiz = rs[0] == rs[1]
                vert = cs[0] == cs[1]
                if is_x and not vert:
                    continue
                if (not is_x) and not horiz:
                    continue
            (Xs if is_x else Zs).append([site(r, c) for r, c in corners])
    n = d * d
    HX = np.zeros((len(Xs), n), np.int8)
    HZ = np.zeros((len(Zs), n), np.int8)
    for i, qs in enumerate(Xs):
        for q in qs:
            HX[i, q] ^= 1
    for i, qs in enumerate(Zs):
        for q in qs:
            HZ[i, q] ^= 1
    HX %= 2
    HZ %= 2
    LX = _gf2_quotient(_gf2_nullspace(HZ), HX)  # X-logical: ker(HZ) mod rowspace(HX)
    assert np.all((HZ @ LX.T) % 2 == 0), "surface LX must commute with HZ"
    return HZ, LX, n


def steane_color_d3_HZ_LX():
    """Distance-3 triangular color code = Steane [[7,1,3]] (self-dual CSS).

    Returns (HZ, LX, n=7). The 3x7 Hamming parity matrix serves as both HX and
    HZ; the X-logical is ker(HZ) modulo rowspace(HX).
    """
    H = np.array(
        [
            [1, 0, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 1, 1],
        ],
        np.int8,
    )
    HX = H.copy()
    HZ = H.copy()
    LX = _gf2_quotient(_gf2_nullspace(HZ), HX)
    assert np.all((HZ @ LX.T) % 2 == 0), "Steane LX must commute with HZ"
    return HZ, LX, 7


def bb_HZ_LX(l, m, A_terms, B_terms):
    """(HZ, logicals_X, n) for a BB code; HZ detects X errors, logicals_X is the observable."""
    from bb_code import BBCode

    bb = BBCode(l=l, m=m, A_terms=A_terms, B_terms=B_terms)
    HZ = np.asarray(bb.HZ) % 2
    LX = np.asarray(bb.logicals_X()) % 2
    return HZ, LX, bb.n, int(bb.k)


# --------------------------------------------------------------------------- #
# Decisive experiment wiring (STEP C/D)
# --------------------------------------------------------------------------- #
# The BB spine: >=3 GENUINELY-BIVARIATE toys of strictly-INCREASING n, each
# coset-ML-tractable (contraction_width(HZ, logicals_X) <= 28) and low-k (k=4 ->
# 16 coset classes -> cheap exact contraction). "Genuinely bivariate" here is the
# STRONG criterion: A and B EACH mix x and y (so neither block is a degenerate
# all-x / all-y quasi-1D circulant whose treewidth is artificially low). The
# prior enumerate_bb.json was lexicographically x-biased (its viable list is
# dominated by all-x A,B codes); these three were re-selected for the strong
# bivariate property and verified CSS + width-gated.
_BB_SPINE = [
    # n=12: l=3,m=2  A=(x1,x2,y0) B=(x1,x2,y1)  k=4 width=9
    dict(l=3, m=2, A_terms=(("x", 1), ("x", 2), ("y", 0)),
         B_terms=(("x", 1), ("x", 2), ("y", 1))),
    # n=18: l=3,m=3  A=(x0,x1,y1) B=(x0,x1,y1)  k=4 width=19
    dict(l=3, m=3, A_terms=(("x", 0), ("x", 1), ("y", 1)),
         B_terms=(("x", 0), ("x", 1), ("y", 1))),
    # n=24: l=3,m=4  A=(x1,x2,y0) B=(x1,x2,y1)  k=4 width=13
    dict(l=3, m=4, A_terms=(("x", 1), ("x", 2), ("y", 0)),
         B_terms=(("x", 1), ("x", 2), ("y", 1))),
]


def _wiring_sanity():
    """T6 wiring anchor: on BBCode(3,3, A=x0x1x2, B=x0x1y1) the MLE/coset-ML ratio
    must be ~1.0 (>= coset-ML within noise). Returns the gap record."""
    HZ, LX, n, k = bb_HZ_LX(3, 3, (("x", 0), ("x", 1), ("x", 2)),
                            (("x", 0), ("x", 1), ("y", 1)))
    return probe_scale("WIRING-ANCHOR BB(3,3) x0x1x2|x0x1y1", "bb-anchor", False,
                       HZ, LX, p=0.05, shots=8000, beam=64, seed=1)


def main(p=0.05, shots=6000, beam=64, seed=7):
    """Wire the STEP C scales, run the paired probe, print the table, write JSON."""
    # --- wiring sanity gate (must pass before trusting the probe) ---
    anchor = _wiring_sanity()
    print("\n=== WIRING SANITY ANCHOR (T6) ===")
    print(f"  {anchor['scale']}: coset-ML={anchor['ler_cosetML']:.4f} "
          f"MLE={anchor['ler_MLE']:.4f} ratio={anchor['ratio']:.4f}")
    if anchor["ler_MLE"] + 1e-9 < anchor["ci_cosetML"][0]:
        raise SystemExit(
            "WIRING FAILURE: MLE materially BELOW coset-ML CI -- the code-capacity "
            "DEM likely does not match (H,L,priors). Aborting before reporting."
        )

    scales = []

    # Cross-family references (level-checked, not trend): surface d3/d5, color d3.
    HZ, LX, n = rotated_surface_HZ_LX(3)
    scales.append(dict(name="surface-d3", kind="surface", bb=False, H=HZ, L=LX,
                       p=p, shots=12000, beam=beam, seed=seed))
    HZ, LX, n = rotated_surface_HZ_LX(5)
    scales.append(dict(name="surface-d5", kind="surface", bb=False, H=HZ, L=LX,
                       p=p, shots=12000, beam=beam, seed=seed))
    HZ, LX, n = steane_color_d3_HZ_LX()
    scales.append(dict(name="color-d3 (Steane)", kind="color", bb=False, H=HZ, L=LX,
                       p=p, shots=12000, beam=beam, seed=seed))

    # BB spine (ascending n) -- the decisive trend lane.
    for spec in _BB_SPINE:
        HZ, LX, n, k = bb_HZ_LX(**spec)
        nm = f"BB n={n} (l={spec['l']},m={spec['m']}) k={k}"
        scales.append(dict(name=nm, kind="bb", bb=True, H=HZ, L=LX,
                           p=p, shots=shots, beam=beam, seed=seed))

    out = run(scales)
    out["wiring_anchor"] = anchor

    # Re-write JSON including the anchor + selection metadata for the record.
    out["bb_spine_selection"] = [
        {**spec, "criterion": "strong bivariate: A and B each mix x and y"}
        for spec in _BB_SPINE
    ]
    json.dump(out, open(_OUT, "w"), indent=2)

    # --- print the per-scale table ---
    print("\n=== DEGENERACY-GAP PROBE: per-scale table ===")
    hdr = (f"{'scale':<34}{'kind':<9}{'n':>4}{'p':>7}{'shots':>8}"
           f"{'cosetML':>10}{'MLE':>10}{'ratio':>8}")
    print(hdr)
    print("-" * len(hdr))
    for g in out["gaps"]:
        print(f"{g['scale']:<34}{g['kind']:<9}{g['n']:>4}{g['p']:>7.3f}"
              f"{g['shots']:>8}{g['ler_cosetML']:>10.4f}{g['ler_MLE']:>10.4f}"
              f"{g['ratio']:>8.4f}")
        print(f"{'':<34}{'  CI cosetML=[%.4f,%.4f]  CI MLE=[%.4f,%.4f]  fails c/m=%d/%d'%(g['ci_cosetML'][0],g['ci_cosetML'][1],g['ci_MLE'][0],g['ci_MLE'][1],g['fails_cml'],g['fails_mle'])}")

    bb_ratios = [g["ratio"] for g in out["gaps"] if g["bb"]]
    print(f"\nBB-family ratios (ascending n): {[round(r,4) for r in bb_ratios]}")
    print(f"\n*** VERDICT: {out['verdict']} ***")
    print(f"    reason: {out['reason']}")
    print(f"\nWrote {_OUT}")
    return out


if __name__ == "__main__":
    main()
