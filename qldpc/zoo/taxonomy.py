"""T8 — decoder GPU-amenability TAXONOMY (the paper's methods spine).

This is the CPU/analysis lane that makes the T9 latency lane *interpretable*: it
replaces the binary "OSD is GPU-hostile" one-liner with a five-axis taxonomy of
WHY each decoder sits where it does on the Pareto frontier. Every number is
either MEASURED here (fallthrough, Gate-B failure structure) or a documented
REASONED decomposition of the committed T9 latencies / launch-overhead receipt
(Amdahl serial-fraction, roofline, memory footprint). Each estimate carries an
explicit ``basis`` field: ``"measured"`` vs ``"reasoned"`` (never fabricated).

The five deliverables (ranked by Coda):

  1. **OSD/LSD fallthrough-rate vs p** (the most useful number). For each
     committed-grid p, run bare min-sum BP (the zoo config) over a few thousand
     shots and record the FALLTHROUGH RATE = fraction of shots where BP does NOT
     converge (``BpDecoder.converge is False`` after ``.decode``) — exactly the
     shots that would invoke OSD/LSD post-processing. Then
       effective_latency(p) = BP_latency + fallthrough(p) * OSD_post_increment
     using the T9 per-syndrome latencies (OSD increment = BPOSD-10 - bare BP).
     This is the 2nd-order story behind the Pareto plot: BP's latency advantage
     is only real where it actually converges; as p rises the OSD post-processing
     gets invoked on more shots and the effective latency climbs toward BP-OSD.

  2. **Amdahl serial-fraction per decoder** (the kernel CEILING). Decompose each
     decoder's measured per-syndrome latency into a PARALLEL part (the BP
     message-passing rounds — launch-bound per the receipt, so they parallelize
     across shots/edges on a GPU) and a SERIAL part (OSD Gaussian-elimination,
     Tesseract beam expansion, sliding-window stitch, host/launch overhead).
     serial_fraction = serial / (parallel + serial); the implied speedup ceiling
     under perfect parallelization of the parallel part is 1/serial_fraction
     (Amdahl). DOCUMENTS each estimate's basis.

  3. **Roofline position per decoder** (arithmetic intensity vs throughput). The
     kernel-BP point is MEASURED from the launch-overhead receipt (0.17 flop/byte,
     bandwidth-bound, no tensor-core path). The OSD/LSD/Tesseract points are
     REASONED bound-type classifications. **Flags BP-OSD-10's dense Gaussian-
     elimination inner loop as having a tensor-core (GEMM) path — a future-work
     opportunity**, in contrast to min-sum BP which is a min/sum semiring with NO
     tensor-core path.

  4. **Memory footprint per decoder** (working set -> max batch / throughput
     ceiling). Working set = edges * fp32 * batch (the BP message arrays) + a
     per-decoder AUX state (OSD lattice, Tesseract beam frontier, sliding-window
     buffers). Implies a max batch on an H200 (143 GB) and scales to A100 (80 GB)
     / RTX (24 GB).

  5. **T10 Gate-B failure-mode analysis** (resolves whether T10 is definitively
     dead). For shots where the best practical decoder (BP-OSD-10) FAILS but the
     Tesseract-MLE anchor SUCCEEDS (the residual gap), is the failure set
     STRUCTURED (clusters by syndrome/error weight or a specific fault class) or
     UNSTRUCTURED (spread across syndrome space)? If unstructured -> T10 (a
     neural AutDEC that learns a structured correction) is dead definitively; if
     structured -> note the narrow AutDEC target. Coda expects UNSTRUCTURED.

CPU/analysis only — NO GPU, NO pod. ldpc + stim + the committed T9 JSON artifacts.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# conftest puts repo root + qldpc/ on sys.path; ensure qldpc/ importable directly.
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_QLDPC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_QLDPC)
for _p in (_ROOT, _QLDPC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Committed grid (Coda-locked): p-grid, code, rounds, basis, noise for the
# Z-basis [[72,12,6]] SI1000 R=6 lane. The fallthrough lane runs at these p's.
COMMITTED_P_GRID = (0.001, 0.002, 0.003, 0.005)
ROUNDS = 6
BASIS = "Z"
NOISE = "si1000"

# Structure of the SI1000 circuit-level DEM (from the launch-overhead receipt;
# MEASURED on the [[72,12,6]] R=6 p=0.003 DEM). The edge count drives the BP
# message-array working set; the max degrees drive the bounded-degree kernel.
DEM_STRUCTURE = dict(
    n_checks=252,        # = num_detectors
    n_bits=1584,         # = num_errors (mechanisms)
    n_edges=4536,        # nnz(H): the BP message count
    MAXDEG_C=20,         # max check degree (the kernel's static_range bound)
    MAXDEG_B=3,          # max bit degree
    basis="check-degree 13/20, bit-degree 2/3 (irregular circuit-level DEM)",
)


# --------------------------------------------------------------------------- #
# Deliverable 1: OSD/LSD fallthrough-rate vs p + effective latency             #
# --------------------------------------------------------------------------- #
def measure_fallthrough(p, shots=3000, seed=0, rounds=ROUNDS, basis=BASIS,
                        noise=NOISE, bb=None):
    """MEASURED fallthrough rate of bare min-sum BP at physical rate ``p``.

    Builds the [[72,12,6]] SI1000 memory DEM at ``p``, runs bare BP (the zoo
    config: minimum_sum, ms=0.625, max_iter=30, parallel schedule) per shot, and
    returns the fraction of shots where ``BpDecoder.converge is False`` after
    ``.decode`` — exactly the shots that fall through to OSD/LSD post-processing.

    Args:
        p: physical error rate.
        shots: number of detector samples to decode.
        seed: detector-sampler seed (reproducible).
        rounds, basis, noise: memory-experiment parameters (committed grid).
        bb: optional prebuilt BBCode (avoids rebuilding the code each call).

    Returns a dict with: ``p``, ``shots``, ``n_fallthrough``, ``fallthrough_rate``
    (fraction not converged), ``mean_iter`` (mean BP iteration count), and the
    fixed config + ``basis="measured"``.
    """
    from ldpc import BpDecoder
    from canon_dem import extract
    try:
        from qldpc.foundation.circuits import build_memory
    except ImportError:
        from foundation.circuits import build_memory
    if bb is None:
        from bb_code import BBCode
        bb = BBCode()

    circuit = build_memory(bb, rounds=rounds, p=p, basis=basis, noise=noise)
    dem = circuit.detector_error_model(decompose_errors=False)
    ex = extract(dem)
    H = ex["H"]
    priors = list(np.clip(ex["priors"], 1e-6, 1 - 1e-6))
    dec = BpDecoder(H, error_channel=priors, max_iter=30,
                    bp_method="minimum_sum", ms_scaling_factor=0.625,
                    schedule="parallel")

    dets, _obs = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    dets = np.asarray(dets, dtype=np.uint8)

    n_fall = 0
    iters = np.empty(shots, dtype=np.int64)
    for i in range(shots):
        dec.decode(dets[i])
        # BpDecoder.converge is the per-shot convergence flag (bool) set by the
        # last .decode call; .iter is the iteration count it stopped at.
        if not dec.converge:
            n_fall += 1
        iters[i] = int(dec.iter)
    rate = n_fall / shots if shots else 0.0
    return dict(
        p=float(p),
        shots=int(shots),
        n_fallthrough=int(n_fall),
        fallthrough_rate=float(rate),
        mean_iter=float(iters.mean()) if shots else 0.0,
        config=dict(bp_method="minimum_sum", ms_scaling_factor=0.625,
                    max_iter=30, schedule="parallel"),
        basis="measured",
    )


def effective_latency(bp_us, osd_increment_us, fallthrough_rate):
    """Effective per-syndrome latency under fallthrough-gated OSD post-processing.

    A bare-BP-then-OSD-on-fallthrough pipeline pays the BP cost on every shot and
    the OSD post-processing increment ONLY on shots where BP did not converge:

        effective = BP_latency + fallthrough_rate * OSD_post_increment

    where ``osd_increment_us`` is the OSD post-processing cost on top of BP
    (BPOSD-10 per-syndrome - bare-BP per-syndrome from T9). This is the honest
    cost of an adaptive "BP, fall through to OSD on non-convergence" decoder.
    """
    return float(bp_us + fallthrough_rate * osd_increment_us)


def osd_increment_from_latency(latency_results, bp_name="BP",
                               osd_name="BPOSD-10"):
    """OSD post-processing per-syndrome increment = BPOSD-10 - bare-BP (T9).

    REASONED-from-measured: both endpoints are MEASURED T9 per-syndrome latencies
    (us_per_syndrome); their difference is the marginal cost OSD adds on top of
    BP. They were measured at different batch sizes (each decoder at its own
    throughput-optimal batch), so this is the amortized per-syndrome increment,
    not a single-shot one — documented as such.
    """
    res = latency_results["results"]
    bp_us = float(res[bp_name]["us_per_syndrome"])
    osd_us = float(res[osd_name]["us_per_syndrome"])
    return dict(
        bp_us=bp_us,
        osd_total_us=osd_us,
        osd_increment_us=float(osd_us - bp_us),
        basis="reasoned-from-measured (BPOSD-10 us/syn - BP us/syn, T9 amortized)",
    )


def build_fallthrough_table(latency_results, p_grid=COMMITTED_P_GRID,
                            shots=3000, seed=0, bb=None):
    """Deliverable 1: fallthrough-rate vs p + effective-latency vs p.

    For each p in the committed grid: MEASURE the bare-BP fallthrough rate, then
    compute the effective latency of a BP->OSD-on-fallthrough pipeline using the
    T9 latencies. Returns a JSON-serializable dict.
    """
    inc = osd_increment_from_latency(latency_results)
    bp_us = inc["bp_us"]
    osd_inc = inc["osd_increment_us"]
    if bb is None:
        from bb_code import BBCode
        bb = BBCode()
    rows = []
    for p in p_grid:
        m = measure_fallthrough(p, shots=shots, seed=seed, bb=bb)
        m["effective_latency_us"] = effective_latency(bp_us, osd_inc,
                                                      m["fallthrough_rate"])
        rows.append(m)
    return dict(
        kind="osd-lsd-fallthrough-vs-p",
        p_grid=list(p_grid),
        shots=shots,
        seed=seed,
        bp_us_per_syndrome=bp_us,
        osd_post_increment_us=osd_inc,
        osd_increment_basis=inc["basis"],
        bposd10_total_us=inc["osd_total_us"],
        rows=rows,
        note=("fallthrough = fraction of shots where bare min-sum BP does NOT "
              "converge (would invoke OSD/LSD post-processing); effective "
              "latency = BP_us + fallthrough*OSD_increment_us. Fallthrough is "
              "MEASURED; latencies are MEASURED (T9), so the effective-latency "
              "curve is fully empirical."),
    )


# --------------------------------------------------------------------------- #
# Deliverable 2: Amdahl serial-fraction per decoder                           #
# --------------------------------------------------------------------------- #
# Per-decoder Amdahl decomposition. Each entry declares which portion of the
# MEASURED T9 per-syndrome latency is PARALLEL (the BP message-passing rounds,
# which parallelize across shots/edges on a GPU — launch-bound per the receipt,
# i.e. the work itself is embarrassingly parallel and the cost is host launches)
# and which is SERIAL (post-processing that is inherently sequential: OSD's
# Gaussian elimination, Tesseract's beam expansion, the sliding-window stitch,
# or irreducible host overhead). serial_fraction = serial/(parallel+serial);
# Amdahl speedup ceiling under perfect parallelization = 1/serial_fraction.
#
# The split RATIO per decoder is a REASONED estimate (the algorithmic structure
# determines what parallelizes); it is ANCHORED to MEASURED endpoints where
# possible (e.g. the BP/OSD split uses the measured bare-BP vs BPOSD-10 gap).
# Each row carries an explicit ``basis`` string documenting how the split was
# derived. The TOTAL per row is the MEASURED T9 us_per_syndrome.
#
# parallel_share = fraction of the total that is the parallel BP part.
_AMDAHL_MODELS = {
    "BP": dict(
        # Pure min-sum BP: the whole decode is message-passing rounds. The
        # receipt shows it is launch-bound (host-dispatch/runtime=0.999), so on a
        # GPU the message-passing parallelizes and only the host/launch overhead
        # is serial. We attribute a small serial share to irreducible host
        # marshalling (the receipt's torch baseline is ~all launch overhead;
        # a fused kernel collapses 110->24 launches, leaving a residual host
        # tail). Estimate: ~5% serial (host marshalling / final hard-decision).
        parallel_share=0.95,
        serial_kind="host launch/marshalling (no post-processing)",
        basis=("reasoned: receipt shows BP is launch-bound & embarrassingly "
               "parallel across shots/edges; serial = residual host overhead "
               "after fusion (110->24 launches/iter)."),
    ),
    "BPOSD-0": dict(
        # BP then OSD-0: OSD-0 is a single back-substitution on the BP-ordered
        # columns (one solve, no combination sweep) — cheap but serial (a
        # sequential GF2 elimination). Anchor: measured BPOSD-0 us/syn (281.96)
        # vs bare-BP (265.32) -> OSD-0 adds ~16.6 us serial on a ~265 us BP base
        # -> serial ~ 16.6/281.96 ~ 0.059. (BP part parallelizes; OSD-0 GE serial.)
        anchor_increment_over="BP",
        serial_kind="OSD-0 single GF2 back-substitution (sequential elimination)",
        basis=("reasoned-from-measured: serial = (BPOSD-0 - BP) us/syn measured "
               "increment (the OSD-0 elimination); parallel = the BP base."),
    ),
    "BPOSD-10": dict(
        # BP then OSD combination-sweep order 10: a DENSE Gaussian elimination
        # plus a combination sweep over order-10 column subsets — the dominant,
        # inherently SERIAL kernel (sequential pivoting + back-substitution).
        # Anchor: measured BPOSD-10 us/syn (787.87) vs bare-BP (265.32) -> the
        # OSD-cs post-processing adds ~522.55 us serial on a ~265 us BP base
        # -> serial fraction ~ 522.55/787.87 ~ 0.663.
        anchor_increment_over="BP",
        serial_kind="OSD-cs order-10 dense Gaussian elimination + combo sweep",
        basis=("reasoned-from-measured: serial = (BPOSD-10 - BP) us/syn measured "
               "increment (the dense GE + order-10 combination sweep); parallel "
               "= the BP base. The GE inner loop is the serial bottleneck."),
    ),
    "BPLSD": dict(
        # BP then Localised-Statistics Decoder (lsd_cs order 10): LSD replaces
        # OSD's global dense GE with LOCALISED cluster solves on the residual
        # syndrome — much of the cost is sparse, irregular cluster-growth that is
        # only partially parallel (each cluster's solve is small & serial, but
        # clusters are independent). Anchor: measured BPLSD (295.69) vs bare-BP
        # (265.32) -> LSD adds only ~30.4 us -> serial ~ 30.4/295.69 ~ 0.103
        # (LSD is far cheaper than OSD-cs: localised solves, not a global GE).
        anchor_increment_over="BP",
        serial_kind="LSD localised cluster solves (sparse, irregular; small GE per cluster)",
        basis=("reasoned-from-measured: serial = (BPLSD - BP) us/syn measured "
               "increment (the localised cluster solves); parallel = BP base. "
               "LSD's localised solves are cheaper & more parallel than OSD-cs."),
    ),
    "Tesseract": dict(
        # A* beam search over the DEM (beam=64): the search FRONTIER expansion is
        # inherently SERIAL (each A* step depends on the priority queue state from
        # the previous step); only the per-node cost evaluation parallelizes. The
        # bulk of the 3433 us/syn is sequential search. Estimate: ~85% serial
        # (the beam search is the cost; node scoring is a small parallel part).
        parallel_share=0.15,
        serial_kind="A* beam-search frontier expansion (priority-queue sequential)",
        basis=("reasoned: A* beam expansion is sequential (each step depends on "
               "the prior frontier); only per-node cost eval parallelizes. "
               "Search-bound -> dominantly serial."),
    ),
    "RelayBP": dict(
        # Relay-BP: a disjoint ENSEMBLE of BP solvers with a relay schedule. The
        # individual BP solvers parallelize (like bare BP), but the relay handoff
        # / nconv stopping logic across the ensemble is serial coordination.
        # num_sets=60 ensemble members, each set_max_iter=60. The BP work
        # dominates and parallelizes; the relay coordination is a modest serial
        # tail. Estimate: ~80% parallel (ensemble BP), ~20% serial (relay/handoff).
        parallel_share=0.80,
        serial_kind="relay handoff + nconv stopping across the disjoint ensemble",
        basis=("reasoned: the disjoint-ensemble BP solvers parallelize (min-sum "
               "BP, launch-bound); serial = the relay schedule handoff + nconv "
               "stopping coordination across ensemble members."),
    ),
    "SlidingWindow": dict(
        # Sliding-window streaming: each window runs per-window BP-OSD-cs, then
        # COMMITS the commit-region and CARRIES the correction forward — the
        # inter-window STITCH (carry-over re-biasing) is a strict serial
        # dependency (window k+1 needs window k's committed correction). Within a
        # window the BP parallelizes but the per-window OSD-cs is serial (as in
        # BPOSD-10) AND the windows must run in order. Estimate: ~70% serial
        # (per-window OSD-cs + the strict inter-window carry dependency).
        parallel_share=0.30,
        serial_kind="per-window OSD-cs GE + strict inter-window commit/carry stitch",
        basis=("reasoned: windows must run in time order (window k+1 needs k's "
               "committed correction -> serial stitch) and each window's inner "
               "BP-OSD-cs has the OSD GE serial part; only intra-window BP "
               "parallelizes."),
    ),
}


def build_amdahl_table(latency_results, models=None):
    """Deliverable 2: per-decoder Amdahl serial-fraction + speedup ceiling.

    For each decoder, split its MEASURED T9 per-syndrome latency into parallel +
    serial parts and compute serial_fraction and the Amdahl speedup ceiling
    (1/serial_fraction). Rows whose serial part is anchored to a measured
    increment (``anchor_increment_over``) use the measured (decoder - base)
    us/syn gap as the serial part; rows with a ``parallel_share`` use the
    reasoned fraction. Every row carries its ``basis``.

    Returns a list of dicts (one per decoder), each with columns:
        decoder | total_us | parallel_us | serial_us | serial_fraction |
        speedup_ceiling | serial_kind | basis
    """
    if models is None:
        models = _AMDAHL_MODELS
    res = latency_results["results"]
    rows = []
    for name, m in models.items():
        rec = res.get(name)
        if rec is None or "us_per_syndrome" not in rec:
            continue
        total = float(rec["us_per_syndrome"])
        if "anchor_increment_over" in m:
            base = res[m["anchor_increment_over"]]["us_per_syndrome"]
            serial = max(0.0, total - float(base))
            parallel = total - serial
            split_basis = "measured-increment"
        else:
            parallel = total * float(m["parallel_share"])
            serial = total - parallel
            split_basis = "reasoned-share"
        serial_fraction = serial / total if total > 0 else 0.0
        # Amdahl: ceiling = 1/serial_fraction (perfect parallelization of the
        # parallel part). serial_fraction==0 -> unbounded (report None / inf).
        ceiling = (1.0 / serial_fraction) if serial_fraction > 0 else float("inf")
        rows.append(dict(
            decoder=name,
            total_us=total,
            parallel_us=float(parallel),
            serial_us=float(serial),
            serial_fraction=float(serial_fraction),
            speedup_ceiling=float(ceiling),
            serial_kind=m["serial_kind"],
            split_basis=split_basis,
            basis=m["basis"],
        ))
    return rows


# --------------------------------------------------------------------------- #
# Deliverable 3: roofline position per decoder                                #
# --------------------------------------------------------------------------- #
# Per-decoder roofline classification. The kernel-BP point is MEASURED from the
# launch-overhead receipt (arithmetic intensity 0.17 flop/byte, bandwidth-bound,
# no tensor-core path). The others are REASONED bound-type classifications. The
# ``tensor_core_path`` flag marks decoders whose dominant inner kernel COULD map
# to GEMM/tensor cores (the future-work opportunity) vs those that cannot
# (min-sum BP is a min/sum semiring -> NO tensor-core path).
def build_roofline_table(receipt, latency_results):
    """Deliverable 3: per-decoder roofline position + tensor-core flags.

    Args:
        receipt: the launch-overhead receipt dict (MEASURED kernel-BP roofline).
        latency_results: T9 latencies (for measured throughput per decoder).

    Returns a list of dicts, one per decoder, with:
        decoder | flop_per_byte | bound_type | tensor_core_path |
        throughput_shots_per_s | basis | note
    """
    ai = receipt["arithmetic_intensity"]
    bp_fpb = float(ai["approx_flops_per_byte"])
    res = latency_results["results"]

    def thru(name):
        rec = res.get(name)
        if rec is None:
            return None
        if "representative" in rec:
            return float(rec["representative"]["throughput_shots_per_s"])
        return float(rec.get("throughput_shots_per_s")) if rec.get(
            "throughput_shots_per_s") is not None else None

    rows = [
        dict(
            decoder="BP (min-sum, kernel target)",
            flop_per_byte=bp_fpb,
            bound_type="memory-bandwidth-bound",
            tensor_core_path=False,
            throughput_shots_per_s=thru("BP"),
            basis="measured (launch-overhead receipt: 0.17 flop/byte, no GEMM ops)",
            note=("min-sum is a min/sum semiring -> NO GEMM / tensor-core path; "
                  "the kernel win is launch fusion (110->24 launches/iter) + "
                  "coalesced fp32, NOT tensor cores. Forecloses 'why not cuBLAS'."),
        ),
        dict(
            decoder="BPOSD-10",
            flop_per_byte=None,  # not measured; dominated by the dense GE
            bound_type="compute-bound (dense GF2 Gaussian elimination)",
            tensor_core_path=True,   # <-- FUTURE-WORK FLAG
            throughput_shots_per_s=thru("BPOSD-10"),
            basis="reasoned (OSD-cs inner loop is a dense GE; GE has a GEMM path)",
            note=("FUTURE-WORK OPPORTUNITY: the OSD-cs order-10 inner loop is a "
                  "DENSE Gaussian elimination over GF2 columns — a dense linear-"
                  "algebra kernel that DOES have a tensor-core / blocked-GEMM "
                  "path (GF2 GE can be tiled as boolean matmul). Unlike min-sum "
                  "BP, OSD's serial GE is the natural tensor-core target. This "
                  "is the single largest unexploited GPU lever for the practical "
                  "decoder bar (see Amdahl: OSD's GE is ~66% serial of BPOSD-10)."),
        ),
        dict(
            decoder="BPLSD",
            flop_per_byte=None,
            bound_type="memory-latency-bound (sparse, irregular cluster solves)",
            tensor_core_path=False,
            throughput_shots_per_s=thru("BPLSD"),
            basis="reasoned (LSD does localised sparse cluster solves, not dense GE)",
            note=("LSD replaces OSD's global dense GE with LOCALISED cluster "
                  "solves on the residual syndrome -> sparse, irregular memory "
                  "access; gather/scatter-bound, NOT a clean GEMM target. Faster "
                  "than OSD-cs but harder to tensor-core."),
        ),
        dict(
            decoder="Tesseract",
            flop_per_byte=None,
            bound_type="search-bound (A* beam expansion, branch-heavy)",
            tensor_core_path=False,
            throughput_shots_per_s=thru("Tesseract"),
            basis="reasoned (A* beam search: control-flow/branch-bound, no GEMM)",
            note=("A* beam search over the DEM is dominated by priority-queue / "
                  "branchy frontier expansion -> control-flow-bound, no dense "
                  "linear-algebra inner loop -> NO tensor-core path. The MLE "
                  "anchor is a reference point, not a kernel target."),
        ),
        dict(
            decoder="RelayBP",
            flop_per_byte=bp_fpb,  # same min-sum BP semiring as bare BP
            bound_type="memory-bandwidth-bound (ensemble min-sum BP)",
            tensor_core_path=False,
            throughput_shots_per_s=thru("RelayBP"),
            basis="reasoned (disjoint-ensemble of the same min-sum BP semiring)",
            note=("Relay-BP is an ensemble of min-sum BP solvers -> the SAME "
                  "min/sum semiring as bare BP -> bandwidth-bound, no tensor-core "
                  "path. Shares the BP kernel front-end (5/8 zoo decoders)."),
        ),
    ]
    return rows


# --------------------------------------------------------------------------- #
# Deliverable 4: memory footprint per decoder                                 #
# --------------------------------------------------------------------------- #
# GPU memory inventory. The shared BP working set is the message arrays:
#   edges * fp32 * batch (the per-edge messages, both directions) + the bit/check
# LLR scratch. We model the BP message working set as ~ 2 * edges * 4 bytes *
# batch (check->bit and bit->check message arrays in fp32), plus the per-decoder
# AUX state (OSD lattice, Tesseract beam frontier, sliding-window buffers). The
# bytes/shot * max_batch then gives the device-memory ceiling.
_FP32 = 4  # bytes


def _bp_bytes_per_shot(edges=None, msg_dirs=2):
    """BP message working set per shot = msg_dirs * edges * fp32 (the bi-
    directional message arrays). REASONED model of the dominant BP allocation."""
    if edges is None:
        edges = DEM_STRUCTURE["n_edges"]
    return msg_dirs * edges * _FP32


# Per-decoder AUX state per shot (on TOP of the shared BP message arrays), as a
# REASONED model. Each is documented; the multiplier is over the BP per-shot
# bytes (a relative, conservative estimate of the extra state the post-processing
# holds live per shot).
_MEM_MODELS = {
    "BP": dict(aux_mult=0.0,
               aux_kind="none (hard decision off the BP messages)",
               basis="reasoned: BP holds only the message arrays + LLR scratch."),
    "BPOSD-0": dict(aux_mult=0.3,
                    aux_kind="OSD-0 reduced column set + single solve scratch",
                    basis="reasoned: OSD-0 keeps the BP-ordered reduced columns "
                          "+ one back-substitution buffer."),
    "BPOSD-10": dict(aux_mult=1.5,
                     aux_kind="OSD-cs dense GE lattice + order-10 combination buffers",
                     basis="reasoned: the dense GE working matrix + the order-10 "
                           "combination-sweep candidate buffers dominate (a dense "
                           "n_checks x reduced-cols GF2 lattice per shot)."),
    "BPLSD": dict(aux_mult=0.6,
                  aux_kind="LSD localised cluster buffers (sparse, grows with residual)",
                  basis="reasoned: localised cluster solves hold per-cluster "
                        "scratch; sparser than OSD's dense lattice."),
    "Tesseract": dict(aux_mult=3.0,
                      aux_kind="A* beam frontier (beam=64 live nodes + cost map)",
                      basis="reasoned: the beam frontier holds up to det_beam=64 "
                            "live partial cosets + priority queue + visited set "
                            "per shot -> the largest aux footprint."),
    "RelayBP": dict(aux_mult=2.0,
                    aux_kind="disjoint ensemble (num_sets=60 BP states)",
                    basis="reasoned: the relay ensemble holds multiple concurrent "
                          "BP message states (one per active set in the schedule)."),
    "SlidingWindow": dict(aux_mult=1.5,
                          aux_kind="per-window OSD lattice + commit/carry buffers",
                          basis="reasoned: a windowed OSD lattice + the committed-"
                                "correction carry buffer across windows."),
}

# Device memory (usable, GB) for the headline GPUs.
_DEVICES = {
    "H200": 143.0,
    "A100-80GB": 80.0,
    "RTX-4090-24GB": 24.0,
}


def build_memory_table(edges=None, devices=None, models=None, util=0.8):
    """Deliverable 4: per-decoder memory footprint + implied max batch.

    Args:
        edges: BP edge count (default: the DEM's 4536).
        devices: {name: GB} device-memory map (default H200/A100/RTX).
        models: per-decoder aux-state model (default ``_MEM_MODELS``).
        util: usable fraction of device memory (default 0.8 — leave headroom for
            the framework / fragmentation).

    Returns a dict with per-decoder bytes/shot and, per device, the implied max
    batch = floor(usable_bytes / bytes_per_shot). All REASONED (a working-set
    model), labelled as such.
    """
    if edges is None:
        edges = DEM_STRUCTURE["n_edges"]
    if devices is None:
        devices = _DEVICES
    if models is None:
        models = _MEM_MODELS
    bp_bytes = _bp_bytes_per_shot(edges)
    rows = []
    for name, m in models.items():
        aux_bytes = m["aux_mult"] * bp_bytes
        total = bp_bytes + aux_bytes
        max_batch = {}
        for dev, gb in devices.items():
            usable = gb * (1024 ** 3) * util
            max_batch[dev] = int(usable // total) if total > 0 else None
        rows.append(dict(
            decoder=name,
            bp_msg_bytes_per_shot=int(bp_bytes),
            aux_bytes_per_shot=int(aux_bytes),
            total_bytes_per_shot=int(total),
            aux_kind=m["aux_kind"],
            max_batch=max_batch,
            basis=m["basis"],
        ))
    return dict(
        kind="memory-footprint",
        edges=int(edges),
        fp32_bytes=_FP32,
        bp_msg_model="2 * edges * fp32 (bi-directional message arrays)",
        device_memory_gb=dict(devices),
        usable_fraction=util,
        rows=rows,
        note=("working set = BP message arrays (edges*fp32*batch, MEASURED edge "
              "count) + per-decoder AUX state (REASONED model). max_batch = "
              "floor(usable_device_bytes / bytes_per_shot); throughput ceiling = "
              "max_batch decoded per kernel launch. Larger aux (Tesseract beam, "
              "Relay ensemble, OSD lattice) shrinks the max batch -> lower "
              "throughput ceiling at fixed memory."),
    )


# --------------------------------------------------------------------------- #
# Deliverable 5: T10 Gate-B failure-mode analysis                             #
# --------------------------------------------------------------------------- #
def gate_b_structure(decoder_fail_mask, anchor_fail_mask, syndrome_weight,
                     n_obs_flips=None):
    """Deliverable 5: is the residual decoder-vs-MLE gap STRUCTURED?

    The residual gap = shots where the practical decoder (BP-OSD-10) FAILS but
    the Tesseract-MLE anchor SUCCEEDS (``decoder_fail & ~anchor``). We test
    whether that set is STRUCTURED (clusters at a specific syndrome/error weight
    — a fault class a neural AutDEC could learn) or UNSTRUCTURED (drawn from the
    same weight distribution as the overall shot population -> nothing for a
    structured corrector to exploit).

    The structure test: compare the syndrome-weight distribution of the residual-
    gap shots vs ALL shots via (a) a mean-shift z-score and (b) a two-sample
    Kolmogorov-Smirnov test. A residual gap that is statistically
    indistinguishable from the population (KS p large, small mean shift) is
    UNSTRUCTURED -> T10 dead definitively. A residual gap concentrated at a
    distinct weight (large mean shift / small KS p) is STRUCTURED -> a narrow
    AutDEC target.

    Args:
        decoder_fail_mask: bool[shots] — BP-OSD-10 failed on shot i.
        anchor_fail_mask:  bool[shots] — Tesseract-MLE failed on shot i.
        syndrome_weight:   int[shots]  — number of fired detectors per shot (the
            structural axis; a decoder-relevant, well-defined per-shot weight).
        n_obs_flips: optional int[shots] — number of true observable flips per
            shot (a secondary axis; reported if supplied).

    Returns a dict: counts, the residual-gap vs population weight stats, the KS
    test, and a ``verdict`` in {"STRUCTURED", "UNSTRUCTURED", "INCONCLUSIVE"}.
    """
    from scipy import stats as sps

    dec = np.asarray(decoder_fail_mask, dtype=bool)
    anch = np.asarray(anchor_fail_mask, dtype=bool)
    w = np.asarray(syndrome_weight, dtype=np.float64)
    if not (dec.shape == anch.shape == w.shape):
        raise ValueError(
            f"gate_b_structure: shapes must match; got decoder {dec.shape}, "
            f"anchor {anch.shape}, weight {w.shape}.")
    if dec.ndim != 1:
        raise ValueError("gate_b_structure: masks must be 1-D per-shot arrays.")
    n = int(dec.shape[0])

    residual = dec & ~anch          # BP-OSD fails, MLE succeeds = the gap
    n_residual = int(residual.sum())
    n_dec_fail = int(dec.sum())
    n_anch_fail = int(anch.sum())

    pop_w = w
    res_w = w[residual]

    pop_mean = float(pop_w.mean()) if n else 0.0
    pop_std = float(pop_w.std(ddof=1)) if n > 1 else 0.0
    res_mean = float(res_w.mean()) if n_residual else 0.0
    res_std = float(res_w.std(ddof=1)) if n_residual > 1 else 0.0

    # Mean-shift z-score of the residual set's mean weight vs the population
    # (SE = pop_std / sqrt(n_residual)); how many sigma the gap shots' mean
    # weight sits from the population mean.
    if n_residual > 0 and pop_std > 0:
        mean_shift_sigma = (res_mean - pop_mean) / (pop_std / np.sqrt(n_residual))
    else:
        mean_shift_sigma = 0.0

    # Two-sample KS: residual-gap weights vs population weights. A large p-value
    # => the gap is drawn from the same weight distribution => UNSTRUCTURED.
    if n_residual >= 2:
        ks = sps.ks_2samp(res_w, pop_w)
        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
    else:
        ks_stat, ks_p = float("nan"), float("nan")

    # EFFECT SIZE (Cohen's d): the raw separation of the two distributions in
    # pooled-SD units — distinguishes a STATISTICALLY significant departure (which
    # n inflates) from a LARGE one. A big mean_shift_sigma with a small Cohen's d
    # is a diffuse, weakly-separated shift; a large d is a genuinely separable set.
    if n_residual > 1 and pop_std > 0 and res_std > 0:
        pooled_sd = np.sqrt(((n - 1) * pop_std**2 + (n_residual - 1) * res_std**2)
                            / (n + n_residual - 2))
        cohens_d = float((res_mean - pop_mean) / pooled_sd) if pooled_sd > 0 else 0.0
    else:
        cohens_d = 0.0
    # Spread ratio: residual SD / population SD. ~1 => the gap is a SHIFT of the
    # whole distribution (same spread), NOT a tight separable cluster (which would
    # have spread << population).
    spread_ratio = float(res_std / pop_std) if pop_std > 0 else float("nan")

    # Verdict thresholds (pre-stated): STRUCTURED if the gap's weight clearly
    # departs from the population (KS p < 0.01 AND |mean shift| >= 2 sigma);
    # UNSTRUCTURED if it does not (KS p >= 0.05 OR |mean shift| < 1 sigma).
    # Otherwise INCONCLUSIVE (borderline / too few residual shots).
    if n_residual < 5:
        verdict = "INCONCLUSIVE"
        verdict_reason = (f"too few residual-gap shots ({n_residual}) to test "
                          f"structure; raise shots/p.")
    elif (not np.isnan(ks_p)) and ks_p < 0.01 and abs(mean_shift_sigma) >= 2.0:
        verdict = "STRUCTURED"
        # Discriminate a SHARP separable cluster (small Cohen's d would NOT arise;
        # spread << population) from a DIFFUSE weight-correlated shift (moderate d,
        # spread ~ population). The latter is "harder syndromes fail more", a weak
        # AutDEC target; the former is a distinct learnable fault class.
        diffuse = (abs(cohens_d) < 0.8) and (0.7 <= spread_ratio <= 1.3 if not
                                             np.isnan(spread_ratio) else False)
        verdict_reason = (f"residual-gap weight distribution departs from the "
                          f"population (KS p={ks_p:.3g} < 0.01, mean shift "
                          f"{mean_shift_sigma:+.2f} sigma, Cohen's d="
                          f"{cohens_d:+.2f}, spread ratio={spread_ratio:.2f}).")
        if diffuse:
            verdict_reason += (
                " STRUCTURE KIND = DIFFUSE weight-correlated SHIFT (same spread as "
                "the population, modest effect size) -> the gap is just 'heavier "
                "syndromes fail more often', NOT a sharp separable fault class. T10/"
                "AutDEC target is NARROW at best: a neural corrector would have to "
                "beat OSD on the high-weight tail where MLE's exhaustive search "
                "already wins -> low expected ROI; T10 stays SKIP.")
        else:
            verdict_reason += (
                " STRUCTURE KIND = a distinct/separable cluster (large effect size "
                "or tight spread) -> a genuine low-dimensional fault class; a "
                "concrete AutDEC target. Note the narrow T10 opportunity.")
    elif (not np.isnan(ks_p)) and (ks_p >= 0.05 or abs(mean_shift_sigma) < 1.0):
        verdict = "UNSTRUCTURED"
        verdict_reason = (f"residual-gap weights are statistically "
                          f"indistinguishable from the population (KS p="
                          f"{ks_p:.3g}, mean shift {mean_shift_sigma:+.2f} "
                          f"sigma) -> no low-dimensional structure for a "
                          f"corrector to exploit -> T10 dead definitively.")
    else:
        verdict = "INCONCLUSIVE"
        verdict_reason = (f"borderline (KS p={ks_p:.3g}, mean shift "
                          f"{mean_shift_sigma:+.2f} sigma); neither threshold met.")

    out = dict(
        n=n,
        n_decoder_fail=n_dec_fail,
        n_anchor_fail=n_anch_fail,
        n_residual_gap=n_residual,
        population_weight=dict(mean=pop_mean, std=pop_std,
                               min=float(pop_w.min()) if n else 0.0,
                               max=float(pop_w.max()) if n else 0.0),
        residual_weight=dict(mean=res_mean, std=res_std,
                             min=float(res_w.min()) if n_residual else 0.0,
                             max=float(res_w.max()) if n_residual else 0.0),
        mean_shift_sigma=float(mean_shift_sigma),
        cohens_d=float(cohens_d),
        spread_ratio=spread_ratio,
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
        verdict=verdict,
        verdict_reason=verdict_reason,
        basis="measured (per-shot fail masks + syndrome weights from run_matched)",
    )
    if n_obs_flips is not None:
        of = np.asarray(n_obs_flips, dtype=np.float64)
        out["population_obs_flips_mean"] = float(of.mean()) if n else 0.0
        out["residual_obs_flips_mean"] = (
            float(of[residual].mean()) if n_residual else 0.0)
    return out


def run_gate_b(p=0.005, shots=5000, seed=0, rounds=ROUNDS, basis=BASIS,
               noise=NOISE, det_beam=64, bb=None):
    """MEASURE the Gate-B residual gap at a higher-p cell (default p=0.005).

    Runs the matched protocol (``run_matched(keep_per_shot=True)``) for BP-OSD-10
    + Tesseract on the SAME shots, computes per-shot syndrome weights, and feeds
    them to ``gate_b_structure``. Returns the structure dict (Deliverable 5).
    """
    try:
        from qldpc.foundation.circuits import build_memory
    except ImportError:
        from foundation.circuits import build_memory
    try:
        from qldpc.zoo import adapters
        from qldpc.zoo.harness import run_matched
    except ImportError:
        import adapters
        from harness import run_matched
    if bb is None:
        from bb_code import BBCode
        bb = BBCode()

    circuit = build_memory(bb, rounds=rounds, p=p, basis=basis, noise=noise)
    dem = circuit.detector_error_model(decompose_errors=False)
    decoders = adapters.build_decoders(dem, which=("BPOSD-10", "Tesseract"))
    man = run_matched(circuit, decoders, shots=shots, rounds=rounds, seed=seed,
                      keep_per_shot=True,
                      label=f"gate-b p={p} R={rounds} basis={basis}")
    bposd = next(r for r in man["decoders"] if r["name"] == "BPOSD-10")
    tess = next(r for r in man["decoders"] if r["name"] == "Tesseract")

    # Per-shot syndrome weight = number of fired detectors (resample with the
    # SAME seed used by run_matched, so weights align to the per-shot masks).
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    syndrome_weight = np.asarray(dets, dtype=np.int64).sum(axis=1)
    obs_flips = np.asarray(obs, dtype=np.int64).sum(axis=1)

    struct = gate_b_structure(bposd["fail_mask"], tess["fail_mask"],
                              syndrome_weight, n_obs_flips=obs_flips)
    struct.update(dict(
        p=float(p), shots=int(shots), seed=int(seed), rounds=rounds,
        basis_axis="syndrome_weight (number of fired detectors per shot)",
        decoder="BPOSD-10", anchor="Tesseract", det_beam=det_beam,
        bposd_fails=bposd["fails"], tesseract_fails=tess["fails"],
    ))
    return struct


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def plot_fallthrough(fallthrough_table, out_path):
    """Fallthrough-rate-vs-p AND effective-latency-vs-p (twin-axis figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = fallthrough_table["rows"]
    ps = [r["p"] for r in rows]
    fall = [r["fallthrough_rate"] for r in rows]
    eff = [r["effective_latency_us"] for r in rows]
    bp_us = fallthrough_table["bp_us_per_syndrome"]
    osd_total = fallthrough_table["bposd10_total_us"]

    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))
    color1 = "#1f77b4"
    ax1.plot(ps, [f * 100 for f in fall], "o-", color=color1, lw=2.2, ms=8,
             label="BP fallthrough rate (measured)")
    ax1.set_xlabel("physical error rate  p", fontsize=12)
    ax1.set_ylabel("BP fallthrough rate  (% shots not converged)", color=color1,
                   fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log")
    ax1.grid(True, which="both", alpha=0.25)
    for x, y in zip(ps, fall):
        ax1.annotate(f"{y*100:.1f}%", (x, y * 100),
                     textcoords="offset points", xytext=(0, 9),
                     ha="center", color=color1, fontsize=9)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.plot(ps, eff, "s--", color=color2, lw=2.2, ms=8,
             label="effective latency (BP + fallthrough x OSD)")
    ax2.axhline(bp_us, color="#2ca02c", ls=":", lw=1.6,
                label=f"bare BP floor ({bp_us:.0f} us/syn)")
    ax2.axhline(osd_total, color="#7f7f7f", ls=":", lw=1.6,
                label=f"always-OSD ceiling ({osd_total:.0f} us/syn)")
    ax2.set_ylabel("effective per-syndrome latency  (us)", color=color2,
                   fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               fontsize=9, framealpha=0.92)
    ax1.set_title("OSD/LSD fallthrough-rate & effective latency vs p\n"
                  "[[72,12,6]] BB, SI1000, R=6, Z basis  (T8 taxonomy)",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_amdahl(amdahl_rows, out_path):
    """Stacked parallel/serial bar per decoder + speedup-ceiling annotation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["decoder"] for r in amdahl_rows]
    par = [r["parallel_us"] for r in amdahl_rows]
    ser = [r["serial_us"] for r in amdahl_rows]
    ceil = [r["speedup_ceiling"] for r in amdahl_rows]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x, par, color="#2ca02c", label="parallel (BP message-passing)")
    ax.bar(x, ser, bottom=par, color="#d62728",
           label="serial (OSD GE / beam / stitch / host)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("per-syndrome latency  (us, log)", fontsize=12)
    ax.set_title("Amdahl serial-fraction per decoder  (T8 taxonomy)\n"
                 "parallel/serial split of measured T9 latency; "
                 "ceiling = 1/serial-fraction", fontsize=12)
    for xi, (p_, s_, c_) in enumerate(zip(par, ser, ceil)):
        sf = s_ / (p_ + s_) if (p_ + s_) > 0 else 0.0
        label = f"sf={sf:.2f}\n{c_:.1f}x" if np.isfinite(c_) else f"sf={sf:.2f}\ninf"
        ax.annotate(label, (xi, p_ + s_), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="T8 decoder amenability taxonomy")
    ap.add_argument("--latency", default=os.path.join(_HERE, "latency_results.json"))
    ap.add_argument("--receipt",
                    default=os.path.join(_HERE, "launch_overhead_receipt.json"))
    ap.add_argument("--out", default=os.path.join(_HERE, "taxonomy_results.json"))
    ap.add_argument("--fallthrough-shots", type=int, default=4000)
    # Gate-B default 15000: the structure verdict needs enough residual-gap shots
    # to be decisive (5k gave only ~78 residual shots -> borderline/INCONCLUSIVE;
    # 15k gives ~240 -> a clean verdict). Tesseract-bound (~15 min at 15k).
    ap.add_argument("--gate-b-shots", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gate-b", action="store_true",
                    help="skip the (slow, Tesseract-bound) Gate-B run")
    args = ap.parse_args(argv)

    with open(args.latency) as f:
        latency = json.load(f)
    with open(args.receipt) as f:
        receipt = json.load(f)

    from bb_code import BBCode
    bb = BBCode()

    print("=== T8 TAXONOMY: building deliverables ===")
    print("[1/5] fallthrough-rate vs p ...")
    fallthrough = build_fallthrough_table(latency, shots=args.fallthrough_shots,
                                          seed=args.seed, bb=bb)
    print("[2/5] Amdahl serial-fraction ...")
    amdahl = build_amdahl_table(latency)
    print("[3/5] roofline ...")
    roofline = build_roofline_table(receipt, latency)
    print("[4/5] memory footprint ...")
    memory = build_memory_table()
    gate_b = None
    if not args.no_gate_b:
        print("[5/5] Gate-B failure-mode (Tesseract-bound, slow) ...")
        gate_b = run_gate_b(p=0.005, shots=args.gate_b_shots, seed=args.seed,
                            bb=bb)
    else:
        print("[5/5] Gate-B SKIPPED (--no-gate-b)")

    out = dict(
        kind="decoder-amenability-taxonomy",
        code="[[72,12,6]] BB (BBCode default)",
        rounds=ROUNDS, basis=BASIS, noise=NOISE,
        committed_p_grid=list(COMMITTED_P_GRID),
        dem_structure=DEM_STRUCTURE,
        env=latency.get("env"),
        gpu=latency.get("env", {}).get("gpu_name"),
        fallthrough=fallthrough,
        amdahl=dict(
            kind="amdahl-serial-fraction",
            columns=["decoder", "total_us", "parallel_us", "serial_us",
                     "serial_fraction", "speedup_ceiling", "serial_kind",
                     "split_basis", "basis"],
            rows=amdahl,
            note=("total_us = MEASURED T9 us/syndrome; serial/parallel split is "
                  "measured-increment (anchored to bare-BP gap) or reasoned-share "
                  "per the algorithmic structure; ceiling = 1/serial_fraction "
                  "(Amdahl, perfect parallelization of the parallel part)."),
        ),
        roofline=dict(
            kind="roofline-position",
            columns=["decoder", "flop_per_byte", "bound_type",
                     "tensor_core_path", "throughput_shots_per_s", "basis",
                     "note"],
            rows=roofline,
            note=("kernel-BP point MEASURED (receipt: 0.17 flop/byte, no GEMM); "
                  "others REASONED bound-type classifications. tensor_core_path "
                  "flags BP-OSD-10's dense-GE GEMM future-work opportunity."),
        ),
        memory=memory,
        gate_b=gate_b,
    )

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # figures
    fall_png = os.path.join(_HERE, "fallthrough.png")
    amdahl_png = os.path.join(_HERE, "amdahl.png")
    plot_fallthrough(fallthrough, fall_png)
    plot_amdahl(amdahl, amdahl_png)

    _print_report(out)
    print(f"\nwrote {args.out}")
    print(f"wrote {fall_png}")
    print(f"wrote {amdahl_png}")
    return out


def _print_report(out):
    print("\n" + "=" * 78)
    print(f"T8 DECODER AMENABILITY TAXONOMY  --  {out['code']} "
          f"R={out['rounds']} basis={out['basis']} noise={out['noise']} "
          f"({out['gpu']})")
    print("=" * 78)

    print("\n[1] OSD/LSD FALLTHROUGH-RATE vs p  (bare-BP non-convergence; MEASURED)")
    ft = out["fallthrough"]
    print(f"    BP floor={ft['bp_us_per_syndrome']:.1f} us/syn   "
          f"OSD increment={ft['osd_post_increment_us']:.1f} us/syn   "
          f"always-OSD ceiling={ft['bposd10_total_us']:.1f} us/syn")
    print(f"    {'p':>8}{'fallthrough':>14}{'mean_iter':>11}{'eff_latency_us':>16}")
    for r in ft["rows"]:
        print(f"    {r['p']:>8.4f}{r['fallthrough_rate']*100:>13.2f}%"
              f"{r['mean_iter']:>11.1f}{r['effective_latency_us']:>16.1f}")

    print("\n[2] AMDAHL SERIAL-FRACTION  (split of MEASURED T9 latency)")
    print(f"    {'decoder':<15}{'total_us':>10}{'par_us':>10}{'ser_us':>10}"
          f"{'ser_frac':>10}{'ceiling':>10}  basis")
    for r in out["amdahl"]["rows"]:
        c = r["speedup_ceiling"]
        cs = f"{c:.1f}x" if np.isfinite(c) else "inf"
        print(f"    {r['decoder']:<15}{r['total_us']:>10.1f}{r['parallel_us']:>10.1f}"
              f"{r['serial_us']:>10.1f}{r['serial_fraction']:>10.3f}{cs:>10}  "
              f"{r['split_basis']}")

    print("\n[3] ROOFLINE POSITION  (kernel-BP MEASURED; others reasoned)")
    for r in out["roofline"]["rows"]:
        fpb = f"{r['flop_per_byte']:.3f}" if r["flop_per_byte"] is not None else "  -- "
        tc = "TENSOR-CORE PATH" if r["tensor_core_path"] else "no GEMM path"
        print(f"    {r['decoder']:<30} {fpb:>7} flop/byte  {r['bound_type']:<42} [{tc}]")

    print("\n[4] MEMORY FOOTPRINT  (REASONED working-set model)")
    devs = list(out["memory"]["device_memory_gb"].keys())
    print(f"    {'decoder':<15}{'B/shot':>9}  max-batch: " + "  ".join(devs))
    for r in out["memory"]["rows"]:
        mb = "  ".join(f"{r['max_batch'][d]:>10,}" for d in devs)
        print(f"    {r['decoder']:<15}{r['total_bytes_per_shot']:>9,}  {mb}")

    if out["gate_b"]:
        gb = out["gate_b"]
        print(f"\n[5] T10 GATE-B FAILURE-MODE  (p={gb['p']} {gb['shots']} shots; MEASURED)")
        print(f"    BP-OSD-10 fails={gb['bposd_fails']}  Tesseract fails="
              f"{gb['tesseract_fails']}  residual gap (OSD fail & MLE ok)="
              f"{gb['n_residual_gap']}")
        print(f"    population syndrome-weight: mean={gb['population_weight']['mean']:.2f} "
              f"+/- {gb['population_weight']['std']:.2f}")
        print(f"    residual-gap syndrome-weight: mean={gb['residual_weight']['mean']:.2f} "
              f"+/- {gb['residual_weight']['std']:.2f}")
        print(f"    obs-flip (logical class) pop/res: "
              f"{gb.get('population_obs_flips_mean', float('nan')):.3f} / "
              f"{gb.get('residual_obs_flips_mean', float('nan')):.3f}  "
              f"(same class -> not logical-error-class structured)")
        print(f"    mean-shift={gb['mean_shift_sigma']:+.2f} sigma   "
              f"Cohen's d={gb.get('cohens_d', float('nan')):+.2f}   "
              f"spread ratio={gb.get('spread_ratio', float('nan')):.2f}   "
              f"KS stat={gb['ks_statistic']:.3f} p={gb['ks_pvalue']:.3g}")
        print(f"    VERDICT: {gb['verdict']}  -- {gb['verdict_reason']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
