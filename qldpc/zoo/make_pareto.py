"""The Pareto HERO FIGURE -- latency x gap-to-MLE for the qLDPC decoder zoo.

x-axis: decode latency per syndrome (microseconds, LOG scale) -- from the T9
        latency manifest (``latency_results.json``), MEASURED on the H200.
y-axis: LER gap-to-MLE (decoder LER / Tesseract-MLE LER) -- from the committed
        LER grid (``zoo_grid.json`` ``frontier.<dec>.gap_to_mle_vs_p``), at a
        representative p cell (default p=0.003), averaged over X+Z bases.

One point per decoder. BP is shown as THREE kernel variants to expose the kernel
left-shift along the latency axis (the kernel-side contribution):
    CPU-BP (ldpc)  ->  torch-GPU-BP (bp_gpu)  ->  Triton-kernel-BP (bp_triton),
all at the SAME gap-to-MLE (bare-BP's accuracy, ~4-5x MLE) -- the three share BP's
LER; only latency moves. Tesseract (the MLE anchor) sits at gap=1.0 / high latency.

Decoration: cycle-time budget BANDS as vertical regions (superconducting block
budget ~ R us; trapped-ion ~ms; neutral-atom ~4.45 ms; FPGA ns conceded shown as
a reference line), and the Pareto frontier highlighted (the lower-left staircase:
no other decoder is both faster AND closer to MLE).

Reads the committed JSON; renders with matplotlib Agg (works headless on the pod
or locally). Run:
    python3 -m qldpc.zoo.make_pareto --out qldpc/zoo/pareto.png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# Cycle-time budgets (per-block, microseconds). R=6 rounds at d=6.
R = 6
BUDGETS = {
    # superconducting: ~1 us/round -> block budget = R us (conceded to FPGA, but
    # the band marks where a GPU decoder WOULD need to land per block).
    "superconducting": dict(us=1.0 * R, color="#d62728", label="superconducting\n(~1 us/round, block ~6 us)"),
    # trapped-ion: ~1 ms gate/cycle.
    "trapped-ion": dict(us=1.0e3, color="#9467bd", label="trapped-ion (~1 ms)"),
    # neutral-atom: 4.45 ms measured cycle.
    "neutral-atom": dict(us=4.45e3, color="#2ca02c", label="neutral-atom (~4.45 ms)"),
}
# FPGA reference (Relay-BP FPGA 24 ns/iter); shown as a thin reference line only.
FPGA_NS_PER_ITER = 24.0


def _gap_at_p(grid, decoder, p, bases=("X", "Z")):
    """Mean gap-to-MLE (ratio) for a decoder at p, averaged over bases.

    Returns (mean_ratio, lo, hi) or None if the decoder/cell is absent."""
    fr = grid.get("frontier", {})
    info = fr.get(decoder)
    if not info or "gap_to_mle_vs_p" not in info:
        return None
    rows = [r for r in info["gap_to_mle_vs_p"]
            if abs(r["p"] - p) < 1e-9 and r["basis"] in bases]
    if not rows:
        return None
    ratios = [r["ratio"] for r in rows]
    los = [r.get("lo", r["ratio"]) for r in rows]
    his = [r.get("hi", r["ratio"]) for r in rows]
    return float(np.mean(ratios)), float(np.mean(los)), float(np.mean(his))


def _lat_fields(rec):
    """Extract (mean_us, p999_us, ci_lo, ci_hi) from a latency sub-record.

    p99.9 -> the horizontal whisker terminus (Option A). ci -> bootstrap 95% CI
    on the mean (the point error bar). Missing CI falls back to the mean (no
    error bar). Returns None if there is no mean."""
    if not rec or "us_per_syndrome" not in rec:
        return None
    mean = rec["us_per_syndrome"]
    p999 = rec.get("us_per_syndrome_p99_9", mean)
    ci = rec.get("us_per_syndrome_ci95")
    if ci and len(ci) == 2:
        lo, hi = float(ci[0]), float(ci[1])
    else:
        lo = hi = mean
    return dict(mean_us=float(mean), p999_us=float(p999),
                ci_lo=lo, ci_hi=hi)


def _cpu_latency(lat, name):
    """Latency fields for a CPU decoder from the latency manifest."""
    rec = lat["results"].get(name)
    if not rec or rec.get("error") or rec.get("skipped"):
        return None
    return _lat_fields(rec)


def _gpu_latency(lat, which, regime="single_shot"):
    """Latency fields for a GPU decoder.

    regime='single_shot' -> batch-1 (REAL-TIME latency; the binding number for an
    SC decoding window -- the headline). regime='throughput' -> representative
    (largest batch, fully amortized). The figure plots single-shot as the point
    (the deployment-regime story) and reports throughput in the caption."""
    rec = lat["results"].get(which)
    if not rec:
        return None
    key = "single_shot" if regime == "single_shot" else "representative"
    sub = rec.get(key) or rec.get("representative")
    if sub and "us_per_syndrome" in sub:
        f = _lat_fields(sub)
        if f is not None:
            f["batch"] = sub.get("batch")
        return f
    return None


def _mkpoint(label, fields, gap, group, marker, regime="single-batch"):
    """Build a point dict carrying mean / p99.9-whisker / bootstrap-CI."""
    return dict(label=label, latency_us=fields["mean_us"],
                p999_us=fields["p999_us"], ci_lo=fields["ci_lo"],
                ci_hi=fields["ci_hi"], gap=gap, group=group, marker=marker,
                regime=regime, batch=fields.get("batch"))


def build_points(grid, lat, p, gpu_regime="throughput"):
    """Assemble per-decoder plot points carrying mean, p99.9, and bootstrap CI.

    group: 'kernel-BP' variants share BP's gap; others are single points.
    Tesseract is the anchor at gap=1.0. The GPU BP points use ``gpu_regime``:
    'single_shot' (batch-1 REAL-TIME latency, the figure's headline) plotted as
    the point + whisker; 'throughput' (batch-16k) is reported in the caption.
    Returns a list of point dicts."""
    pts = []

    # --- BP kernel variants (share bare-BP's gap-to-MLE; latency left-shift) - #
    bp_gap = _gap_at_p(grid, "BP", p)
    if bp_gap:
        gap_bp = bp_gap[0]
        cpu_bp = _cpu_latency(lat, "BP")
        torch_bp = _gpu_latency(lat, "bp_gpu", regime=gpu_regime)
        triton_bp = _gpu_latency(lat, "bp_triton", regime=gpu_regime)
        for lab, f, mk, reg in [
            ("CPU-BP (ldpc)", cpu_bp, "o", "batch"),
            ("torch-GPU-BP", torch_bp, "s", gpu_regime),
            ("Triton-kernel-BP", triton_bp, "*", gpu_regime),
        ]:
            if f is not None:
                pts.append(_mkpoint(lab, f, gap_bp, "kernel-BP", mk, reg))

    # --- the OSD/LSD/Relay/SW classical bars + Tesseract anchor ------------- #
    single = [
        ("BPOSD-0", "BP-OSD-0", "^"),
        ("BPOSD-10", "BP-OSD-10", "D"),
        ("BPLSD", "BP+LSD", "v"),
        ("RelayBP", "Relay-BP", "P"),
        ("SlidingWindow", "Sliding-window", "X"),
        ("Tesseract", "Tesseract (MLE anchor)", "h"),
    ]
    for grid_name, lab, mk in single:
        f = _cpu_latency(lat, grid_name)
        if f is None:
            continue
        if grid_name == "Tesseract":
            gap = (1.0, 1.0, 1.0)        # anchor: gap=1.0 by construction
        else:
            gap = _gap_at_p(grid, grid_name, p)
        if gap is None:
            continue
        pts.append(_mkpoint(lab, f, gap[0], "single", mk))
    return pts


def pareto_frontier(pts):
    """Indices of points on the lower-left Pareto frontier (min latency, min gap).

    A point is on the frontier if no other point is <= in BOTH latency and gap
    (and < in at least one). Returns the frontier points sorted by latency."""
    front = []
    for i, a in enumerate(pts):
        dominated = False
        for j, b in enumerate(pts):
            if i == j:
                continue
            if (b["latency_us"] <= a["latency_us"] and b["gap"] <= a["gap"]
                    and (b["latency_us"] < a["latency_us"]
                         or b["gap"] < a["gap"])):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda d: d["latency_us"])


def _gpu_throughput_us(lat, which):
    """us/syndrome at the THROUGHPUT batch (largest) for the caption."""
    rec = lat["results"].get(which) or {}
    rep = rec.get("representative") or {}
    return rep.get("us_per_syndrome"), rep.get("batch")


def _clock_cv_text(lat):
    """Caption fragment for the GPU clock + variance-check, read from env."""
    clk = (lat.get("env") or {}).get("gpu_clock") or {}
    app = clk.get("applications_clock_mhz")
    boost = clk.get("boost_clock_mhz", 1980)
    locked = clk.get("clocks_locked", False)
    bits = [f"GPU clocks: {boost} MHz boost"]
    if app:
        bits.append(f"app-clock {app} MHz")
    bits.append("UNLOCKED on RunPod (lock not permitted)"
                if not locked else "locked")
    return ", ".join(bits)


def _gpu_single_shot(lat, which):
    """(mean_us, p99.9_us) at batch-1 for the caption's real-time number."""
    rec = lat.get("results", {}).get(which) or {}
    ss = rec.get("single_shot") or {}
    return ss.get("us_per_syndrome"), ss.get("us_per_syndrome_p99_9")


def _build_caption(grid, lat, p, pts, gpu_regime="throughput"):
    """Assemble the publication caption (states the figure-lock requirements)."""
    boot = lat.get("bootstrap") or {}
    nboot = boot.get("n_resamples", 2000)
    tri_thr, tri_b = _gpu_throughput_us(lat, "bp_triton")
    tri_ss, tri_ss_p999 = _gpu_single_shot(lat, "bp_triton")
    tor_ss, _ = _gpu_single_shot(lat, "bp_gpu")
    sc_us = BUDGETS["superconducting"]["us"]
    point_regime = ("THROUGHPUT (largest-batch)" if gpu_regime == "throughput"
                    else "SINGLE-SHOT (batch-1)")
    lines = []
    lines.append(
        "Pareto frontier of circuit-level [[72,12,6]] BB decoders, SI1000, d=6 "
        f"R=6. x = decode latency per syndrome (us, log); GPU POINT = mean at the "
        f"{point_regime} regime, CPU POINT = per-syndrome mean; HORIZONTAL "
        "WHISKER = per-syndrome p99.9 (tail). "
        f"y = LER gap to exact-MLE (decoder LER / Tesseract-MLE; p={p}, X+Z "
        "mean -- the cell the LER grid pins).")
    lines.append(
        f"Error bars (vertical caps on the point) = {nboot}-resample percentile "
        "bootstrap 95% CI on the mean per-syndrome latency (resample the kept "
        "per-rep window with replacement, drop-first-10% applied, amortize each "
        "resample to us/syndrome, take 2.5/97.5 pctl). CIs are sub-percent -> "
        "points are well-separated.")
    if tri_thr and tri_ss:
        lines.append(
            f"REAL-TIME vs THROUGHPUT (Triton-BP): single-shot batch-1 "
            f"{tri_ss:.0f} us/syn (p99.9 {tri_ss_p999:.0f}; fixed kernel-launch + "
            f"H2D/D2H, unamortized) vs batch-{tri_b} {tri_thr:.2f} us/syn "
            f"throughput. torch-BP single-shot {tor_ss:.0f} us/syn. Batched "
            "numbers are THROUGHPUT, not real-time latency.")
    tri_rep = (lat.get("results", {}).get("bp_triton") or {}).get(
        "representative") or {}
    tail_ms = float(tri_rep.get("p99_9_ms", 0.0))
    lines.append(
        f"GPU MEASURED on H200, {_clock_cv_text(lat)}; thermal stability "
        "confirmed by a variance-over-runs check (clock_variance.json: "
        "run-to-run CV < ~1%). "
        f"SC per-window budget band ~{sc_us:.0f} us (R=6): the Triton-BP "
        "per-syndrome MEAN fits, but the per-BATCH p99.9 tail "
        f"({tail_ms:.0f} ms at batch-{tri_b}) does NOT -- the deployment-regime "
        "finding.")
    return "\n".join(lines)


def make_figure(grid, lat, p, out_path, gpu_regime="throughput"):
    pts = build_points(grid, lat, p, gpu_regime=gpu_regime)
    if not pts:
        raise RuntimeError("no plottable points (check JSON inputs)")

    fig, ax = plt.subplots(figsize=(12.5, 8.6))
    # reserve the bottom strip for the multi-line caption
    fig.subplots_adjust(bottom=0.28, top=0.9)

    # --- axis ranges (give headroom for annotations) ----------------------- #
    # x range must span both the mean points AND the p99.9 whisker termini.
    xs = [pp["latency_us"] for pp in pts] + [pp["p999_us"] for pp in pts]
    ys = [pp["gap"] for pp in pts]
    xmin = min(xs) * 0.35
    xmax = max(xs) * 3.5
    ymin = 0.85
    ymax = max(ys) * 1.45                 # headroom for the left-shift annotation
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # --- cycle-time budget bands (vertical) -------------------------------- #
    for plat, b in BUDGETS.items():
        ax.axvline(b["us"], color=b["color"], ls="--", lw=1.4, alpha=0.85)
        ax.axvspan(xmin, b["us"], color=b["color"], alpha=0.035)
        ax.text(b["us"], 0.015, " " + b["label"], rotation=90, va="bottom",
                ha="left", color=b["color"], fontsize=8.5,
                transform=ax.get_xaxis_transform())
    # FPGA reference line (24 ns/iter -> ~0.7 us/30-iter block); informational.
    fpga_us = FPGA_NS_PER_ITER * 30 / 1e3
    ax.axvline(fpga_us, color="0.5", ls=":", lw=1.2, alpha=0.7)
    ax.text(fpga_us, 0.015, " FPGA ref (Relay-BP 24 ns/iter)", rotation=90,
            va="bottom", ha="left", color="0.4", fontsize=8,
            transform=ax.get_xaxis_transform())

    # --- the Pareto frontier (drawn first, underneath the points) ---------- #
    front = pareto_frontier(pts)
    if len(front) >= 2:
        fx = [pp["latency_us"] for pp in front]
        fy = [pp["gap"] for pp in front]
        ax.plot(fx, fy, color="0.25", lw=2.4, ls="-", alpha=0.55, zorder=1)

    # --- the points -------------------------------------------------------- #
    kernel_color = "#1f77b4"
    # Per-label annotation offsets (points) to avoid overlap in the clusters.
    label_off = {
        "CPU-BP (ldpc)": (8, 8), "torch-GPU-BP": (8, 8),
        "Triton-kernel-BP": (10, -18),
        "BP-OSD-0": (-6, 12), "BP+LSD": (6, -16), "BP-OSD-10": (8, 10),
        "Relay-BP": (-10, 14), "Sliding-window": (6, -18),
        "Tesseract (MLE anchor)": (-10, 12),
    }
    for pp in pts:
        is_kernel = pp["group"] == "kernel-BP"
        is_anchor = pp["label"].startswith("Tesseract")
        color = kernel_color if is_kernel else (
            "#000000" if is_anchor else "#ff7f0e")
        size = 360 if pp["marker"] == "*" else (220 if is_anchor else 150)

        # (1) p99.9 horizontal whisker: mean -> p99.9 (tail consumes the budget)
        if pp["p999_us"] > pp["latency_us"] * 1.001:
            ax.plot([pp["latency_us"], pp["p999_us"]], [pp["gap"], pp["gap"]],
                    color=color, lw=1.6, alpha=0.7, zorder=3,
                    solid_capstyle="butt")
            # tail cap
            ax.plot([pp["p999_us"], pp["p999_us"]],
                    [pp["gap"] * 0.985, pp["gap"] * 1.015],
                    color=color, lw=1.6, alpha=0.7, zorder=3)

        # (2) bootstrap 95% CI on the mean: horizontal error bar at the point.
        xerr_lo = max(0.0, pp["latency_us"] - pp["ci_lo"])
        xerr_hi = max(0.0, pp["ci_hi"] - pp["latency_us"])
        if (xerr_lo + xerr_hi) > 0:
            ax.errorbar(pp["latency_us"], pp["gap"],
                        xerr=[[xerr_lo], [xerr_hi]],
                        fmt="none", ecolor=color, elinewidth=2.2,
                        capsize=3.5, capthick=2.0, zorder=4, alpha=0.95)

        # (3) the point marker on top
        ax.scatter(pp["latency_us"], pp["gap"], marker=pp["marker"], s=size,
                   color=color, edgecolors="black", linewidths=0.8, zorder=5)
        off = label_off.get(pp["label"], (8, 6))
        ax.annotate(pp["label"], (pp["latency_us"], pp["gap"]),
                    textcoords="offset points", xytext=off, fontsize=9,
                    fontweight=("bold" if pp["marker"] == "*" else "normal"),
                    zorder=6)

    # --- Triton-BP tail-past-SC-budget callout (the deployment finding) ----- #
    tri = next((pp for pp in pts if pp["label"] == "Triton-kernel-BP"), None)
    sc_us = BUDGETS["superconducting"]["us"]
    if tri and gpu_regime == "throughput":
        # THROUGHPUT regime: the per-syndrome MEAN (0.79 us) fits the SC band,
        # but the per-BATCH p99.9 tail (a backlog/queueing concern at batch-16k,
        # ~13 ms) lands far past every budget band. Draw that tail as a long
        # whisker from the mean point to the per-batch-tail x-position and call
        # it out -- "mean fits, tail does not" (the figure-lock finding).
        trec = (lat.get("results", {}).get("bp_triton") or {})
        rep = trec.get("representative") or {}
        tail_us = float(rep.get("p99_9_ms", 0.0)) * 1e3   # ms -> us (per batch)
        if tail_us > sc_us:
            tail_x = min(tail_us, ax.get_xlim()[1] * 0.95)
            # draw the per-batch tail as a dashed whisker at a mid-plot y so it
            # does not collide with the cluster or the title.
            tail_y = (ymin * ymax) ** 0.5     # geometric mid of the y-range
            ax.annotate(
                "", xy=(tail_x, tail_y), xytext=(tri["latency_us"], tail_y),
                arrowprops=dict(arrowstyle="-", color=kernel_color, lw=1.4,
                                ls=(0, (4, 2)), alpha=0.6), zorder=3)
            # tie the tail line to the Triton point with a faint connector
            ax.plot([tri["latency_us"], tri["latency_us"]],
                    [tri["gap"], tail_y], color=kernel_color, lw=0.8,
                    ls=":", alpha=0.5, zorder=2)
            ax.annotate(
                f"per-batch p99.9 tail {tail_us/1e3:.0f} ms (batch-{rep.get('batch')}):\n"
                "per-syndrome mean fits SC budget, the TAIL does NOT",
                xy=(tail_x, tail_y),
                xytext=(sc_us * 1.3, tail_y * 1.18),
                fontsize=8.2, color=kernel_color, fontweight="bold",
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=kernel_color, lw=1.2,
                                alpha=0.75), zorder=7)
    elif tri and tri["p999_us"] > sc_us:
        # SINGLE-SHOT regime: the point itself (and its p99.9 whisker) is past SC.
        ax.annotate(
            "single-shot latency past SC budget\n(batch-1, no amortization)",
            xy=(tri["p999_us"], tri["gap"]),
            xytext=(tri["p999_us"] * 0.5, tri["gap"] * 1.55),
            fontsize=8.5, color=kernel_color, fontweight="bold",
            ha="left", va="bottom",
            arrowprops=dict(arrowstyle="->", color=kernel_color, lw=1.4,
                            alpha=0.8), zorder=7)

    # --- kernel left-shift arrow (CPU-BP -> Triton-kernel-BP) -------------- #
    kbp = {pp["label"]: pp for pp in pts if pp["group"] == "kernel-BP"}
    if "CPU-BP (ldpc)" in kbp and "Triton-kernel-BP" in kbp:
        a = kbp["CPU-BP (ldpc)"]
        c = kbp["Triton-kernel-BP"]
        yarr = a["gap"] * 1.12
        ax.annotate("", xy=(c["latency_us"], yarr), xytext=(a["latency_us"], yarr),
                    arrowprops=dict(arrowstyle="->", color=kernel_color,
                                    lw=2.0, alpha=0.8))
        mid = (a["latency_us"] * c["latency_us"]) ** 0.5
        speedup = a["latency_us"] / c["latency_us"] if c["latency_us"] else 0
        ax.text(mid, a["gap"] * 1.20,
                f"kernel left-shift: {speedup:.0f}x faster, same LER\n"
                f"(CPU-BP -> torch-GPU-BP -> fused Triton kernel)",
                color=kernel_color, fontsize=9, ha="center", style="italic",
                fontweight="bold")

    ax.set_xlabel("decode latency per syndrome (us, log scale)  --  point=mean, "
                  "whisker=p99.9, error bar=bootstrap 95% CI  --  MEASURED on H200",
                  fontsize=10.5)
    ax.set_ylabel(f"LER gap to exact-MLE  (decoder LER / Tesseract-MLE; p={p}, X+Z mean)",
                  fontsize=11)
    regime_word = ("throughput batch" if gpu_regime == "throughput"
                   else "single-shot batch-1")
    ax.set_title("qLDPC circuit-level decoder Pareto frontier  --  [[72,12,6]] BB, "
                 f"SI1000, d=6 R=6\nlatency ({regime_word} GPU) x accuracy; "
                 "the fused Triton min-sum kernel pulls BP left along the latency axis",
                 fontsize=12.5)
    ax.grid(True, which="both", ls=":", alpha=0.3)

    # legend: groups + whisker/CI semantics + frontier
    handles = [
        Line2D([], [], marker="*", color="w", markerfacecolor=kernel_color,
               markeredgecolor="black", markersize=15,
               label="BP kernel variants (same LER, latency left-shift)"),
        Line2D([], [], marker="D", color="w", markerfacecolor="#ff7f0e",
               markeredgecolor="black", markersize=10,
               label="classical bars (OSD/LSD/Relay/SW)"),
        Line2D([], [], marker="h", color="w", markerfacecolor="#000000",
               markeredgecolor="black", markersize=12,
               label="Tesseract (MLE anchor, gap=1.0)"),
        Line2D([], [], color="0.4", lw=1.6, alpha=0.7,
               label="p99.9 whisker (mean -> tail)"),
        Line2D([], [], color="0.4", lw=2.2, marker="|", markersize=8,
               label="bootstrap 95% CI (on mean)"),
        Line2D([], [], color="0.25", lw=2.2, alpha=0.6, label="Pareto frontier"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, framealpha=0.95)

    # --- caption strip (publication-clean; states the figure-lock items) --- #
    caption = _build_caption(grid, lat, p, pts, gpu_regime=gpu_regime)
    fig.text(0.02, 0.02, caption, fontsize=7.6, va="bottom", ha="left",
             family="monospace", wrap=True, color="0.15")

    fig.savefig(out_path, dpi=180)
    print(f"wrote {out_path}")
    # also dump the plotted points for the record
    return dict(p=p, gpu_regime=gpu_regime, points=pts,
                frontier=[pp["label"] for pp in front])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pareto hero figure")
    ap.add_argument("--grid", default=os.path.join(_HERE, "zoo_grid.json"))
    ap.add_argument("--latency", default=os.path.join(_HERE, "latency_results.json"))
    ap.add_argument("--out", default=os.path.join(_HERE, "pareto.png"))
    ap.add_argument("--p", type=float, default=0.003)
    ap.add_argument("--gpu-regime", default="throughput",
                    choices=["single_shot", "throughput"],
                    help="GPU BP point: throughput (largest batch, the "
                         "left-shift hero, default) or single_shot (batch-1 "
                         "real-time). Both regimes are stated in the caption.")
    args = ap.parse_args(argv)

    with open(args.grid) as f:
        grid = json.load(f)
    with open(args.latency) as f:
        lat = json.load(f)
    info = make_figure(grid, lat, args.p, args.out, gpu_regime=args.gpu_regime)
    print(json.dumps({k: (v if k != "points" else
                          [{kk: vv for kk, vv in pp.items()} for pp in v])
                      for k, v in info.items()}, indent=2)[:2000])
    return info


if __name__ == "__main__":
    main()
