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


def _cpu_latency(lat, name):
    """us/syndrome (mean) for a CPU decoder from the latency manifest."""
    rec = lat["results"].get(name)
    if not rec or rec.get("error") or rec.get("skipped"):
        return None
    return rec.get("us_per_syndrome")


def _gpu_latency(lat, which):
    """us/syndrome (mean) for a GPU decoder's representative (largest) batch."""
    rec = lat["results"].get(which)
    if not rec:
        return None
    reps = rec.get("representative")
    if reps and "us_per_syndrome" in reps:
        return reps["us_per_syndrome"]
    return None


def build_points(grid, lat, p):
    """Assemble (label, latency_us, gap, group, marker) per plotted decoder.

    group: 'kernel-BP' variants share BP's gap; others are single points.
    Tesseract is the anchor at gap=1.0. Returns a list of point dicts."""
    pts = []

    # --- BP kernel variants (share bare-BP's gap-to-MLE; latency left-shift) - #
    bp_gap = _gap_at_p(grid, "BP", p)
    if bp_gap:
        gap_bp = bp_gap[0]
        cpu_bp = _cpu_latency(lat, "BP")
        torch_bp = _gpu_latency(lat, "bp_gpu")
        triton_bp = _gpu_latency(lat, "bp_triton")
        for lab, us, mk in [
            ("CPU-BP (ldpc)", cpu_bp, "o"),
            ("torch-GPU-BP", torch_bp, "s"),
            ("Triton-kernel-BP", triton_bp, "*"),
        ]:
            if us is not None:
                pts.append(dict(label=lab, latency_us=us, gap=gap_bp,
                                group="kernel-BP", marker=mk))

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
        us = _cpu_latency(lat, grid_name)
        if us is None:
            continue
        if grid_name == "Tesseract":
            gap = (1.0, 1.0, 1.0)        # anchor: gap=1.0 by construction
        else:
            gap = _gap_at_p(grid, grid_name, p)
        if gap is None:
            continue
        pts.append(dict(label=lab, latency_us=us, gap=gap[0],
                        group="single", marker=mk))
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


def make_figure(grid, lat, p, out_path):
    pts = build_points(grid, lat, p)
    if not pts:
        raise RuntimeError("no plottable points (check JSON inputs)")

    fig, ax = plt.subplots(figsize=(12, 7.5))

    # --- axis ranges (give headroom for annotations) ----------------------- #
    xs = [pp["latency_us"] for pp in pts]
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
        "Triton-kernel-BP": (10, -16),
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
        ax.scatter(pp["latency_us"], pp["gap"], marker=pp["marker"], s=size,
                   color=color, edgecolors="black", linewidths=0.8, zorder=5)
        off = label_off.get(pp["label"], (8, 6))
        ax.annotate(pp["label"], (pp["latency_us"], pp["gap"]),
                    textcoords="offset points", xytext=off, fontsize=9,
                    fontweight=("bold" if pp["marker"] == "*" else "normal"),
                    zorder=6)

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

    ax.set_xlabel("decode latency per syndrome (us, log scale)  --  MEASURED on H200",
                  fontsize=11)
    ax.set_ylabel(f"LER gap to exact-MLE  (decoder LER / Tesseract-MLE; p={p}, X+Z mean)",
                  fontsize=11)
    ax.set_title("qLDPC circuit-level decoder Pareto frontier  --  [[72,12,6]] BB, "
                 f"SI1000, d=6 R=6\nlatency x accuracy; the fused Triton min-sum "
                 "kernel pulls BP left along the latency axis",
                 fontsize=12.5)
    ax.grid(True, which="both", ls=":", alpha=0.3)

    # legend: groups + frontier
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
        Line2D([], [], color="0.25", lw=2.2, alpha=0.6, label="Pareto frontier"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(f"wrote {out_path}")
    # also dump the plotted points for the record
    return dict(p=p, points=pts,
                frontier=[pp["label"] for pp in front])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pareto hero figure")
    ap.add_argument("--grid", default=os.path.join(_HERE, "zoo_grid.json"))
    ap.add_argument("--latency", default=os.path.join(_HERE, "latency_results.json"))
    ap.add_argument("--out", default=os.path.join(_HERE, "pareto.png"))
    ap.add_argument("--p", type=float, default=0.003)
    args = ap.parse_args(argv)

    with open(args.grid) as f:
        grid = json.load(f)
    with open(args.latency) as f:
        lat = json.load(f)
    info = make_figure(grid, lat, args.p, args.out)
    print(json.dumps({k: (v if k != "points" else
                          [{kk: vv for kk, vv in pp.items()} for pp in v])
                      for k, v in info.items()}, indent=2)[:2000])
    return info


if __name__ == "__main__":
    main()
