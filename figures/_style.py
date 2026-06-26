"""Paper-grade matplotlib styling for Pathfinder figures.

Matches the modern-blog aesthetic of https://bledden.github.io/blog/* :
seaborn whitegrid background, Helvetica Neue sans-serif, wide aspect ratios,
bold insight-titles, prominent value labels, background-shaded zones for
meaning, callout boxes with arrows.

Palette: a Plotly-Express-inspired modern qualitative set, with a *semantic*
mapping to each decoder family so the reader's eye can track a series across
figures without re-reading legends:

    PyMatching (baseline, algorithmic)   ->  slate grey            #6B7280
    Lange family (prior-art GNN)         ->  warm red / orange     #EF553B / #FFA15A
    Pathfinder family (this work, CNN)   ->  clean blue            #4A90E2
    Pathfinder-Triad (HERO ensemble)     ->  saturated purple      #9333EA
    Hybrid CNN+GNN (negative result)     ->  emerald green         #00CC96
    Union-Find / other baselines         ->  light grey            #B0B0B0

Region shades:
    strict-CI-win  ->  pale gold     #FEF3C7
    cycle-budget   ->  pale green    #DCFCE7

Inspired by Wong's color-blind-safe principles + Plotly Express default palette.
The combination is distinguishable under deuteranopia and protanopia.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAL = {
    # Baseline (algorithmic)
    "pm":          "#6B7280",   # slate grey
    "pm_alt":      "#A1A8B3",   # PyMatching 4-param (lighter)

    # Lange family (prior art — "the thing to beat")
    "lange":       "#EF553B",   # bright red — prior art
    "lange_ft":    "#FFA15A",   # warm orange — fine-tuned variant

    # Pathfinder family (CNN, this work)
    "pf":          "#4A90E2",   # clean blue
    "pf_kd":       "#7DB4EA",   # Pathfinder-KD (lighter blue)
    "pfwl3s":      "#1F6FCC",   # PFWL3S (deeper blue)

    # HERO — Pathfinder-Triad (most saturated)
    "triad":       "#9333EA",   # vivid purple
    "triad_kd":    "#A86BEB",   # KD-variant Triad

    # Other architectures
    "hybrid":      "#00CC96",   # emerald — Hybrid CNN+GNN (the negative result)

    # Baselines / comparators
    "uf":          "#B0B0B0",   # light grey — Union-Find
    "alpha_qubit": "#B45100",   # deep amber — AlphaQubit
    "gu_etal":     "#10B981",   # teal — Gu et al.

    # Background zones
    "win_band":    "#FEF3C7",   # pale gold for strict-CI-win shading
    "budget_band": "#DCFCE7",   # pale mint for cycle-budget region

    # Chrome
    "grid":        "#E5E7EB",
    "axis":        "#1F2937",
    "annotation":  "#111827",
}

# Status colors for the dominance heatmap (avoiding red-green colorblind issues)
STATUS = {
    "win":  "#16A34A",   # vivid green — strict win
    "loss": "#7C3AED",   # purple — strict loss (NOT red, so colorblind-safe vs win)
    "tie":  "#F3F4F6",   # near-white — overlap
    "na":   "#FFFFFF",   # white — N/A
}

def apply():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        # Typography — sans-serif, Helvetica Neue (matches blog aesthetic)
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Helvetica Neue", "Helvetica", "Arial",
                                 "DejaVu Sans"],
        "font.size":            12,
        "font.weight":          "normal",

        # Title (insight-stating, bold)
        "axes.titlesize":       14.5,
        "axes.titleweight":     "bold",
        "axes.titlepad":        12,

        # Axes
        "axes.labelsize":       12.5,
        "axes.labelweight":     "normal",
        "axes.labelpad":        6,
        "axes.linewidth":       1.0,
        "axes.edgecolor":       PAL["axis"],
        "axes.labelcolor":      PAL["axis"],
        "axes.facecolor":       "#FFFFFF",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.spines.left":     True,
        "axes.spines.bottom":   True,

        # Ticks
        "xtick.labelsize":      11,
        "ytick.labelsize":      11,
        "xtick.color":          PAL["axis"],
        "ytick.color":          PAL["axis"],
        "xtick.major.width":    1.0,
        "ytick.major.width":    1.0,
        "xtick.major.pad":      5,
        "ytick.major.pad":      4,

        # Legend (clean, no frame, inside corner)
        "legend.fontsize":      11,
        "legend.frameon":       True,
        "legend.framealpha":    0.92,
        "legend.edgecolor":     "#CBD5E0",
        "legend.facecolor":     "#FFFFFF",
        "legend.borderaxespad": 0.7,

        # Grid (subtle)
        "grid.color":           PAL["grid"],
        "grid.linestyle":       "-",
        "grid.linewidth":       0.7,
        "grid.alpha":           0.85,

        # Figure / save
        "figure.dpi":           140,
        "savefig.dpi":          200,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.20,
        "savefig.facecolor":    "white",
        "figure.facecolor":     "white",

        # Lines & markers
        "lines.linewidth":      2.4,
        "lines.markersize":     8,
        "lines.markeredgewidth": 1.2,
        "lines.markeredgecolor": "white",

        # Bars
        "patch.linewidth":      1.0,
        "patch.edgecolor":      "#1F2937",
    })

def thin_spine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(1.0)
        ax.spines[s].set_color(PAL["axis"])

def insight_title(ax, title, subtitle=None):
    """Bold insight-stating title plus optional smaller subtitle on a second line."""
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=14.5, weight="bold", pad=14)
        # NOTE: matplotlib doesn't easily mix weights within a title; use ax.set_title with two-line text.
    else:
        ax.set_title(title, fontsize=14.5, weight="bold", pad=12)

def footer(ax, text):
    """Small italic provenance line below the chart. Wrap long footers so a single long line
    doesn't force a wide savefig bbox (which left the figure small + left-justified)."""
    import textwrap
    text = "\n".join(textwrap.wrap(text, width=115))
    ax.text(0.0, -0.20, text, ha="left", va="top", transform=ax.transAxes,
            fontsize=11, style="italic", color="#6B7280")

def callout(ax, x, y, text, dx=0, dy=0, color=None, fontsize=10.5):
    """Speech-bubble-style annotation with a thin arrow pointing to (x, y)."""
    if color is None:
        color = PAL["annotation"]
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        fontsize=fontsize, weight="bold", color=color, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor=color, linewidth=1.1, alpha=0.97),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                        connectionstyle="arc3,rad=-0.15"),
    )
