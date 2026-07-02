"""Generate all 11 paper figures from raw experimental data.

Run from the repo root:
  python3 figures/make_figures.py

Writes figures/fig{01..11}_*.{png,pdf}.

Aesthetic: seaborn whitegrid + Helvetica Neue sans-serif + insight-stating
bold titles + prominent value labels + background-shaded zones + speech-bubble
callouts with arrows. Matches the user's blog aesthetic at
https://bledden.github.io/blog/*.

Palette & decoder-family color mapping: see _style.py.

Data sources (all under bench/results/):
  comprehensive_eval.json
  h200_lange_headtohead_{low,high}_p.json
  h200_main/phase2/ensemble_results_final.json
  h200_main/tuned/ensemble_results_tuned.json
  h200_main/tierC1/ensemble_pfwl3s_full.json
  h200_main/tierC1/ensemble_pfwl3s_d9.json
  h200_main/tierC1/lange_finetuned_eval_d7.json
  h200_main/tierC1/triton_h384_stability.json
  h200_main/hybrid_d7_3seed/hybrid_eval_d7.json
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

from _style import apply, PAL, STATUS, thin_spine, footer, callout

apply()

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT  = HERE

def _save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"))
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    plt.close(fig)
    print(f"  wrote figures/{name}.{{png,pdf}}")

def _pct(x, _=None):
    if x == 0: return "0%"
    if x >= 1: return f"{x:g}%"
    if x >= 0.1: return f"{x:.1f}%"
    if x >= 0.01: return f"{x:.2f}%"
    return f"{x:.3f}%"

from matplotlib.ticker import NullFormatter, LogLocator

def _pct_log_axis(ax, minor_subs=(2, 3, 5)):
    """Force a log-y axis to show percentage labels at major decades AND at
    selected sub-decade ticks (2x, 3x, 5x) — necessary when the visible range
    spans less than one decade, where matplotlib's default LogFormatter only
    labels the bounding decades.
    """
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    if minor_subs:
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=minor_subs, numticks=12))
        ax.yaxis.set_minor_formatter(FuncFormatter(_pct))
        ax.tick_params(axis="y", which="minor", labelsize=9, pad=4)
    else:
        ax.yaxis.set_minor_formatter(NullFormatter())

# -- Load data -----------------------------------------------------------------
J = lambda p: json.load(open(os.path.join(REPO, p)))

comp        = J("bench/results/comprehensive_eval.json")["results"]
h2h         = {**J("bench/results/h200_lange_headtohead_low_p.json"),
               **J("bench/results/h200_lange_headtohead_high_p.json")}
phase2      = J("bench/results/h200_main/phase2/ensemble_results_final.json")
tuned       = J("bench/results/h200_main/tuned/ensemble_results_tuned.json")
pfw_full    = J("bench/results/h200_main/tierC1/ensemble_pfwl3s_full.json")
pfw_v2      = J("bench/results/h200_main/tierC1/ensemble_pfwl3s_v2.json")   # Table 11 source: d3/d5 re-eval (d7 rows byte-identical to pfw_full)
pfw_d9      = J("bench/results/h200_main/tierC1/ensemble_pfwl3s_d9.json")
final_eval  = J("bench/results/final_eval.json")                            # Table 1 d=3/d=5 source
clean_d7    = {r["p"]: r for r in J("bench/results/h200_main/clean_d7_eval.json")["rows"]}  # Table 1 d=7: leak-free single d7_p015
lange_ft    = J("bench/results/h200_main/tierC1/lange_finetuned_eval_d7.json")
triton_h384 = J("bench/results/h200_main/tierC1/triton_h384_stability.json")
hybrid      = J("bench/results/h200_main/hybrid_d7_3seed/hybrid_eval_d7.json")

ens_final = {}
for k, v in phase2.items():
    d = int(k.split("_")[0][1:])
    ens_final[k] = tuned[k] if (d == 7 and k in tuned) else v


# ============================================================================
# F1 — HERO: Two named systems at d=7 — insight title, hero color emphasis
# ============================================================================
def fig01_hero():
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    ps = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015]
    ps_op = [0.005, 0.007, 0.010, 0.015]

    pm    = [h2h[f"d7_p{p}"]["pm_ler"]    * 100 for p in ps]
    la    = [h2h[f"d7_p{p}"]["lange_ler"] * 100 for p in ps]
    pfw   = []
    for p in ps:
        k = f"d7_p{p}"
        r = pfw_full[k] if k in pfw_full else h2h[k]
        pfw.append(r["pf_ler"]*100)
    tri    = [pfw_full[f"d7_p{p}"]["majority_ler"]   *100 for p in ps_op]
    tri_lo = [pfw_full[f"d7_p{p}"]["majority_ci"][0] *100 for p in ps_op]
    tri_hi = [pfw_full[f"d7_p{p}"]["majority_ci"][1] *100 for p in ps_op]

    # Operational-regime background shading, with the label up at the TOP of the chart
    ax.axvspan(0.005, 0.015, color=PAL["win_band"], alpha=0.55, zorder=0)
    ax.text(0.0087, 50, "operational regime", fontsize=11,
            color="#92400E", style="italic", weight="bold", ha="center")

    # Non-hero series — lighter weight + smaller markers
    ax.plot(ps, pm,  "o-",  color=PAL["pm"],     linewidth=2.0, markersize=7,
            label="PyMatching (baseline)")
    ax.plot(ps, la,  "D-",  color=PAL["lange"],  linewidth=2.0, markersize=7,
            label="Lange GNN (prior art)")
    ax.plot(ps, pfw, "s-",  color=PAL["pf"],     linewidth=2.0, markersize=8,
            label="PFWL3S (this work)")

    # HERO: thicker line, larger markers, CI band
    ax.fill_between(ps_op, tri_lo, tri_hi, color=PAL["triad"], alpha=0.20, linewidth=0)
    ax.plot(ps_op, tri, "o-", color=PAL["triad"], linewidth=3.5, markersize=12,
            markeredgecolor="white", markeredgewidth=2.0,
            label="Pathfinder-Triad (HERO)", zorder=10)

    # Value labels only at the headline rate (p=0.007) and worst rate (p=0.015) — clean
    # White-rounded background so the label is readable on top of the operational-regime band
    for x, y in [(0.007, tri[1]), (0.015, tri[3])]:
        ax.annotate(f"{y:.2f}%", (x, y), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=11, weight="bold", color=PAL["triad"],
                    bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                              edgecolor=PAL["triad"], linewidth=0.8, alpha=0.95))

    # Callout in the lower-LEFT area (well clear of the curves)
    ax.annotate("Pathfinder-Triad strict-CI\nbeats Lange (0.372 pp gap)",
                xy=(0.007, 2.39),
                xytext=(0.00125, 0.32),
                fontsize=11, weight="bold", color=PAL["triad"], ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=PAL["triad"], linewidth=1.2, alpha=0.97),
                arrowprops=dict(arrowstyle="-|>", color=PAL["triad"], lw=1.4,
                                connectionstyle="arc3,rad=0.18"))

    # Honesty callout: PM competitive at the saturated regime (p=0.015)
    # At p=0.015 d=7: PM 27.16% vs PFWL3S 27.33% — a 0.17 pp point-estimate
    # edge with OVERLAPPING CIs (a statistical tie, not a strict win)
    ax.annotate("Above-threshold regime:\nPM edges PFWL3S by 0.17 pp here\n(point estimate; CIs overlap — §6.3)",
                xy=(0.015, 27.16),
                xytext=(0.0028, 46),
                fontsize=10, weight="bold", color=PAL["pm"], ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=PAL["pm"], linewidth=1.0, alpha=0.97),
                arrowprops=dict(arrowstyle="-|>", color=PAL["pm"], lw=1.1,
                                connectionstyle="arc3,rad=-0.20"))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Physical error rate  p", fontsize=12.5)
    ax.set_ylabel("Logical error rate", fontsize=12.5)
    ax.set_xticks(ps); ax.set_xticklabels([f"{p:g}" for p in ps])
    ax.set_ylim(0.005, 100)        # tighten the range so the Triad line stays prominent
    _pct_log_axis(ax, minor_subs=None)   # range spans 4 decades — major ticks only
    ax.set_title("Pathfinder-Triad beats every individual decoder at d=7 operational rates",
                 fontsize=14.5, weight="bold", pad=14, loc="left")
    ax.grid(True, which="both", alpha=0.6)
    ax.legend(loc="lower right", fontsize=11)
    thin_spine(ax)
    footer(ax, "d=7 rotated surface code · 100K shots / point · 95% Wilson CIs shaded for the hero series · "
                "Triad = PFWL3S + Lange + PyMatching majority vote.")
    fig.tight_layout()
    _save(fig, "fig01_hero_d7")

# ============================================================================
# F2 — Pathfinder vs PM 3-param, small-multiples d=3,5,7
# ============================================================================
def fig02_3param_multid():
    ps = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015]
    N = 100000
    from math import sqrt
    def _wilson(f, n, z=1.96):
        pr = f / n; den = 1 + z * z / n
        c = (pr + z * z / (2 * n)) / den
        h = z * sqrt(pr * (1 - pr) / n + z * z / (4 * n * n)) / den
        return c - h, c + h
    def _table1(d, p):
        """EXACTLY Table 1's sources: final_eval.json for d=3/5; the leak-free
        clean_d7_eval.json (single d7_p015, val-selected/test-reported) for d=7.
        Returns (pf_ler, pm_ler, uf_ler) as fractions."""
        fe = final_eval.get(f"rotZ_d{d}_p{p:g}", {})
        if d == 7:
            r = clean_d7[p]
            pf = r["all_test_ler"]["d7_p015"]
            pm = r["pm_test_ler"]
            if pm is None:      # PM not re-run at the two lowest rates; Table 1 prints 0.000
                pm = 0.0
            return pf, pm, fe.get("uf", comp.get(f"d{d}_p{p:g}", {}).get("uf_ler", float("nan")))
        return fe["neural"], fe["pm"], fe.get("uf", float("nan"))
    # Vertical 3x1 stack: each d-panel spans the FULL column width (the side-by-side
    # 1x3 layout forced each panel to ~1/3 width once scaled to 0.92\linewidth).
    fig, axs = plt.subplots(3, 1, figsize=(9.0, 11.5), sharex=True)
    win_count = {3: 0, 5: 0, 7: 0}
    tie_count = {3: 0, 5: 0, 7: 0}
    loss_count = {3: 0, 5: 0, 7: 0}
    for ax, d in zip(axs, [3, 5, 7]):
        vals   = [_table1(d, p) for p in ps]
        nl     = np.array([v[0] * 100 for v in vals])
        pm_    = np.array([v[1] * 100 for v in vals])
        uf_    = np.array([v[2] * 100 for v in vals])
        ax.plot(ps, pm_, "o-", color=PAL["pm"], linewidth=2.6, markersize=9, label="PyMatching")
        ax.plot(ps, nl,  "s-", color=PAL["pf"], linewidth=3.0, markersize=10, label="Pathfinder")
        ax.plot(ps, uf_, "^--", color=PAL["uf"], linewidth=2.0, alpha=0.85, label="Union-Find")
        # Per-cell verdict markers — 95% Wilson CIs recomputed from failure
        # counts, the SAME strict-CI test as Table 1's text. Markers sit on a
        # fixed row near the panel floor (axes coords) so zero-LER cells can
        # never render outside the axes; glyphs are forced onto DejaVu Sans,
        # which actually contains ✓/✗ (the paper font substitutes tofu).
        from matplotlib.transforms import blended_transform_factory
        tform = blended_transform_factory(ax.transData, ax.transAxes)
        for p, y, py in zip(ps, nl, pm_):
            f_pf, f_pm = round(y / 100 * N), round(py / 100 * N)
            lo_pf, hi_pf = _wilson(f_pf, N)
            lo_pm, hi_pm = _wilson(f_pm, N)
            if f_pf == 0 and f_pm == 0:
                sym, color = "≈", "#888"; tie_count[d] += 1       # zero observed errors
            elif hi_pf < lo_pm:
                sym, color = "✓", "#16A34A"; win_count[d] += 1   # PF strict-wins
            elif hi_pm < lo_pf:
                sym, color = "✗", "#7C3AED"; loss_count[d] += 1  # PF strict-loses
            else:
                sym, color = "≈", "#888"; tie_count[d] += 1       # overlap
            ax.text(p, 0.045, sym, transform=tform, ha="center", va="bottom",
                    fontsize=15, color=color, weight="bold",
                    fontfamily="DejaVu Sans", clip_on=True)
        ax.set_xscale("log"); ax.set_yscale("log")
        verdict = f"{win_count[d]}W / {tie_count[d]}T / {loss_count[d]}L vs PM"
        ax.set_title(f"d = {d}   ({verdict})", fontsize=15.5, weight="bold", pad=8)
        ax.grid(True, which="both", alpha=0.55)
        ax.set_xticks(ps)
        ax.set_ylabel("Logical error rate", fontsize=13)
        ax.tick_params(axis="y", labelsize=11.5)
        thin_spine(ax)
        _pct_log_axis(ax, minor_subs=None)
        if d == 3:
            ax.legend(loc="lower right", fontsize=12.5)
            # In-axes legend for the verdict markers
            ax.text(0.02, 0.98,
                    "✓ Pathfinder strict-CI wins    ≈ overlap (tie)    ✗ PM strict-CI wins",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=11.5, color="#374151", fontfamily="DejaVu Sans",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                              edgecolor="#D1D5DB", linewidth=0.6, alpha=0.95))
    axs[-1].set_xlabel("Physical error rate  p", fontsize=13)
    axs[-1].set_xticklabels([f"{p:g}" for p in ps], rotation=35, fontsize=12)
    total_w = sum(win_count.values()); total_t = sum(tie_count.values()); total_l = sum(loss_count.values())
    loss_word = "loss" if total_l == 1 else "losses"
    fig.suptitle(
        f"Pathfinder vs PyMatching at d $\\in$ {{3,5,7}}: {total_w} strict-CI wins / {total_t} statistical ties / "
        f"{total_l} strict-CI {loss_word} (3-param noise, 100K shots/point)",
        fontsize=16, weight="bold", y=0.997, x=0.5, ha="center")
    fig.text(0.5, 0.012,
             "Strict-CI test: 95% Wilson intervals recomputed from failure counts (N=100K) — the same test as Table 1.\n"
             "d=7 uses the leak-free single d7_p015 checkpoint (val-selected, test-reported; clean_d7_eval.json).\n"
             "Pathfinder is never strictly beaten; the closest cell is d=7 p=0.002, a one-failure difference (PF 5, PM 4).",
             ha="center", va="bottom", fontsize=11, style="italic", color="#6B7280")
    fig.tight_layout(rect=[0, 0.055, 1, 0.975])
    _save(fig, "fig02_3param_multid")

# ============================================================================
# F3 — Pareto: accuracy vs latency at d=7, p=0.007
# ============================================================================
def fig03_pareto():
    # Open-source decoders measured here on H200 SXM / Apple M4 — apples-to-apples comparable
    P_main = [
        (6.12,  1.041, "Pathfinder+Triton (3-param)",                PAL["pf"],         "o", 150),
        (6.12,  2.492, "PFWL3S (4-param, HERO)",                     PAL["pfwl3s"],     "s", 200),
        (20.4,  2.78,  "PFWL3S single-seed (H=384)",                 PAL["pf_kd"],      "D", 130),
        (9.65,  1.489, "PyMatching (3-param, M4 CPU)",               PAL["pm"],         "o", 140),
        (9.65,  3.366, "PyMatching (4-param)",                       PAL["pm_alt"],     "^", 130),
        (71.67, 2.956, "Lange GNN (published)",                      PAL["lange"],      "D", 160),
        (71.67, 2.739, "Lange GNN (fine-tuned)",                     PAL["lange_ft"],   "v", 150),
        (72.0,  2.384, "Pathfinder-Triad (HERO)",                    PAL["triad"],      "o", 330),
        (72.0,  2.326, "Triad w/ Lange-FT",                          PAL["triad_kd"],   "X", 190),
    ]
    # Closed-source / non-matched comparators — indicative only, different hardware OR noise model
    P_indicative = [
        (63.0,  2.14,  "AlphaQubit (TPU, Sycamore real-hw)",         PAL["alpha_qubit"],"p", 130),
        (40.0,  1.0,   "Gu et al. (Gross codes, not surface)",       PAL["gu_etal"],    "h", 130),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 7.2),
                                  gridspec_kw=dict(width_ratios=[3.4, 1.0]),
                                  sharey=True)
    P = P_main
    # Cycle-budget shaded region
    ax.axvspan(3, 7, color=PAL["budget_band"], alpha=0.9, zorder=0)
    ax.axvline(7.0, color="#15803D", ls=":", lw=1.4, zorder=1)
    ax.text(7.4, 2.5, "7 μs / syndrome service-rate target", fontsize=11, color="#15803D",
            ha="left", va="center", rotation=90, style="italic", weight="bold")
    ax.text(4.7, 0.78, "below service-rate target", fontsize=11, color="#15803D",
            style="italic", ha="center", weight="bold")

    # Pareto frontier (dotted)
    P_sorted = sorted(P, key=lambda r: r[0])
    front, best = [], float("inf")
    for lat, ler, *_ in P_sorted:
        if ler < best:
            best = ler; front.append((lat, ler))
    ax.plot([p[0] for p in front], [p[1] for p in front],
            color="#9CA3AF", lw=1.6, ls=":", zorder=1)

    # Manual offset table to keep numbered labels from overlapping
    offset = {
        1: (1.18, 0.97), 2: (1.18, 0.97), 3: (1.20, 0.98), 4: (1.16, 0.97),
        5: (1.16, 0.97), 6: (1.14, 1.03), 7: (1.14, 0.90), 8: (0.78, 1.05),
        9: (0.78, 0.90),
    }
    handles = []
    for i, (lat, ler, lbl, color, marker, size) in enumerate(P, 1):
        h = ax.scatter(lat, ler, s=size, c=color, marker=marker,
                       edgecolors="#1F2937", linewidths=1.0, zorder=4)
        handles.append(h)
        dx, dy = offset.get(i, (1.18, 0.985))
        ax.text(lat * dx, ler * dy, str(i), fontsize=11, fontweight="bold",
                color=color, ha="left", va="center", zorder=5)

    leg_labels = [f"({i}) {p[2]}" for i, p in enumerate(P, 1)]
    ax.legend(handles, leg_labels, loc="lower right",
              title="Open-source decoders (measured here)", title_fontsize=10.5,
              frameon=True, fontsize=9.5, ncol=2,
              labelspacing=1.0, columnspacing=1.4, handletextpad=0.7, borderpad=0.9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Latency  (μs / syndrome, throughput-optimal batch)", fontsize=12.5)
    ax.set_ylabel("Logical error rate at d=7, p=0.007", fontsize=12.5)
    ax.set_xlim(3, 200); ax.set_ylim(0.6, 5)
    ax.set_xticks([3,5,10,20,50,100,200]); ax.set_xticklabels(['3','5','10','20','50','100','200'])
    _pct_log_axis(ax)
    ax.set_title("Open-source decoders on matched hardware (H200 SXM / Apple M4)",
                 fontsize=12.5, weight="bold", pad=10, loc="left")
    ax.grid(True, which="both", alpha=0.55)
    thin_spine(ax)

    # Second panel: closed-source / non-matched comparators (indicative only)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlim(20, 200); ax2.set_ylim(0.6, 5)
    ax2.set_xticks([20,50,100,200]); ax2.set_xticklabels(['20','50','100','200'])
    ax2.set_xlabel("Latency  (μs / syn)", fontsize=12)
    ax2.grid(True, which="both", alpha=0.55)
    ax2.set_facecolor("#FAFAFA")
    _pct_log_axis(ax2)
    for i, (lat, ler, lbl, color, marker, size) in enumerate(P_indicative, 10):
        ax2.scatter(lat, ler, s=size, c=color, marker=marker,
                    edgecolors="#1F2937", linewidths=1.0, zorder=4)
        ax2.text(lat * 1.18, ler * 0.98, str(i), fontsize=11, fontweight="bold",
                 color=color, ha="left", va="center")
        # Inline labels in the small panel
        ax2.text(lat, ler * 1.45, lbl.split(" (")[0], ha="center", fontsize=9,
                 color=color, weight="bold")
    ax2.set_title("Indicative only\n(non-matched noise/hardware)",
                  fontsize=11, weight="bold", pad=10, loc="left", color="#6B7280")
    thin_spine(ax2)

    fig.suptitle("Pathfinder+Triton is the only decoder below the 7 μs/syndrome service-rate target (batched)",
                 fontsize=14.5, weight="bold", y=1.02, x=0.04, ha="left")
    fig.text(0.04, -0.04,
             "Latencies at throughput-optimal batch on each decoder's reported hardware. "
             "Dotted line = Pareto frontier (lowest LER for each latency). "
             "Right panel: AlphaQubit (TPU, real Sycamore noise) and Gu et al. (Gross codes, not surface) "
             "are reported on different hardware OR a different problem — shown for context, not strictly comparable.",
             fontsize=10.5, style="italic", color="#6B7280", wrap=True)
    fig.tight_layout()
    _save(fig, "fig03_pareto_d7")

# ============================================================================
# F4 — Triton vs reference at H=384 (audit M10 finding)
# ============================================================================
def fig04_triton_vs_ref():
    L = triton_h384["latency"]
    batches = ["B1", "B64", "B1024"]
    ref = [L[b]["ref_us_per_syn"] for b in batches]
    tri = [L[b]["tri_us_per_syn"] for b in batches]
    x = np.arange(len(batches))
    w = 0.34
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    bars_ref = ax.bar(x - w/2, ref, w, color=PAL["pf"],    edgecolor="#1F2937",
                      linewidth=1.0, label="Reference (PyTorch)")
    bars_tri = ax.bar(x + w/2, tri, w, color=PAL["triad"], edgecolor="#1F2937",
                      linewidth=1.0, label="Triton kernel")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(["B = 1", "B = 64", "B = 1024"], fontsize=12)
    ax.set_ylabel("Latency  (μs / syndrome, log scale)", fontsize=12.5)
    ax.grid(True, which="both", axis="y", alpha=0.55)
    # Value labels
    for bars, vals in [(bars_ref, ref), (bars_tri, tri)]:
        for b, v in zip(bars, vals):
            label = f"{v:.1f}" if v < 100 else f"{v:.0f}"
            ax.text(b.get_x()+b.get_width()/2, v*1.10, label,
                    ha="center", fontsize=10.5, weight="bold", color="#1F2937")
    # Speedup labels above each batch group
    for i, b in enumerate(batches):
        sp = L[b]["speedup_x"]
        color = "#16A34A" if sp >= 1 else "#7C3AED"
        winner = "Triton wins" if sp >= 1 else "ref wins"
        ratio = sp if sp >= 1 else 1/sp
        ymax = max(ref[i], tri[i])
        ax.text(i, ymax * 1.4, f"{ratio:.2f}x\n({winner})",
                ha="center", va="bottom", fontsize=11.5, color=color, weight="bold")
    # Cycle budget line
    ax.axhline(7.0, color="#15803D", ls=":", lw=1.4)
    ax.text(2.45, 8.0, "7 μs / syndrome service-rate target", color="#15803D",
            ha="right", va="bottom", fontsize=10.5, style="italic", weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#15803D", linewidth=0.8, alpha=0.92))
    ax.set_ylim(0.5, 9000)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_title("Triton kernel tuned for H=256 LOSES to PyTorch reference at H=384 B=1024 (M10 audit)",
                 fontsize=14.5, weight="bold", pad=14, loc="left")
    thin_spine(ax)
    footer(ax, "Pathfinder-Wide-Long (H=384), d=7, p=0.007 on NVIDIA H200 SXM. "
                "Triton kernel block-tile sizes were chosen for H=256 (1.78x speedup at B=1024) and "
                "are inefficient at H=384.")
    fig.tight_layout()
    _save(fig, "fig04_triton_h384")

# ============================================================================
# F5 — Decoder failure overlap (Venn diagram for set overlap)
# ============================================================================
def fig05_failure_overlap():
    r = tuned["d7_p0.007"]  # canonical fine-tune (A2 fix: was phase2/KD-era)
    n = r["n"]
    both    = r["both_wrong_pf_lange"]
    pf_only = r["pf_wrong_lange_right"]
    la_only = r["pf_right_lange_wrong"]
    neither = n - both - pf_only - la_only

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    R = 1.3
    cx_pf, cx_la, cy = -0.72, 0.72, 0.0
    ax.add_patch(plt.Circle((cx_pf, cy), R, color=PAL["pf"],    alpha=0.40,
                            linewidth=2.5, edgecolor=PAL["pf"]))
    ax.add_patch(plt.Circle((cx_la, cy), R, color=PAL["lange"], alpha=0.40,
                            linewidth=2.5, edgecolor=PAL["lange"]))

    pf_only_pct = pf_only/n*100
    la_only_pct = la_only/n*100
    both_pct    = both/n*100

    # Headers above each circle
    ax.text(-0.72, 1.55, "Pathfinder\nwrong", ha="center", va="center",
            fontsize=13.5, weight="bold", color=PAL["pf"])
    ax.text( 0.72, 1.55, "Lange GNN\nwrong", ha="center", va="center",
            fontsize=13.5, weight="bold", color=PAL["lange"])
    # Count + pct in each region
    ax.text(-1.15, cy - 0.10, f"only-PF\n{pf_only:,} shots\n{pf_only_pct:.3f}%",
            ha="center", va="center", fontsize=11.5, color="#1F2937")
    ax.text( 1.15, cy - 0.10, f"only-Lange\n{la_only:,} shots\n{la_only_pct:.3f}%",
            ha="center", va="center", fontsize=11.5, color="#1F2937")
    ax.text(  0.0, cy - 0.10, f"BOTH wrong\n{both:,} shots\n{both_pct:.3f}%",
            ha="center", va="center", fontsize=11.5, color="white", weight="bold")
    # Background line
    ax.text(0.0, -2.15, f"Both correct on {neither:,} of {n:,} shots ({neither/n*100:.2f}%)",
            ha="center", va="top", fontsize=11, color="#4B5563")

    ax.set_xlim(-3.1, 3.1); ax.set_ylim(-2.7, 2.7)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top","right","left","bottom"): ax.spines[s].set_visible(False)
    ax.set_title(f"Pathfinder and Lange fail on largely disjoint syndromes "
                 f"(only {both_pct:.2f}% shot overlap)",
                 fontsize=14.5, weight="bold", pad=14, loc="center")
    ax.text(0.5, -0.02, "This near-disjoint failure mode is the structural reason "
            "the §5.12 three-way majority vote (Pathfinder-Triad) beats every individual decoder.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, style="italic", color="#4B5563")
    fig.tight_layout()
    _save(fig, "fig05_failure_overlap")

# ============================================================================
# F6 — PFWL3S vs Lange (published vs fine-tuned) at d=7
# ============================================================================
def fig06_lange_ft():
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ps = [0.005, 0.007, 0.010, 0.015]
    pfw    = [lange_ft["rates"][f"p{p}"]["pf_ler"]    *100 for p in ps]
    pub    = [lange_ft["rates"][f"p{p}"]["lange_pub_ler"]    *100 for p in ps]
    ftL    = [lange_ft["rates"][f"p{p}"]["lange_ft_ler"]    *100 for p in ps]
    pm_    = [lange_ft["rates"][f"p{p}"]["pm_ler"]    *100 for p in ps]

    ax.plot(ps, pm_,  "o-",  color=PAL["pm"],       linewidth=2.0, markersize=7, label="PyMatching")
    ax.plot(ps, pub,  "D-",  color=PAL["lange"],    linewidth=2.4, markersize=8, label="Lange GNN (published)")
    ax.plot(ps, ftL,  "v--", color=PAL["lange_ft"], linewidth=2.4, markersize=8, label="Lange GNN (fine-tuned @ p=0.007)")
    ax.plot(ps, pfw,  "s-",  color=PAL["pfwl3s"],   linewidth=3.2, markersize=10,
            markeredgecolor="white", markeredgewidth=1.6, label="PFWL3S (this work, HERO)", zorder=10)

    # Label only the headline-rate (p=0.007) PFWL3S point to anchor the strict-CI claim
    idx = ps.index(0.007)
    ax.annotate(f"PFWL3S {pfw[idx]:.2f}%", (0.007, pfw[idx]),
                xytext=(0.0089, 1.55), textcoords="data",
                ha="left", va="center", fontsize=10.5, weight="bold", color=PAL["pfwl3s"],
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=PAL["pfwl3s"], linewidth=0.8, alpha=0.97),
                arrowprops=dict(arrowstyle="-", color=PAL["pfwl3s"], lw=0.8))
    ax.annotate(f"Lange-FT {ftL[idx]:.2f}%", (0.007, ftL[idx]),
                xytext=(0.0089, 5.5), textcoords="data",
                ha="left", va="center", fontsize=10.5, weight="bold", color=PAL["lange_ft"],
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=PAL["lange_ft"], linewidth=0.8, alpha=0.97),
                arrowprops=dict(arrowstyle="-", color=PAL["lange_ft"], lw=0.8))

    # Honesty callout at p=0.015 — computed from the plotted data, and labeled
    # as the point-estimate edge it is (the CIs overlap: a statistical tie)
    idx15 = ps.index(0.015)
    ax.annotate(
        f"At p=0.015: PM {pm_[idx15]:.2f}% edges PFWL3S {pfw[idx15]:.2f}% by {pfw[idx15]-pm_[idx15]:.2f} pp\n(point estimate; CIs overlap — statistical tie; above-threshold regime, §6.3)",
        xy=(0.015, pm_[idx15]),
        xytext=(0.0050, 60), textcoords="data",
        ha="left", va="center", fontsize=10, weight="bold", color=PAL["pm"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor=PAL["pm"], linewidth=1.0, alpha=0.97),
        arrowprops=dict(arrowstyle="-|>", color=PAL["pm"], lw=1.0,
                        connectionstyle="arc3,rad=0.30"))

    # Callout placed in the upper-left empty region (high above the curves)
    callout(ax, 0.007, 2.74,
            "PFWL3S strict-CI beats\nfine-tuned Lange\n(0.049 pp gap at p=0.007)",
            dx=0.0, dy=15.0, color=PAL["pfwl3s"])

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Physical error rate  p", fontsize=12.5)
    ax.set_ylabel("Logical error rate", fontsize=12.5)
    ax.set_xticks(ps); ax.set_xticklabels([f"{p:g}" for p in ps])
    ax.set_xlim(0.0042, 0.020)
    ax.set_ylim(0.4, 100)
    _pct_log_axis(ax)
    ax.set_title("PFWL3S strictly beats Lange even after Lange is fine-tuned at p=0.007 (C2 audit)",
                 fontsize=14.5, weight="bold", pad=14, loc="left")
    ax.grid(True, which="both", alpha=0.55)
    ax.legend(loc="lower right", fontsize=11)
    thin_spine(ax)
    footer(ax, "100K shots / point. Lange fine-tune: 30 epochs at p=0.007, Adam lr=1e-4, "
                "resume from Lange's published d7 ckpt.")
    fig.tight_layout()
    _save(fig, "fig06_lange_ft")

# ============================================================================
# F7 — Strict-CI dominance heatmap with CI-gap magnitudes in percentage points (pp)
# ============================================================================
def fig07_dominance_heatmap():
    rows = []
    for d in [3, 5, 7]:
        for p in [0.005, 0.007, 0.010, 0.015]:
            rows.append((d, p))
    for p in [0.005, 0.007, 0.010, 0.015]:
        rows.append((9, p))
    cols = [
        "PFWL3S\nvs Lange-pub",
        "PFWL3S\nvs Lange-FT",
        "Triad\nvs Lange-pub",
        "Triad\nvs PyMatching",
    ]
    def status_and_gap(a_lo, a_hi, b_lo, b_hi):
        if a_hi < b_lo: return "win",  b_lo - a_hi
        if b_hi < a_lo: return "loss", a_lo - b_hi
        return "tie", 0.0
    def fetch(d, p):
        kd = f"d{d}_p{p}"
        # PFWL3S rows come ONLY from the PFWL3S evals: pfw_d9 for d=9 (the
        # H256-d9 variant, Table 14) and the v2 re-eval for d<=7 (Table 11's
        # source; the pre-v2 pfw_full carried stale d3/d5 rows that inverted
        # the d=5 win/loss direction). No silent fallback to canonical-PF
        # (h2h) data — a missing key is a bug, fail loudly.
        if d == 9 and kd in pfw_d9:        r = pfw_d9[kd]
        elif kd in pfw_v2:                 r = pfw_v2[kd]
        else:                              raise KeyError(f"fig07: no PFWL3S data for {kd}")
        ft_l, ft_ci = None, None
        if d == 7 and f"p{p}" in lange_ft["rates"]:
            ft_l  = lange_ft["rates"][f"p{p}"]["lange_ft_ler"]
            ft_ci = lange_ft["rates"][f"p{p}"]["lange_ft_ci"]
        return r, (ft_l, ft_ci)

    cells = []
    for (d, p) in rows:
        r, (ft_l, ft_ci) = fetch(d, p)
        out = []
        out.append(status_and_gap(r["pf_ci"][0], r["pf_ci"][1],
                                  r["lange_ci"][0], r["lange_ci"][1]))
        if ft_l is None:
            out.append(("na", 0.0))
        else:
            out.append(status_and_gap(r["pf_ci"][0], r["pf_ci"][1],
                                      ft_ci[0], ft_ci[1]))
        if "majority_ci" not in r:
            out.append(("na", 0.0)); out.append(("na", 0.0))
        else:
            out.append(status_and_gap(r["majority_ci"][0], r["majority_ci"][1],
                                      r["lange_ci"][0], r["lange_ci"][1]))
            out.append(status_and_gap(r["majority_ci"][0], r["majority_ci"][1],
                                      r["pm_ci"][0], r["pm_ci"][1]))
        cells.append(out)

    fig, ax = plt.subplots(figsize=(9.0, 9.0))
    for i, row in enumerate(cells):
        y = len(cells)-1-i
        for j, (status, gap) in enumerate(row):
            color = STATUS[status]
            ax.add_patch(plt.Rectangle((j, y), 1, 1,
                                        facecolor=color,
                                        edgecolor="white", linewidth=2.0))
            if status == "win":
                # show a third decimal for razor-thin margins so a marginal
                # non-overlap never renders as a self-contradictory "+0.00 pp"
                fmt = ".3f" if gap * 100 < 0.005 else ".2f"
                lbl = f"WIN\n+{gap*100:{fmt}} pp";  tc = "white"; weight = "bold"
            elif status == "loss":
                fmt = ".3f" if gap * 100 < 0.005 else ".2f"
                lbl = f"loss\n-{gap*100:{fmt}} pp"; tc = "white"; weight = "bold"
            elif status == "tie":
                lbl = "overlap"; tc = "#4B5563"; weight = "normal"
            else:
                lbl = "—"; tc = "#9CA3AF"; weight = "normal"
            ax.text(j+0.5, y+0.5, lbl, ha="center", va="center",
                    fontsize=10.5, color=tc, weight=weight)

    ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(rows))
    ax.set_xticks(np.arange(len(cols))+0.5); ax.set_xticklabels(cols, fontsize=12, weight="bold")
    ax.set_yticks(np.arange(len(rows))+0.5)
    ax.set_yticklabels([f"d={d}, p={p}" for (d, p) in reversed(rows)], fontsize=12)
    for s in ("top","right","left","bottom"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    handles = [
        mpatches.Patch(facecolor=STATUS["win"],  label="strict-CI WIN  (95% non-overlap)"),
        mpatches.Patch(facecolor=STATUS["tie"],  label="overlap (statistical tie)"),
        mpatches.Patch(facecolor=STATUS["loss"], label="strict-CI loss"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.06),
              ncol=3, frameon=False, fontsize=11)
    # Title tallies are computed from the rendered cells — never hardcoded
    # (an earlier draft shipped a stale "5 of 8" against a 7-of-8 grid).
    d7plus = [i for i, (d, p) in enumerate(rows) if d >= 7]
    tvl_wins = sum(1 for i in d7plus if cells[i][2][0] == "win")
    ax.set_title(f"Triad strict-CI beats Lange at {tvl_wins} of {len(d7plus)} d ≥ 7 ops points;\n"
                 "loses to PM at d=9 p=0.015 (above threshold)",
                 fontsize=15, weight="bold", pad=14, loc="left")
    footer(ax, "Wilson 95% CIs at 100K shots / point. WIN = row-decoder strictly beats column-decoder. "
                "Numbers show CI-EDGE separation in percentage points (pp; point-estimate gaps are larger). "
                "PFWL3S rows: d≤7 = the Table-11 re-eval (ensemble_pfwl3s_v2.json); d=9 = the PFWL3S-H256-d9 "
                "variant (Table 14). The 'loss' cell at d=9 p=0.015 in the right column reflects that the d=9 "
                "surface code is above its pseudo-threshold there, where PM's combinatorial structure is provably "
                "near-optimal.")
    fig.tight_layout()
    _save(fig, "fig07_dominance_heatmap")

# ============================================================================
# F8 — Triad-distillation arc (bars vs Triad baseline reference line)
# ============================================================================
def fig08_triad_distill():
    recipes = [
        ("Original\nPFWL3S",                 2.492),
        ("Soft Triad-distill\nfrom-scratch", 2.71),
        ("Hardlabel Triad-distill\nfrom-scratch", 2.71),
        ("Warm-init Triad-distill\n3-seed avg",   2.507),
        ("H=512 Triad-distill\n3-seed avg",       2.558),
        ("PF+PM-only KD\nwarm-init",              2.57),
    ]
    triad_baseline = 2.384
    labels = [r[0] for r in recipes]
    vals   = [r[1] for r in recipes]
    x = np.arange(len(recipes))
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    bars = ax.bar(x, vals, width=0.65,
                  color=[PAL["pfwl3s"]] + [PAL["pf_kd"]]*5,
                  edgecolor="#1F2937", linewidth=1.0)
    ax.axhline(triad_baseline, color=PAL["triad"], ls="--", lw=2.4, zorder=4,
               label=f"Pathfinder-Triad baseline  ({triad_baseline:.3f}%)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("LER at d=7, p=0.007", fontsize=12.5)
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v + 0.02, f"{v:.3f}%",
                ha="center", fontsize=10.5, weight="bold", color="#1F2937")
    ax.set_ylim(2.2, 2.85)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, which="major", axis="y", alpha=0.55)
    ax.set_title("No distilled single PF student beats the Triad — coverage is architectural",
                 fontsize=14.5, weight="bold", pad=14, loc="left")
    thin_spine(ax)
    footer(ax, "Six recipe variants ($110 follow-up compute). Bars: 3-seed-avg distilled-student LER. "
                "Dashed line: Pathfinder-Triad baseline.")
    fig.tight_layout()
    _save(fig, "fig08_triad_distill")

# ============================================================================
# F9 — Hybrid CNN+GNN vs PFWL3S — paired-difference plot (canonical for ties)
# ============================================================================
def fig09_hybrid_vs_pfwl3s():
    ps = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015]
    diffs, cis = [], []
    for p in ps:
        r = hybrid["rates"][f"p{p}"]
        n = r["n"]
        pa = r["hybrid_ler"]; pb = r["pfwl3s_ler"]
        d_ler = pa - pb
        var = pa*(1-pa)/n + pb*(1-pb)/n
        hw = 1.96 * np.sqrt(var)
        diffs.append(d_ler * 100)        # pp
        cis.append(hw * 100)             # pp

    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    x = np.arange(len(ps))
    # Zero-line shaded band: "statistically indistinguishable" region
    ax.axhspan(-max(cis), max(cis), color="#F3F4F6", alpha=0.7, zorder=0)
    ax.axhline(0, color="#4B5563", linewidth=1.4, zorder=1)
    ax.errorbar(x, diffs, yerr=cis, fmt="o", color=PAL["hybrid"],
                ecolor="#1F2937", elinewidth=1.4, capsize=6, capthick=1.4,
                markersize=11, markeredgecolor="white", markeredgewidth=1.6,
                zorder=4)
    # Value labels (ΔLER in pp) above each point
    for xi, d, c in zip(x, diffs, cis):
        ax.annotate(f"{d:+.2f}", (xi, d + c + 0.06),
                    ha="center", fontsize=10, weight="bold", color="#1F2937")
    ax.set_xticks(x); ax.set_xticklabels([f"{p:g}" for p in ps], fontsize=11.5)
    ax.set_xlabel("Physical error rate  p", fontsize=12.5)
    ax.set_ylabel("Hybrid − PFWL3S  (LER difference, percentage points)", fontsize=12.5)
    ax.grid(True, axis="y", alpha=0.55)
    ax.set_ylim(min(min(diffs)-max(cis), -0.25) * 1.4, max(max(diffs)+max(cis), 0.25) * 1.4)
    ax.set_title("Hybrid CNN+GNN statistically ties PFWL3S at all 8 noise rates",
                 fontsize=14.5, weight="bold", pad=14, loc="left")
    # Caption-style headline
    ax.text(0.5, -0.20,
            "All 8 differences contain zero (error bars cross the zero line). "
            "The architectural fusion does not measurably help at this scale.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, style="italic", color="#4B5563")
    thin_spine(ax)
    fig.tight_layout()
    _save(fig, "fig09_hybrid_vs_pfwl3s")

# ============================================================================
# F10 — Muon ablation depth dependence
# ============================================================================
def fig10_muon_ablation():
    d_ab = [3, 5, 7]
    full = [1.818, 1.28, 1.041]
    adam = [2.14,  2.20, 34.8]
    x = np.arange(len(d_ab))
    w = 0.34
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    # NOTE: these are the §6.2 ablation run's own LERs (the d=7 Muon bar is the
    # ablation-era canonical model, 1.04% — NOT Table 1's d7_p015 value, 1.071%).
    b1 = ax.bar(x - w/2, full, w, color=PAL["triad"], edgecolor="#1F2937",
                linewidth=1.0, label="Full Muon  (§6.2 ablation run)")
    b2 = ax.bar(x + w/2, adam, w, color=PAL["lange"], edgecolor="#1F2937",
                linewidth=1.0, label="AdamW only  (§6.2)")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"d = {d}" for d in d_ab], fontsize=12.5)
    ax.set_ylabel("LER at p=0.007  (log scale)", fontsize=12.5)
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.grid(True, which="both", axis="y", alpha=0.55)
    # Value labels (bold, in the bar's color)
    for b, v, c in zip(list(b1), full, [PAL["triad"]]*3):
        ax.text(b.get_x()+b.get_width()/2, v*1.12, f"{v:.2f}%",
                ha="center", fontsize=11, weight="bold", color=c)
    for b, v, c in zip(list(b2), adam, [PAL["lange"]]*3):
        ax.text(b.get_x()+b.get_width()/2, v*1.12, f"{v:.2f}%",
                ha="center", fontsize=11, weight="bold", color=c)
    # Relative regression callouts
    for i, d in enumerate(d_ab):
        rel = (adam[i]/full[i] - 1) * 100
        lbl = f"+{rel:.0f}%" if rel < 300 else "catastrophic\nfailure"
        color = "#7C3AED" if rel >= 50 else "#16A34A"
        ymax = max(full[i], adam[i])
        ax.annotate(lbl, xy=(x[i], ymax*2.4), ha="center",
                    fontsize=12, fontweight="bold", color=color)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_ylim(0.5, 200)
    ax.set_title("Muon's effect grows with depth — AdamW-only fails to converge at d=7\n(within the matched 80K-step budget)",
                 fontsize=14.5, weight="bold", pad=14, loc="left")
    thin_spine(ax)
    footer(ax, "80K training steps / configuration, matched budget (AdamW not separately LR-tuned) — a training-choice "
                "comparison, not a tuned-optimizer duel (§4.2/§6.2). AdamW-only d=7 fails to escape its initial plateau.")
    fig.tight_layout()
    _save(fig, "fig10_muon_ablation")

# ============================================================================
# F11 — d=9 Triad extension (grouped bars with CIs)
# ============================================================================
def fig11_d9_triad():
    ps = [0.005, 0.007, 0.010, 0.015]
    pfw    = [pfw_d9[f"d9_p{p}"]["pf_ler"]    *100 for p in ps]
    pfw_lo = [pfw_d9[f"d9_p{p}"]["pf_ci"][0]  *100 for p in ps]
    pfw_hi = [pfw_d9[f"d9_p{p}"]["pf_ci"][1]  *100 for p in ps]
    la     = [pfw_d9[f"d9_p{p}"]["lange_ler"] *100 for p in ps]
    la_lo  = [pfw_d9[f"d9_p{p}"]["lange_ci"][0]*100 for p in ps]
    la_hi  = [pfw_d9[f"d9_p{p}"]["lange_ci"][1]*100 for p in ps]
    pm_    = [pfw_d9[f"d9_p{p}"]["pm_ler"]    *100 for p in ps]
    pm_lo  = [pfw_d9[f"d9_p{p}"]["pm_ci"][0]  *100 for p in ps]
    pm_hi  = [pfw_d9[f"d9_p{p}"]["pm_ci"][1]  *100 for p in ps]
    tri    = [pfw_d9[f"d9_p{p}"]["majority_ler"]   *100 for p in ps]
    tri_lo = [pfw_d9[f"d9_p{p}"]["majority_ci"][0] *100 for p in ps]
    tri_hi = [pfw_d9[f"d9_p{p}"]["majority_ci"][1] *100 for p in ps]
    x = np.arange(len(ps))
    w = 0.20
    fig, ax = plt.subplots(figsize=(12.0, 6.8))
    def bars(offset, vals, lo, hi, color, label, linewidth=1.0):
        err_lo = np.array(vals) - np.array(lo)
        err_hi = np.array(hi)   - np.array(vals)
        ax.bar(x + offset, vals, w, yerr=[err_lo, err_hi], capsize=4,
               color=color, edgecolor="#1F2937", linewidth=linewidth, label=label,
               error_kw=dict(elinewidth=1.2, ecolor="#1F2937"))
    bars(-1.5*w, pm_, pm_lo, pm_hi, PAL["pm"],     "PyMatching")
    bars(-0.5*w, la,  la_lo, la_hi, PAL["lange"],  "Lange GNN")
    bars( 0.5*w, pfw, pfw_lo, pfw_hi, PAL["pfwl3s"], "PFWL3S-H256-d9")
    bars( 1.5*w, tri, tri_lo, tri_hi, PAL["triad"], "Pathfinder-Triad", linewidth=1.6)
    ax.set_xticks(x); ax.set_xticklabels([f"p = {p:g}" for p in ps], fontsize=12)
    ax.set_yscale("log")
    ax.set_ylabel("Logical error rate at d=9  (log scale)", fontsize=12.5)
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.grid(True, which="both", axis="y", alpha=0.55)
    # generous headroom so callouts sit ABOVE the bars in clear space, not over them
    all_hi = pm_hi + la_hi + pfw_hi + tri_hi
    cluster_max = [max(pm_[j], la[j], pfw[j], tri[j]) for j in range(len(ps))]
    ax.set_ylim(top=max(all_hi) * 3.6)
    ax.legend(loc="upper left", ncol=2, fontsize=10, framealpha=0.9)
    # Triad strict-CI wins (p=0.007, p=0.010): callout ABOVE the cluster, short arrow down to the bar
    for j, p in enumerate(ps):
        if tri_hi[j] < la_lo[j] and not (pm_hi[j] < tri_lo[j]):  # skip where PM itself beats Triad (p=0.015)
            ax.annotate("Triad strict-CI\nbeats Lange",
                        xy=(x[j]+1.5*w, tri_hi[j]),
                        xytext=(x[j]+1.5*w, cluster_max[j]*2.3),
                        ha="center", va="bottom", fontsize=9.5, color="#16A34A", weight="bold",
                        arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.3))
    # Honesty: at p=0.015 PM strictly beats Triad (above-threshold regime) — callout above its cluster
    j15 = ps.index(0.015)
    if pm_hi[j15] < tri_lo[j15]:
        ax.annotate(
            f"PM wins here\n(PM {pm_[j15]:.1f}% vs Triad {tri[j15]:.1f}%)\nabove-threshold regime",
            xy=(x[j15]-1.5*w, pm_hi[j15]),
            xytext=(x[j15], cluster_max[j15]*2.0),
            ha="center", va="bottom", fontsize=9.5, color="#7C3AED", weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#7C3AED", linewidth=1.1, alpha=0.97),
            arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.3))
    ax.set_title("Triad strict-CI win extends from d=7 to d=9 at operational rates —\nbut PM wins at p=0.015 (above threshold)",
                 fontsize=12.5, weight="bold", pad=12, loc="left")
    thin_spine(ax)
    footer(ax, "100K shots / point. PFWL3S-H256-d9 loses to Lange individually at every rate, "
                "but the Triad strictly beats Lange at p=0.007 (0.154 pp CI-edge) and p=0.010 (1.831 pp CI-edge); "
                "point-estimate gaps are larger (0.346 / 2.233 pp).")
    fig.tight_layout()
    _save(fig, "fig11_d9_triad")


def main():
    print("Generating Pathfinder paper figures (modern aesthetic, insight-titles)...")
    fig01_hero()
    fig02_3param_multid()
    fig03_pareto()
    fig04_triton_vs_ref()
    fig05_failure_overlap()
    fig06_lange_ft()
    fig07_dominance_heatmap()
    fig08_triad_distill()
    fig09_hybrid_vs_pfwl3s()
    fig10_muon_ablation()
    fig11_d9_triad()
    print("All 11 figures written.")

if __name__ == "__main__":
    main()
