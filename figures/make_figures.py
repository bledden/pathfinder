"""Generate all four paper figures from raw experimental data.

Run from the repo root:
  python3 figures/make_figures.py

Reads:
  bench/results/h200_session3/phase2/ensemble_results_final.json
  bench/results/h200_lange_headtohead_low_p.json
  bench/results/h200_lange_headtohead_high_p.json
  bench/results/h200_session3/phase2/lange_latency.json

Writes figures/fig{1..4}_*.{png,pdf}.
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = HERE

ens_phase2 = json.load(open(f"{REPO}/bench/results/h200_session3/phase2/ensemble_results_final.json"))
ens_tuned  = json.load(open(f"{REPO}/bench/results/h200_session3/tuned/ensemble_results_tuned.json"))
# Canonical-Pathfinder-Triad: d=3/5 from phase2 (finetune_d3, finetune_d5);
# d=7 from tuned (finetune_d7). Mirrors the paper's canonical choice.
ens_final = {}
for k, v in ens_phase2.items():
    d = int(k.split('_')[0][1:])
    ens_final[k] = ens_tuned[k] if (d == 7 and k in ens_tuned) else v
ens_lo_p = json.load(open(f"{REPO}/bench/results/h200_lange_headtohead_low_p.json"))
ens_hi_p = json.load(open(f"{REPO}/bench/results/h200_lange_headtohead_high_p.json"))
h2h = {**ens_lo_p, **ens_hi_p}

# ---- Figure 1: LER vs p at d=7 (log-log) ----
fig, ax = plt.subplots(figsize=(7, 5))
noise_rates = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015]
noise_sub = [0.003, 0.005, 0.007, 0.010]
d7_pf_ood  = [h2h[f"d7_p{p}"]['pf_ler'] * 100 for p in noise_rates]
d7_lange   = [h2h[f"d7_p{p}"]['lange_ler'] * 100 for p in noise_rates]
d7_pm      = [h2h[f"d7_p{p}"]['pm_ler'] * 100 for p in noise_rates]
d7_maj     = [ens_final[f"d7_p{p}"]['majority_ler'] * 100 for p in noise_sub]
d7_pf_t    = [ens_final[f"d7_p{p}"]['pf_ler'] * 100 for p in noise_sub]
ax.loglog(noise_rates, d7_pm,     'o-', label='PyMatching',                 color='#888888', linewidth=2, markersize=8)
ax.loglog(noise_rates, d7_pf_ood, '^-', label='Pathfinder (Table-1 OOD)',   color='#d62728', alpha=0.6, linewidth=1.5, markersize=7)
ax.loglog(noise_sub,   d7_pf_t,   's-', label='Canonical Pathfinder (fine-tune)', color='#d62728', linewidth=2, markersize=8)
ax.loglog(noise_rates, d7_lange,  'D-', label='Lange GNN',                   color='#2ca02c', linewidth=2, markersize=8)
ax.loglog(noise_sub,   d7_maj,    '*-', label='Pathfinder-Triad (this work)', color='#1f77b4', linewidth=2.5, markersize=14)
ax.set_xlabel('Physical error rate  p', fontsize=12)
ax.set_ylabel('Logical error rate (%)', fontsize=12)
ax.set_title('d=7 decoder comparison at matched 4-parameter circuit-level noise\n(60K shots per point, 95% Wilson CIs)', fontsize=11)
ax.grid(True, which='both', ls='--', alpha=0.3)
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_ler_vs_noise_d7.png", dpi=150)
plt.savefig(f"{OUT}/fig1_ler_vs_noise_d7.pdf")
plt.close()

# ---- Figure 2: Latency-vs-LER Pareto at d=7 p=0.007 (with legend) ----
points = [
    (6.12, 1.041, 'Pathfinder+Triton (3-param Table 1)',          '#8b0000', 'o', 170),
    (6.12, 3.34,  'Canonical Pathfinder (4-param, §5.11)',         '#d62728', 's', 140),
    (6.12, 3.09,  'Pathfinder-KD (4-param, §5.13)',                '#e06060', 'D', 140),
    (6.12, 4.010, 'Pathfinder Table-1 OOD (4-param eval)',         '#ff9999', '^', 120),
    (71.67, 2.94, 'Lange et al. GNN (measured here)',              '#2ca02c', 'D', 170),
    (9.65, 1.489, 'PyMatching (3-param)',                          '#606060', 'o', 150),
    (9.65, 3.343, 'PyMatching (4-param)',                          '#b0b0b0', '^', 130),
    (72.0, 2.417, 'Pathfinder-Triad (canonical, §5.12) ★',         '#1f77b4', '*', 260),
    (76.0, 2.495, 'Pathfinder-Triad (KD variant, §5.13)',          '#4a90d9', 'X', 200),
    (63.0, 2.14,  'AlphaQubit (TPU, Sycamore noise)',              '#ff7f0e', 'p', 140),
    (40.0, 1.0,   'Gu et al. (non-matched noise)',                 '#9467bd', 'h', 140),
]
fig, ax = plt.subplots(figsize=(9, 6))
ax.axvline(7.0, color='black', linestyle=':', alpha=0.5, linewidth=1.2)
ax.text(7.2, 0.60, 'd=7 cycle budget\n(7 μs)', fontsize=9, color='gray', va='bottom')
ax.fill_between([2, 7], 0.5, 6, color='#2ca02c', alpha=0.07, zorder=0)
ax.text(3.3, 0.75, 'sustains cycle budget', fontsize=8, color='#2ca02c', ha='center', alpha=0.8, style='italic')
handles = []
for i, (lat, ler, label, color, marker, size) in enumerate(points, start=1):
    h = ax.scatter(lat, ler, s=size, c=color, marker=marker, edgecolors='black', linewidths=0.8, zorder=5)
    handles.append(h)
    off_x = 1.15 if marker in 'Do^sph' else 1.10
    ax.annotate(str(i), (lat * off_x, ler * 0.98), fontsize=9, fontweight='bold', color=color, ha='left', va='center', zorder=6)
legend_labels = [f"({i}) {p[2]}" for i, p in enumerate(points, start=1)]
ax.legend(handles, legend_labels, fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1.0),
          title='Decoder', title_fontsize=10, framealpha=0.95)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Latency (μs/syn, throughput-optimal batch)', fontsize=12)
ax.set_ylabel('Logical error rate at d=7, p=0.007 (%)', fontsize=12)
ax.set_title('Accuracy–latency Pareto at d=7, p=0.007\n(open-source + published comparators on matched or reported hardware)', fontsize=11)
ax.set_xlim(3, 200); ax.set_ylim(0.5, 6)
ax.grid(True, which='both', ls='--', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_pareto_d7.png", dpi=160, bbox_inches='tight')
plt.savefig(f"{OUT}/fig2_pareto_d7.pdf", bbox_inches='tight')
plt.close()

# ---- Figure 3: LER vs distance at p=0.007 ----
fig, ax = plt.subplots(figsize=(7, 5))
d_axis = [3, 5, 7]
pf_table1 = [1.818, 1.521, 1.041]
pm_table1 = [2.014, 1.891, 1.489]
pf_4p_tuned = [ens_final[f"d{d}_p0.007"]['pf_ler']*100 for d in d_axis]
lange_4p    = [ens_final[f"d{d}_p0.007"]['lange_ler']*100 for d in d_axis]
pm_4p       = [ens_final[f"d{d}_p0.007"]['pm_ler']*100 for d in d_axis]
maj_4p      = [ens_final[f"d{d}_p0.007"]['majority_ler']*100 for d in d_axis]
ax.semilogy(d_axis, pm_table1,   'o-',  label='PyMatching (3-param)',       color='#888888', linewidth=2, markersize=8)
ax.semilogy(d_axis, pf_table1,   's-',  label='Pathfinder (3-param, Table 1)', color='#8b0000', linewidth=2, markersize=8)
ax.semilogy(d_axis, pm_4p,       'o--', label='PyMatching (4-param)',        color='#888888', alpha=0.5, linewidth=1.5, markersize=7)
ax.semilogy(d_axis, pf_4p_tuned, 's--', label='Pathfinder (4-param, tuned)', color='#d62728', linewidth=2, markersize=8)
ax.semilogy(d_axis, lange_4p,    'D-',  label='Lange GNN (4-param)',         color='#2ca02c', linewidth=2, markersize=8)
ax.semilogy(d_axis, maj_4p,      '*-',  label='Majority Vote (4-param)',     color='#1f77b4', linewidth=2.5, markersize=14)
ax.set_xlabel('Code distance  d', fontsize=12)
ax.set_ylabel('Logical error rate at p=0.007 (%)', fontsize=12)
ax.set_title('Error-suppression scaling with code distance', fontsize=11)
ax.set_xticks([3, 5, 7])
ax.grid(True, which='both', ls='--', alpha=0.3)
ax.legend(fontsize=9, loc='upper right')
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_ler_vs_distance.png", dpi=150)
plt.savefig(f"{OUT}/fig3_ler_vs_distance.pdf")
plt.close()

# ---- Figure 4: Muon ablation depth dependence ----
fig, ax = plt.subplots(figsize=(7, 5))
d_ab = [3, 5, 7]
full_muon   = [1.818, 1.28, 1.041]
adamw_only  = [2.14, 2.20, 34.8]
width = 0.35
x = np.arange(len(d_ab))
b1 = ax.bar(x - width/2, full_muon,  width, label='Full Muon (Table 1)', color='#1f77b4', edgecolor='black')
b2 = ax.bar(x + width/2, adamw_only, width, label='AdamW only (§6.2)',   color='#ff7f0e', edgecolor='black')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels([f'd={d}' for d in d_ab])
ax.set_ylabel('LER at p=0.007 (%)', fontsize=12)
ax.set_title('Muon effect is depth-dependent (§6.2)\nCatastrophic to remove at d=7; negligible at d=3', fontsize=11)
ax.grid(True, which='both', ls='--', alpha=0.3, axis='y')
for b in list(b1) + list(b2):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h * 1.1, f'{h:.2f}%', ha='center', fontsize=9)
for i, d in enumerate(d_ab):
    rel = (adamw_only[i] / full_muon[i] - 1) * 100
    lbl = f'+{rel:.0f}%' if rel < 200 else 'catastrophic'
    color = '#d62728' if rel >= 50 else '#2ca02c'
    ax.annotate(lbl, xy=(x[i], max(full_muon[i], adamw_only[i]) * 2), ha='center',
                fontsize=10, fontweight='bold', color=color)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_muon_depth.png", dpi=150)
plt.savefig(f"{OUT}/fig4_muon_depth.pdf")
plt.close()

print("All four figures regenerated under figures/.")
