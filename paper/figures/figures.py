"""
Generate all figures for the MLCAD paper.
Run: python figures.py
Outputs: fig1_pareto.pdf, fig2_tradeoff.pdf, fig3_version.pdf, fig4_cost.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.size': 8,
    'font.family': 'serif',
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'axes.grid': False,
})

# Colorblind-safe palette (Okabe-Ito)
CB_BLUE = '#0072B2'
CB_RED = '#D55E00'
CB_GREEN = '#009E73'
CB_GRAY = '#999999'

# ============================================================
# Data
# ============================================================

# C+ results (26Q1, correct ps units)
aes_cands = {
    'CAND-1': {'clk': 760, 'power': 0.0763, 'area': 1657, 'fmax': 1535.57},
    'CAND-2': {'clk': 1140, 'power': 0.0511, 'area': 1654, 'fmax': 1359.38},
    'CAND-3': {'clk': 709, 'power': 0.0816, 'area': 1657, 'fmax': 1561.29},
    'CAND-4': {'clk': 798, 'power': 0.0727, 'area': 1655, 'fmax': 1508.83},
}
jpeg_cands = {
    'CAND-5': {'clk': 1360, 'power': 0.0588, 'area': 6389, 'fmax': 1175.50},
    'CAND-6': {'clk': 1170, 'power': 0.0683, 'area': 6386, 'fmax': 1262.75},
    'CAND-7': {'clk': 1260, 'power': 0.0633, 'area': 6385, 'fmax': 1210.80},
}
aes_refs = {
    'BASE (380ps)': {'clk': 380, 'power': 0.1650, 'area': 1833, 'fmax': 2454.63},
    'CTRL-1 (570ps)': {'clk': 570, 'power': 0.1020, 'area': 1670, 'fmax': 1842.37},
}
jpeg_refs = {
    'BASE (578ps)': {'clk': 578, 'power': 0.1170, 'area': 5685, 'fmax': 1730.10},
    'CTRL-2 (1140ps)': {'clk': 1140, 'power': 0.0701, 'area': 6389, 'fmax': 1217.56},
}

# Calibration probe: v3.0 vs 26Q1 deltas (%)
cal_designs = ['aes', 'ibex', 'jpeg', 'riscv32i']
cal_wns = [-97.4, 1.4, -193.4, -56.0]
cal_power = [9.7, 3.1, -6.5, -2.9]
cal_area = [11.8, 2.6, -3.1, 0.7]
cal_fmax = [-3.3, 0.2, -4.5, -4.7]

# Cost breakdown
cost_phases = [
    'Phase 0\n(Bootstrap)', 'Phase 1\n(RL)', 'Phase 2a\n(GAT+BO)',
    'Phase 2a\n(EDA Calib.)', 'Phase 2a\n(C+ Probe)'
]
cost_values = [0.85, 1.20, 2.95, 0.65, 2.70]
cost_colors = [CB_BLUE, CB_BLUE, CB_BLUE, CB_RED, CB_GREEN]


# ============================================================
# Figure 1: Pareto front — power vs fmax (2 panels: AES, JPEG)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

# AES panel
for name, d in aes_refs.items():
    ax1.scatter(d['fmax'], d['power']*1000, marker='s', s=50, c=CB_GRAY,
                edgecolors='black', linewidths=0.5, zorder=3)
    ax1.annotate(name, (d['fmax'], d['power']*1000), fontsize=6,
                 xytext=(5, 4), textcoords='offset points')
for name, d in aes_cands.items():
    ax1.scatter(d['fmax'], d['power']*1000, marker='*', s=100, c=CB_RED,
                edgecolors='black', linewidths=0.3, zorder=4)
    ax1.annotate(name.split('-')[1], (d['fmax'], d['power']*1000), fontsize=6,
                 xytext=(4, -8), textcoords='offset points', color=CB_RED)

# Draw Pareto front line for AES (sorted by fmax descending)
aes_all = list(aes_refs.values()) + list(aes_cands.values())
aes_all_sorted = sorted(aes_all, key=lambda x: -x['fmax'])
# Filter to Pareto front (non-dominated in power-fmax)
pf = []
min_power = float('inf')
for pt in aes_all_sorted:
    if pt['power'] < min_power:
        pf.append(pt)
        min_power = pt['power']
pf_fmax = [p['fmax'] for p in pf]
pf_power = [p['power']*1000 for p in pf]
ax1.plot(pf_fmax, pf_power, '--', color=CB_RED, alpha=0.4, linewidth=1, zorder=1,
         label='Pareto front')

ax1.set_xlabel('fmax (MHz)')
ax1.set_ylabel('Power (mW)')
ax1.set_title('(a) AES — 4/4 candidates on front')
ax1.invert_yaxis()

# Arrow showing trade-off direction
ax1.annotate('', xy=(1400, 55), xytext=(1800, 95),
             arrowprops=dict(arrowstyle='->', color=CB_BLUE, lw=1.5))
ax1.text(1550, 80, 'BO\ntrade-off', fontsize=6.5, color=CB_BLUE, ha='center')

# JPEG panel
for name, d in jpeg_refs.items():
    ax2.scatter(d['fmax'], d['power']*1000, marker='s', s=50, c=CB_GRAY,
                edgecolors='black', linewidths=0.5, zorder=3)
    ax2.annotate(name, (d['fmax'], d['power']*1000), fontsize=6,
                 xytext=(5, 4), textcoords='offset points')
for name, d in jpeg_cands.items():
    c = CB_GREEN if name == 'CAND-6' else CB_RED
    lbl = name.split('-')[1]
    sz = 180 if name == 'CAND-6' else 100
    ax2.scatter(d['fmax'], d['power']*1000, marker='*', s=sz, c=c,
                edgecolors='black', linewidths=0.5 if name == 'CAND-6' else 0.3, zorder=5 if name == 'CAND-6' else 4)
    if name == 'CAND-6':
        ax2.annotate('CAND-6\n(dominates\nCTRL-2)', (d['fmax'], d['power']*1000),
                     fontsize=6.5, fontweight='bold', color=CB_GREEN,
                     xytext=(-60, 8), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color=CB_GREEN, lw=1.2))
    else:
        ax2.annotate(lbl, (d['fmax'], d['power']*1000), fontsize=5.5,
                     xytext=(5, -10), textcoords='offset points', color=c)

# Pareto front line for JPEG
jpeg_all = list(jpeg_refs.values()) + list(jpeg_cands.values())
jpeg_all_sorted = sorted(jpeg_all, key=lambda x: -x['fmax'])
pf = []
min_power = float('inf')
for pt in jpeg_all_sorted:
    if pt['power'] < min_power:
        pf.append(pt)
        min_power = pt['power']
pf_fmax = [p['fmax'] for p in pf]
pf_power = [p['power']*1000 for p in pf]
ax2.plot(pf_fmax, pf_power, '--', color=CB_RED, alpha=0.4, linewidth=1, zorder=1,
         label='_nolegend_')

ax2.set_xlabel('fmax (MHz)')
ax2.set_ylabel('Power (mW)')
ax2.set_title('(b) JPEG — 3/3 candidates on front')
ax2.invert_yaxis()

# Legend
ref_patch = mpatches.Patch(color=CB_GRAY, label='Baseline/Control')
cand_patch = mpatches.Patch(color=CB_RED, label='BO Candidate')
dom_patch = mpatches.Patch(color=CB_GREEN, label='Strict domination')
front_line = plt.Line2D([0], [0], color=CB_RED, linestyle='--', alpha=0.4,
                         linewidth=1, label='Pareto front')
ax2.legend(handles=[ref_patch, cand_patch, dom_patch, front_line],
           loc='lower left', fontsize=6.5)

plt.tight_layout()
plt.savefig('fig1_pareto.pdf')
plt.savefig('fig1_pareto.png')
print('Saved fig1_pareto.pdf/.png')
plt.close()


# ============================================================
# Figure 2: Trade-off direction — all 7 candidates show
# consistent power-fmax trade-off vs nearest baseline
# ============================================================
fig, ax = plt.subplots(figsize=(3.5, 2.8))

# For each candidate, show (delta_power%, delta_fmax%) relative to native-clock baseline
# AES baseline: 380ps → power=0.165, fmax=2454.63
# JPEG baseline: 578ps → power=0.117, fmax=1730.10
aes_base = {'power': 0.165, 'fmax': 2454.63}
jpeg_base = {'power': 0.117, 'fmax': 1730.10}

labels_aes = []
labels_jpeg = []
for name, d in aes_cands.items():
    dp = (d['power'] - aes_base['power']) / aes_base['power'] * 100
    df = (d['fmax'] - aes_base['fmax']) / aes_base['fmax'] * 100
    ax.scatter(df, dp, marker='o', s=60, c=CB_BLUE, edgecolors='black',
               linewidths=0.5, zorder=3)
    ax.annotate(name.split('-')[1], (df, dp), fontsize=6.5,
                xytext=(4, 4), textcoords='offset points', color=CB_BLUE)

for name, d in jpeg_cands.items():
    dp = (d['power'] - jpeg_base['power']) / jpeg_base['power'] * 100
    df = (d['fmax'] - jpeg_base['fmax']) / jpeg_base['fmax'] * 100
    ax.scatter(df, dp, marker='^', s=60, c=CB_RED, edgecolors='black',
               linewidths=0.5, zorder=3)
    ax.annotate(name.split('-')[1], (df, dp), fontsize=6.5,
                xytext=(4, 4), textcoords='offset points', color=CB_RED)

ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='-')

# Shade the "desired" quadrant (lower power, lower fmax = relaxed clock trade-off)
ax.fill_between([-50, 0], [-70, -70], [0, 0], alpha=0.06, color='green')
ax.text(-25, -35, 'Consistent\ntrade-off\nregion', fontsize=7, color='green',
        ha='center', va='center', alpha=0.7)

ax.set_xlabel(r'$\Delta$fmax vs baseline (%)')
ax.set_ylabel(r'$\Delta$Power vs baseline (%)')
ax.set_title('Trade-off direction: all 7 candidates\nreduce power at cost of fmax')

aes_h = plt.Line2D([], [], marker='o', color=CB_BLUE, linestyle='None',
                    markersize=5, label='AES (vs 380ps base)')
jpeg_h = plt.Line2D([], [], marker='^', color=CB_RED, linestyle='None',
                     markersize=5, label='JPEG (vs 578ps base)')
ax.legend(handles=[aes_h, jpeg_h], loc='upper left', fontsize=7)

plt.tight_layout()
plt.savefig('fig2_tradeoff.pdf')
plt.savefig('fig2_tradeoff.png')
print('Saved fig2_tradeoff.pdf/.png')
plt.close()


# ============================================================
# Figure 3: ORFS version sensitivity — two-panel: main + inset
# ============================================================
fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(7.0, 2.6),
                                        gridspec_kw={'width_ratios': [1, 1]})

x = np.arange(len(cal_designs))
w = 0.18
metrics = [cal_wns, cal_power, cal_area, cal_fmax]
labels = ['WNS', 'Power', 'Area', 'fmax']
colors = [CB_RED, CB_BLUE, '#E69F00', CB_GREEN]

# Left panel: all metrics, full scale
for i, (vals, lbl, col) in enumerate(zip(metrics, labels, colors)):
    offset = (i - 1.5) * w
    ax_main.bar(x + offset, vals, w, label=lbl, color=col, edgecolor='black',
                linewidth=0.3, zorder=3)

ax_main.axhline(0, color='black', linewidth=0.5)
ax_main.axhline(5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
ax_main.axhline(-5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

ax_main.annotate('WNS sign flip\n(met $\\rightarrow$ violated)',
            xy=(2 - 1.5*w, -193.4), xytext=(2.8, -140),
            fontsize=6, color=CB_RED, ha='center',
            arrowprops=dict(arrowstyle='->', color=CB_RED, lw=1))

ax_main.set_xticks(x)
ax_main.set_xticklabels(cal_designs)
ax_main.set_ylabel(r'$\Delta$ v3.0 $\rightarrow$ 26Q1 (%)')
ax_main.set_title('(a) All metrics (full scale)')
ax_main.legend(loc='lower left', fontsize=6.5, ncol=2)
ax_main.set_ylim(-220, 30)

# Right panel: non-WNS metrics zoomed to ±15%
non_wns = [cal_power, cal_area, cal_fmax]
non_wns_labels = ['Power', 'Area', 'fmax']
non_wns_colors = [CB_BLUE, '#E69F00', CB_GREEN]
w2 = 0.22

for i, (vals, lbl, col) in enumerate(zip(non_wns, non_wns_labels, non_wns_colors)):
    offset = (i - 1) * w2
    ax_zoom.bar(x + offset, vals, w2, label=lbl, color=col, edgecolor='black',
                linewidth=0.3, zorder=3)

ax_zoom.axhline(0, color='black', linewidth=0.5)
ax_zoom.axhspan(-5, 5, alpha=0.08, color='green', zorder=1)
ax_zoom.text(3.3, 6.5, r'$\pm$5%', fontsize=6.5, color='gray')

ax_zoom.set_xticks(x)
ax_zoom.set_xticklabels(cal_designs)
ax_zoom.set_ylabel(r'$\Delta$ (%)')
ax_zoom.set_title('(b) Power/Area/fmax (zoomed)')
ax_zoom.legend(loc='lower left', fontsize=6.5)
ax_zoom.set_ylim(-15, 15)

plt.tight_layout()
plt.savefig('fig3_version.pdf')
plt.savefig('fig3_version.png')
print('Saved fig3_version.pdf/.png')
plt.close()


# ============================================================
# Figure 4: Cost breakdown — stacked bar
# ============================================================
fig, ax = plt.subplots(figsize=(3.5, 2.4))

bars = ax.bar(range(len(cost_phases)), cost_values, color=cost_colors,
              edgecolor='black', linewidth=0.3, zorder=3)

for bar, val in zip(bars, cost_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'${val:.2f}', ha='center', va='bottom', fontsize=6)

ax.set_xticks(range(len(cost_phases)))
ax.set_xticklabels(cost_phases, fontsize=6)
ax.set_ylabel('Cost (USD)')
ax.set_title(f'Total project compute: ${sum(cost_values):.2f}')

# Color legend
infra = mpatches.Patch(color=CB_BLUE, label='Infrastructure / Training')
diag = mpatches.Patch(color=CB_RED, label='Diagnostics (version probe)')
verif = mpatches.Patch(color=CB_GREEN, label='Verification (C+)')
ax.legend(handles=[infra, diag, verif], fontsize=6, loc='upper left')

ax.set_ylim(0, max(cost_values) * 1.3)
plt.tight_layout()
plt.savefig('fig4_cost.pdf')
plt.savefig('fig4_cost.png')
print('Saved fig4_cost.pdf/.png')
plt.close()

print('\nAll figures generated.')
