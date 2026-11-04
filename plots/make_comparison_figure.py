"""
Comparison figure: RAIDNC v1 / v2 vs CPC2 and LncDC.

  A  independent human test set (3,000 lncRNA + 3,000 PC): class-wise recall,
     precision, F1 and overall accuracy
  B  cross-species lncRNA recall on NONCODE v6 (+ macro / micro average)

Sources
  CPC2, LncDC, RAIDNC v1  -> manuscript Figure 2 (test) and Figure 3 (species)
  RAIDNC v2               -> ../model/<preset>/model_config.json (test) and
                             ../results/benchmark_<preset>.csv (species, via RAIDNC.py)
Outputs: Figure_comparison.png (300 dpi), Figure_comparison.pdf, figure_comparison_data.csv
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')

TOOLS = ['CPC2', 'LncDC', 'RAIDNC v1\n(UTR+CDS filter)', 'RAIDNC v1\n(no filter)',
         'RAIDNC v2\n(compact)', 'RAIDNC v2\n(tuned)']
TEST_COLS = ['Recall\nlncRNA', 'Precision\nlncRNA', 'F1\nlncRNA',
             'Recall\nPC', 'Precision\nPC', 'F1\nPC', 'Accuracy']
SPECIES = ['A. thaliana', 'C. elegans', 'D. melanogaster', 'M. musculus', 'S. cerevisiae']
SP_COLS = [f'$\\it{{{s.split()[0]}}}$\n$\\it{{{s.split()[1]}}}$' for s in SPECIES] + ['Macro\navg', 'Micro\navg']
SP_N = np.array([4046, 3154, 42848, 131974, 55])

# manuscript Figure 2 (percent)
PAPER_TEST = {
    'CPC2':  [93.57, 73.79, 82.51, 66.77, 91.21, 77.10, 80.17],
    'LncDC': [96.87, 76.05, 85.21, 69.41, 95.67, 80.45, 83.16],
    'RAIDNC v1\n(UTR+CDS filter)': [98.10, 80.32, 88.33, 75.97, 97.56, 85.42, 87.03],
    'RAIDNC v1\n(no filter)':      [98.00, 80.59, 88.45, 76.40, 97.45, 85.65, 87.20],
}
# manuscript Figure 3 (percent): ATH, C.el, D.mel, M.mus, S.cer, macro, micro
PAPER_SP = {
    'CPC2':  [98.15, 85.42, 92.70, 86.81, 89.09, 90.40, 88.40],
    'LncDC': [98.54, 87.16, 89.26, 94.32, 94.55, 92.80, 93.10],
    'RAIDNC v1\n(UTR+CDS filter)': [96.47, 90.23, 92.76, 98.43, 94.55, 94.50, 96.90],
    'RAIDNC v1\n(no filter)':      [96.86, 89.82, 92.36, 98.30, 94.55, 94.40, 96.70],
}


def v2_rows(preset):
    t = json.load(open(os.path.join(ROOT, 'model', preset, 'model_config.json')))['test']
    test = [t[k] * 100 for k in ('recall_lncRNA', 'precision_lncRNA', 'f1_lncRNA',
                                 'recall_pc', 'precision_pc', 'f1_pc', 'accuracy')]
    b = pd.read_csv(os.path.join(ROOT, 'results', f'benchmark_{preset}.csv')).set_index('dataset')
    r = np.array([b.loc[s, 'recall_lncRNA'] for s in SPECIES]) * 100
    return test, list(r) + [r.mean(), (r * SP_N).sum() / SP_N.sum()]


test, sp = dict(PAPER_TEST), dict(PAPER_SP)
for preset in ('compact', 'tuned'):
    test[f'RAIDNC v2\n({preset})'], sp[f'RAIDNC v2\n({preset})'] = v2_rows(preset)
A = np.array([test[t] for t in TOOLS])
B = np.array([sp[t] for t in TOOLS])

pd.concat([
    pd.DataFrame(A, index=[t.replace('\n', ' ') for t in TOOLS],
                 columns=['test_' + c.replace('\n', '_') for c in TEST_COLS]),
    pd.DataFrame(B, index=[t.replace('\n', ' ') for t in TOOLS],
                 columns=['recall_' + s.replace(' ', '_') for s in SPECIES] + ['recall_macro', 'recall_micro']),
], axis=1).round(2).to_csv(os.path.join(HERE, 'figure_comparison_data.csv'))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
CMAP = plt.get_cmap('Blues')
INK, MUTED = '#1f1f1f', '#555555'


def heat(ax, M, cols, vmin, title, sep_after=(3,), group_sep=None, best_ignore_ties=False):
    norm = Normalize(vmin=vmin, vmax=100)
    nr, nc = M.shape
    for i in range(nr):
        for j in range(nc):
            v = M[i, j]
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=CMAP(norm(v)), edgecolor='white', lw=2))
            col_best = np.round(M[:, j], 2).max()
            is_best = np.isclose(round(v, 2), col_best)
            txt_col = 'white' if norm(v) > 0.55 else INK
            ax.text(j + .5, i + .5, f'{v:.2f}%', ha='center', va='center', fontsize=9.5,
                    color=txt_col, fontweight='bold' if is_best else 'normal')
    for r in sep_after:  # separator between competitors / v1 / v2
        ax.axhline(r + 1, color=INK, lw=1.6)
    if group_sep is not None:
        for c in group_sep:
            ax.axvline(c, color=INK, lw=1.6)
    ax.set_xlim(0, nc); ax.set_ylim(nr, 0)
    ax.set_xticks(np.arange(nc) + .5); ax.set_xticklabels(cols, fontsize=9.5)
    ax.set_yticks(np.arange(nr) + .5); ax.set_yticklabels(TOOLS, fontsize=9.5)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc='left', fontsize=11.5, fontweight='bold', pad=40, color=INK)
    return norm


fig = plt.figure(figsize=(15.5, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[7, 7, 0.25], wspace=0.06)
axA, axB, cax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

vmin = 65
heat(axA, A, TEST_COLS, vmin, 'A   Human independent test set (3,000 lncRNA + 3,000 PC)',
     sep_after=(1, 3), group_sep=(3, 6))
norm = heat(axB, B, SP_COLS, vmin, 'B   Cross-species lncRNA recall (NONCODE v6)',
            sep_after=(1, 3), group_sep=(5,))
axB.set_yticklabels([])

sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
cb = fig.colorbar(sm, cax=cax)
cb.set_label('Score (%)', fontsize=10)
cb.outline.set_visible(False)

fig.text(0.125, 0.015,
         'Bold = best value in column (ties included). CPC2, LncDC and RAIDNC v1 values as reported in the manuscript; '
         'RAIDNC v2 evaluated with the same test set and FASTA files through RAIDNC.py.\n'
         'RAIDNC v2 decision thresholds were fixed on out-of-fold training predictions (lncRNA recall 0.981). '
         'Micro average weighted by number of sequences per species.',
         fontsize=8, color=MUTED, va='bottom')
fig.subplots_adjust(left=0.115, right=0.965, top=0.80, bottom=0.12)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, f'Figure_comparison.{ext}'), dpi=300)
print('saved', os.path.join(HERE, 'Figure_comparison.png'))
