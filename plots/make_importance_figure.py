"""
Feature-importance figure for RAIDNC v2 (compact model).
  A  MDI (Random Forest impurity, training)
  B  permutation importance, single features (ROC-AUC drop on independent test set, mean ± SD, 10 repeats)
  C  mean |SHAP| (2,000 balanced test sequences)
  D  permutation importance of feature groups permuted jointly
Inputs: ../results/importance/*.csv   Outputs: Figure_feature_importance.png / .pdf
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results', 'importance')
TOP = 15
BAR, BAR_NEW = '#3a78b5', '#0b3b6f'
INK, MUTED, GRID = '#1f1f1f', '#555555', '#e3e3e3'

s = pd.read_csv(os.path.join(RES, 'importance_summary.csv'))
g = pd.read_csv(os.path.join(RES, 'permutation_groups.csv'))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5})


def label(f):
    return f + '  (new)' if f == 'frame_maxrun_ratio' else f


def bars(ax, d, col, title, xlabel, err=None, fmt='{:.3f}'):
    d = d.sort_values(col, ascending=False).head(TOP).iloc[::-1]
    colors = [BAR_NEW if f == 'frame_maxrun_ratio' else BAR for f in d['feature']]
    ax.barh(range(len(d)), d[col], height=0.62, color=colors,
            xerr=d[err] if err else None, error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([label(f) for f in d['feature']])
    for t in ax.get_yticklabels():
        if t.get_text().endswith('(new)'):
            t.set_fontweight('bold')
    for i, v in enumerate(d[col]):
        ax.text(v + d[col].max() * 0.02 + (d[err].iloc[i] if err else 0), i, fmt.format(v),
                va='center', fontsize=8, color=MUTED)
    ax.set_xlim(0, d[col].max() * 1.25 + (d[err].max() if err else 0))
    ax.set_title(title, loc='left', fontweight='bold', fontsize=10.5, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y', length=0)


fig, axes = plt.subplots(1, 4, figsize=(19, 6.2), gridspec_kw=dict(wspace=0.62))
bars(axes[0], s, 'MDI', 'A   Impurity importance (MDI)', 'mean decrease in impurity')
bars(axes[1], s, 'auc_drop_mean', 'B   Permutation importance', 'ROC-AUC drop on test set', err='auc_drop_sd')
bars(axes[2], s, 'mean_abs_shap', 'C   SHAP', 'mean |SHAP value|')
gg = g[~g['feature'].str.startswith('3-mer')].copy()   # tri-nucleotide group not reported
bars(axes[3], gg, 'auc_drop_mean', 'D   Grouped permutation', 'ROC-AUC drop on test set', err='auc_drop_sd')

fig.text(0.02, 0.012,
         'RAIDNC v2 (compact). MDI from training; permutation importance on the independent test set '
         '(3,000 + 3,000; baseline ROC-AUC 0.973; mean ± SD of 10 permutations); SHAP on 2,000 balanced test sequences.\n'
         'D: correlated features permuted jointly. "All ORF-based features" = ORF length/structure + hexamer + codon usage + Kozak/UTR.',
         fontsize=8, color=MUTED)
fig.subplots_adjust(left=0.125, right=0.985, top=0.93, bottom=0.14)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, f'Figure_feature_importance.{ext}'), dpi=300)
print('saved')
