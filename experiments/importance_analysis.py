"""
Feature-importance analysis for RAIDNC v2 (compact model), complementing the
impurity-based (MDI) importances stored in model/compact/feature_importances.csv.

  1. permutation importance, single features   (drop in ROC-AUC on the independent test set)
  2. permutation importance, groups of correlated features permuted jointly
  3. SHAP (TreeExplainer): mean |SHAP| and direction, balanced test subsample

Outputs -> ../results/importance/
Run (WSL): python importance_analysis.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, EXP  # noqa: E402
import os, sys, json, time
import numpy as np, pandas as pd, joblib
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, matthews_corrcoef

V2 = ROOT
WORK = EXP
sys.path.insert(0, f'{V2}/scripts')
import raidnc_features as RF  # noqa: E402

OUT = f'{V2}/results/importance'
os.makedirs(OUT, exist_ok=True)
N_REPEATS = 10
rng0 = np.random.default_rng(42)

# ---------------------------------------------------------------- data + model
df = pd.read_pickle(f'{WORK}/cache/split_config_both_60000.pkl')
te = df[df['split'] == 'test']
y = te['label'].values
clf = joblib.load(f'{V2}/model/compact/rf_model.joblib')
clf.n_jobs = -1
tab = joblib.load(f'{V2}/model/compact/hexamer_table.joblib')
thr = json.load(open(f'{V2}/model/compact/model_config.json'))['threshold']
base, orfs = RF.sequence_features_batch(te['sequence'].tolist())
X = RF.full_matrix(base, orfs, tab)
names = RF.FEATURE_NAMES
p0 = clf.predict_proba(X)[:, 1]
auc0, mcc0 = roc_auc_score(y, p0), matthews_corrcoef(y, p0 >= thr)
print(f'baseline test AUC={auc0:.4f} MCC={mcc0:.4f}', flush=True)


def permute_drop(cols, seed):
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    perm = rng.permutation(len(Xp))
    Xp[:, cols] = X[perm][:, cols]  # same row permutation for all columns of a group
    p = clf.predict_proba(Xp)[:, 1]
    return auc0 - roc_auc_score(y, p), mcc0 - matthews_corrcoef(y, p >= thr)


def perm_table(groups):
    rows = []
    for g, cols in groups.items():
        r = np.array([permute_drop(cols, s) for s in range(N_REPEATS)])
        rows.append(dict(feature=g, n_cols=len(cols), auc_drop_mean=r[:, 0].mean(), auc_drop_sd=r[:, 0].std(),
                         mcc_drop_mean=r[:, 1].mean(), mcc_drop_sd=r[:, 1].std()))
    return pd.DataFrame(rows).sort_values('auc_drop_mean', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- 1. single-feature permutation
t = time.time()
single = perm_table({n: [i] for i, n in enumerate(names)})
single.to_csv(f'{OUT}/permutation_single.csv', index=False)
print(f'[1] single permutation done ({time.time()-t:.0f}s)', flush=True)

# ---------------------------------------------------------------- 2. grouped permutation
idx = {n: i for i, n in enumerate(names)}
GROUPS = {
    '3-mer composition (64)':       list(range(64)),
    'ORF length/structure':         [idx[n] for n in ('longest_orf', 'orf_coverage', 'orf_length_ratio',
                                                      'n_orfs', 'orf_density')],
    'hexamer_score':                [idx['hexamer_score']],
    'frame_maxrun_ratio':           [idx['frame_maxrun_ratio']],
    'codon usage in ORF':           [idx['kmer_entropy_orf'], idx['enc']],
    'fickett':                      [idx['fickett']],
    'length':                       [idx['length']],
    'GC / transcript entropy':      [idx['gc_content'], idx['kmer_entropy']],
    'start context (kozak, UTR)':   [idx['kozak'], idx['utr_ratio']],
    'all ORF-based features':       [idx[n] for n in ('longest_orf', 'orf_coverage', 'orf_length_ratio', 'n_orfs',
                                                      'orf_density', 'kmer_entropy_orf', 'enc', 'kozak',
                                                      'utr_ratio', 'hexamer_score')],
}
groups = perm_table(GROUPS)
groups.to_csv(f'{OUT}/permutation_groups.csv', index=False)
print(f'[2] grouped permutation done ({time.time()-t:.0f}s)', flush=True)

# ---------------------------------------------------------------- 3. SHAP
import shap  # noqa: E402
clf.n_jobs = 1


def shap_chunk(rows):
    ex = shap.TreeExplainer(clf)
    sv = ex.shap_values(X[rows], check_additivity=False)
    sv = sv[1] if isinstance(sv, list) else sv
    return sv[:, :, 1] if sv.ndim == 3 else sv


t = time.time()
probe = shap_chunk(np.arange(4))
per_sample = (time.time() - t) / 4
n_workers = 28
budget_s = 20 * 60
n_shap = int(min(2000, max(280, budget_s * n_workers / max(per_sample, 1e-3))))
n_shap -= n_shap % 2
lnc_i = rng0.choice(np.where(y == 1)[0], n_shap // 2, replace=False)
pc_i = rng0.choice(np.where(y == 0)[0], n_shap // 2, replace=False)
rows = np.sort(np.r_[lnc_i, pc_i])
print(f'[3] SHAP: {per_sample:.1f}s/sample/core -> explaining {n_shap} balanced test samples '
      f'on {n_workers} cores', flush=True)
chunks = np.array_split(rows, n_workers * 2)
S = np.vstack(Parallel(n_jobs=n_workers)(delayed(shap_chunk)(c) for c in chunks))
Xs = X[rows]
shap_df = pd.DataFrame({
    'feature': names,
    'mean_abs_shap': np.abs(S).mean(0),
    # direction: correlation between feature value and its SHAP value (+ = higher value -> more lncRNA)
    'direction_corr': [np.corrcoef(Xs[:, j], S[:, j])[0, 1] if Xs[:, j].std() > 0 else np.nan
                       for j in range(S.shape[1])],
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
shap_df.to_csv(f'{OUT}/shap_mean_abs.csv', index=False)
np.save(f'{OUT}/shap_values.npy', S); np.save(f'{OUT}/shap_rows.npy', rows)
print(f'[3] SHAP done ({time.time()-t:.0f}s)', flush=True)

# ---------------------------------------------------------------- summary table (top 15 by each method)
mdi = pd.read_csv(f'{V2}/model/compact/feature_importances.csv')
summ = (mdi.rename(columns={'importance': 'MDI'})
        .merge(single[['feature', 'auc_drop_mean', 'auc_drop_sd']], on='feature')
        .merge(shap_df[['feature', 'mean_abs_shap', 'direction_corr']], on='feature'))
for c in ('MDI', 'auc_drop_mean', 'mean_abs_shap'):
    summ[f'rank_{c}'] = summ[c].rank(ascending=False, method='min').astype(int)
summ = summ.sort_values('rank_mean_abs_shap')
summ.to_csv(f'{OUT}/importance_summary.csv', index=False)
pd.set_option('display.width', 220)
print(summ.head(15).round(4).to_string(index=False))
print(groups.round(4).to_string(index=False))
