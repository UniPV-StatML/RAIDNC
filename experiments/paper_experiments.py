"""
Manuscript experiments re-run with the v2 feature pipeline (same protocol as
RAIDNC_reproduce_ncv6.py steps 1-2, config_both, test set excluded):

  model selection  : 5,000 seq/class, 5-fold CV; RF (90 trees), linear SVM (SGD, hinge),
                     logistic regression, multinomial NB (raw 3-mer counts);
                     v2 features without the hexamer column (as in the original step 1)
  sample size      : 5k / 15k / 30k / 60k seq/class, RF (90 trees), 5-fold CV,
                     full v2 features with per-fold hexamer table

Outputs -> ../results/paper/model_selection.csv, sample_size.csv
Run (WSL): python paper_experiments.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, EXP  # noqa: E402
import os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

sys.path.insert(0, EXP)
sys.path.insert(0, f'{ROOT}/scripts')
from common import R, sample_with_ids  # noqa: E402  (original v1 loading/binning code)
import raidnc_features as RF  # noqa: E402

OUT = f'{ROOT}/results/paper'
os.makedirs(OUT, exist_ok=True)
SEED = 42


def feats(seqs):
    parts = Parallel(n_jobs=28)(delayed(RF.sequence_features_batch)(seqs[i:i + 500])
                                for i in range(0, len(seqs), 500))
    return np.vstack([p[0] for p in parts]), sum((p[1] for p in parts), [])


def m(y, yp, score=None):
    d = dict(accuracy=accuracy_score(y, yp), f1_weighted=f1_score(y, yp, average='weighted'),
             f1_pc=f1_score(y, yp, pos_label=0), f1_lncRNA=f1_score(y, yp, pos_label=1),
             recall_pc=recall_score(y, yp, pos_label=0), recall_lncRNA=recall_score(y, yp, pos_label=1))
    d['roc_auc'] = roc_auc_score(y, score) if score is not None else np.nan
    return d


t0 = time.time()
lnc_bins = R.load_and_bin(R.FASTA_LNC, label=1, n_bins=R.N_BINS)
pc_bins = R.load_and_bin(R.FASTA_PC, label=0, n_bins=R.N_BINS, filter_config='config_both')
_, lx = sample_with_ids(lnc_bins, R.TEST_SEQ_LIMIT, 1, R.TEST_RANDOM_STATE)
_, px = sample_with_ids(pc_bins, R.TEST_SEQ_LIMIT, 0, R.TEST_RANDOM_STATE)


def train_sample(n):
    a, _ = sample_with_ids(lnc_bins, n, 1, SEED, lx)
    b, _ = sample_with_ids(pc_bins, n, 0, SEED, px)
    d = pd.concat([a, b], ignore_index=True)
    return d['sequence'].tolist(), d['label'].values


# ------------------------------------------------------------------ model selection
seqs, y = train_sample(5000)
base, orfs = feats(seqs)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []
models = {
    'RF': RandomForestClassifier(n_estimators=90, random_state=SEED, n_jobs=-1),
    'SVM': SGDClassifier(loss='hinge', max_iter=5000, random_state=SEED, n_jobs=-1),
    'LogReg': LogisticRegression(max_iter=5000, random_state=SEED, n_jobs=-1),
}
for name, mdl in models.items():
    yp = cross_val_predict(mdl, base, y, cv=cv)
    meth = 'predict_proba' if hasattr(mdl, 'predict_proba') else 'decision_function'
    sc = cross_val_predict(mdl, base, y, cv=cv, method=meth)
    sc = sc[:, 1] if sc.ndim == 2 else sc
    rows.append(dict(model=name, **m(y, yp, sc)))
    print(rows[-1], flush=True)
# NB on raw 3-mer counts (needs non-negative integer counts)
def raw_counts(s):
    a = RF._LUT[np.frombuffer(s.upper().encode(), dtype=np.uint8)]
    if len(a) < 3:
        return np.zeros(64, int)
    ok = (a[:-2] >= 0) & (a[1:-1] >= 0) & (a[2:] >= 0)
    return np.bincount((a[:-2] * 16 + a[1:-1] * 4 + a[2:])[ok], minlength=64)


counts = np.array([raw_counts(s) for s in seqs])
yp = cross_val_predict(MultinomialNB(), counts, y, cv=cv)
rows.append(dict(model='NB', **m(y, yp)))
print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/model_selection.csv', index=False)
print(f'model selection done ({time.time()-t0:.0f}s)', flush=True)

# ------------------------------------------------------------------ sample size
rows = []
for n in (5000, 15000, 30000, 60000):
    seqs, y = train_sample(n)
    base, orfs = feats(seqs)
    fm = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(base, y):
        tab = RF.build_hexamer_table([orfs[i] for i in a], y[a])
        Xa = RF.full_matrix(base[a], [orfs[i] for i in a], tab)
        Xb = RF.full_matrix(base[b], [orfs[i] for i in b], tab)
        p = RandomForestClassifier(n_estimators=90, random_state=SEED, n_jobs=-1).fit(Xa, y[a]).predict_proba(Xb)[:, 1]
        fm.append(m(y[b], (p >= 0.5).astype(int), p))
    f = pd.DataFrame(fm)
    row = dict(n_per_class=n, **{k: f[k].mean() for k in f}, **{k + '_sd': f[k].std() for k in f},
               false_positive_rate_pc=1 - f['recall_pc'].mean())
    rows.append(row)
    print({k: round(v, 4) for k, v in row.items()}, flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/sample_size.csv', index=False)
print(f'all done ({time.time()-t0:.0f}s)', flush=True)
