"""
Full model-selection grid (manuscript §2.4 / §3.1) re-run with the v2 feature pipeline.

  data           : 5,000 seq/class (config_both), independent test set excluded
  k              : 3, 6, 9, 12
  regime         : 'normalized' = k-mer frequencies + length + 13 v2 biological features
                                  (hexamer excluded, as in the original model-selection step)
                   'raw'        = raw k-mer counts only
  models         : RF (90 trees), linear SVM (SGD, hinge), logistic regression,
                   multinomial NB (raw counts only: NB needs non-negative inputs)
  protocol       : (1) 80/20 stratified hold-out, primary criterion = lncRNA recall
                   (2) 5-fold stratified CV of every configuration
Outputs -> ../results/paper/model_selection_holdout.csv, model_selection_cv.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, EXP  # noqa: E402
import os, sys, time
import numpy as np, pandas as pd
from scipy import sparse
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

sys.path.insert(0, EXP)
sys.path.insert(0, f'{ROOT}/scripts')
from common import R, sample_with_ids  # noqa: E402
import raidnc_features as RF  # noqa: E402

OUT = f'{ROOT}/results/paper'
os.makedirs(OUT, exist_ok=True)
SEED = 42
KS = (3, 6, 9, 12)

t0 = time.time()
lnc_bins = R.load_and_bin(R.FASTA_LNC, label=1, n_bins=R.N_BINS)
pc_bins = R.load_and_bin(R.FASTA_PC, label=0, n_bins=R.N_BINS, filter_config='config_both')
_, lx = sample_with_ids(lnc_bins, R.TEST_SEQ_LIMIT, 1, R.TEST_RANDOM_STATE)
_, px = sample_with_ids(pc_bins, R.TEST_SEQ_LIMIT, 0, R.TEST_RANDOM_STATE)
a, _ = sample_with_ids(lnc_bins, 5000, 1, SEED, lx)
b, _ = sample_with_ids(pc_bins, 5000, 0, SEED, px)
d = pd.concat([a, b], ignore_index=True)
seqs = [s.upper() for s in d['sequence']]
y = d['label'].values

# 13 biological features + length (k-independent), v2 definitions
parts = Parallel(n_jobs=28)(delayed(RF.sequence_features_batch)(seqs[i:i + 300]) for i in range(0, len(seqs), 300))
BIO = np.vstack([p[0] for p in parts])[:, 64:78]          # length, 12 bio, frame_maxrun_ratio
print(f'data + bio features ready ({time.time()-t0:.0f}s)', flush=True)


def kmer_matrix(fit_idx, k):
    """Raw k-mer counts with vocabulary fitted on fit_idx only (no leakage)."""
    v = CountVectorizer(analyzer='char', ngram_range=(k, k), lowercase=False, dtype=np.float32)
    v.fit([seqs[i] for i in fit_idx])
    return v.transform(seqs).tocsr()


def design(C, regime):
    if regime == 'raw':
        return C
    rs = np.asarray(C.sum(axis=1)).ravel(); rs[rs == 0] = 1
    return sparse.hstack([sparse.diags(1 / rs) @ C, sparse.csr_matrix(BIO)], format='csr')


def make(name):
    return {'RF': RandomForestClassifier(n_estimators=90, random_state=SEED, n_jobs=-1),
            'SVM': SGDClassifier(loss='hinge', max_iter=5000, random_state=SEED),
            'LogReg': LogisticRegression(max_iter=5000, random_state=SEED),
            'NB': MultinomialNB()}[name]


def scores(mdl, X):
    if hasattr(mdl, 'predict_proba'):
        return mdl.predict_proba(X)[:, 1]
    return mdl.decision_function(X)


def metrics(yt, yp, sc):
    return dict(accuracy=accuracy_score(yt, yp), recall_lncRNA=recall_score(yt, yp, pos_label=1),
                recall_pc=recall_score(yt, yp, pos_label=0), precision_lncRNA=precision_score(yt, yp, pos_label=1),
                f1_lncRNA=f1_score(yt, yp, pos_label=1), f1_pc=f1_score(yt, yp, pos_label=0),
                f1_weighted=f1_score(yt, yp, average='weighted'), roc_auc=roc_auc_score(yt, sc))


CONFIGS = [(k, reg, mdl) for k in KS for reg in ('normalized', 'raw')
           for mdl in ('RF', 'SVM', 'LogReg', 'NB') if not (mdl == 'NB' and reg == 'normalized')]

# ---------------------------------------------------------------- (1) 80/20 hold-out
idx = np.arange(len(y))
tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
hold = []
for k in KS:
    C = kmer_matrix(tr, k)
    for reg in ('normalized', 'raw'):
        X = design(C, reg)
        for name in ('RF', 'SVM', 'LogReg', 'NB'):
            if (k, reg, name) not in CONFIGS:
                continue
            t = time.time()
            mdl = make(name).fit(X[tr], y[tr])
            r = dict(k=k, regime=reg, model=name, n_features=X.shape[1],
                     **metrics(y[te], mdl.predict(X[te]), scores(mdl, X[te])), seconds=time.time() - t)
            hold.append(r)
            print('HOLDOUT', {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}, flush=True)
    pd.DataFrame(hold).to_csv(f'{OUT}/model_selection_holdout.csv', index=False)
print(f'hold-out done ({time.time()-t0:.0f}s)', flush=True)

# ---------------------------------------------------------------- (2) 5-fold CV of every configuration
cvrows = []
skf = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(idx, y))
for k in KS:
    fold_C = [kmer_matrix(a_, k) for a_, _ in skf]   # vocabulary refitted inside each fold
    for reg in ('normalized', 'raw'):
        for name in ('RF', 'SVM', 'LogReg', 'NB'):
            if (k, reg, name) not in CONFIGS:
                continue
            t = time.time(); fm = []
            for (a_, b_), C in zip(skf, fold_C):
                X = design(C, reg)
                mdl = make(name).fit(X[a_], y[a_])
                fm.append(metrics(y[b_], mdl.predict(X[b_]), scores(mdl, X[b_])))
            f = pd.DataFrame(fm)
            r = dict(k=k, regime=reg, model=name, **{c: f[c].mean() for c in f},
                     **{c + '_sd': f[c].std() for c in f}, seconds=time.time() - t)
            cvrows.append(r)
            print('CV', {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items() if not kk.endswith('_sd')}, flush=True)
    pd.DataFrame(cvrows).to_csv(f'{OUT}/model_selection_cv.csv', index=False)
print(f'all done ({time.time()-t0:.0f}s)', flush=True)
