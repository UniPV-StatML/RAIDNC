"""Fast drop-one / add-one ablation of the 14 new scalar features (strand-fixed base).

Selection signal = OOB AUC and OOB recall_PC at recall_lnc=0.981 on the training set
(no test / benchmark data used for selection). Benchmarks reported for diagnosis.
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from common import build_split, all_features, hex_column
from run_exp import BASELINE_RF, N_EXTRA_SCALAR
from build_cache import BENCH
import feats as F

EXTRA = F.EXTRA_NAMES[:N_EXTRA_SCALAR]
df = build_split()
Fd = all_features(df['sequence'].values, 'split')
y = df['label'].values
tr = df['split'].values == 'train'
te = ~tr
h, tab = hex_column(Fd, 'orf_f', tr, y)
bench = {b: pd.read_pickle(f'cache/feats_bench_{b}.pkl') for b in BENCH}
bhex = {b: np.array([F.hex_score(o, tab) for o in Fb['orf_f']]) for b, Fb in bench.items()}


def X_of(Fx, hx, extra_idx):
    return np.hstack([Fx['kmer'], Fx['length'][:, None], Fx['bio_f'],
                      Fx['extra'][:, extra_idx], hx[:, None]]).astype(np.float32)


def pc_at(yt, p, tpr_target=0.981):
    fpr, tpr, _ = roc_curve(yt, p)
    return 1 - fpr[min(np.searchsorted(tpr, tpr_target), len(fpr) - 1)]


def evaluate(name, extra_idx):
    X = X_of(Fd, h, extra_idx)
    clf = RandomForestClassifier(**BASELINE_RF, oob_score=True, random_state=42, n_jobs=-1)
    clf.fit(X[tr], y[tr])
    oob = clf.oob_decision_function_[:, 1]
    pt = clf.predict_proba(X[te])[:, 1]
    r = dict(name=name, oob_auc=roc_auc_score(y[tr], oob), oob_pc_at981=pc_at(y[tr], oob),
             test_auc=roc_auc_score(y[te], pt))
    for b, Fb in bench.items():
        pb = clf.predict_proba(X_of(Fb, bhex[b], extra_idx))[:, 1]
        r[b] = float((pb >= 0.5).mean())
        if b == 'celegans':
            np.save(f'results/abl_{name}_celegans.npy', pb)
    print(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items()), flush=True)
    return r


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    rows = [evaluate('fix', []), evaluate('fix_scalar', list(range(N_EXTRA_SCALAR)))]
    for i, n in enumerate(EXTRA):
        rows.append(evaluate(f'add_{n}', [i]))
    for i, n in enumerate(EXTRA):
        rows.append(evaluate(f'drop_{n}', [j for j in range(N_EXTRA_SCALAR) if j != i]))
    pd.DataFrame(rows).to_csv('results/ablation.csv', index=False)
