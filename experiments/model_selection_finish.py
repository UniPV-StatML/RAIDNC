"""
Completes the model-selection grid started by model_selection_full.py.

* hold-out: keeps the 23 configurations already logged; runs the remaining k = 12 ones.
  Logistic regression at k = 12 (7.3 M features, unscaled length column) did not converge
  within 45 min with max_iter = 5000, so it is capped at max_iter = 1000 for k = 12 and flagged.
* 5-fold CV: best configuration of each model family on the hold-out (ranked by ROC-AUC,
  i.e. threshold-free discrimination), plus RF k = 3 normalized (the RAIDNC setting).
Outputs -> ../results/paper/model_selection_holdout.csv (complete), model_selection_cv.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, EXP  # noqa: E402
import ast, re, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import model_selection_full_lib as L   # data + helpers, no execution side effects

OUT = L.OUT
log = open(f'{ROOT}/results/model_selection_full.log').read()
done = [ast.literal_eval(m) for m in re.findall(r'^HOLDOUT (\{.*\})$', log, flags=re.M)]
hold = [dict(r, converged=True) for r in done]
have = {(r['k'], r['regime'], r['model']) for r in hold}
print(f'reusing {len(hold)} hold-out results', flush=True)

y, idx = L.y, np.arange(len(L.y))
tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=L.SEED)


def make(name, k):
    if name == 'LogReg' and k == 12:
        return LogisticRegression(max_iter=1000, random_state=L.SEED)
    return L.make(name)


t0 = time.time()
C = None
for k, reg, name in L.CONFIGS:
    if (k, reg, name) in have:
        continue
    if C is None or C[0] != k:
        C = (k, L.kmer_matrix(tr, k))
    X = L.design(C[1], reg)
    t = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', ConvergenceWarning)
        mdl = make(name, k).fit(X[tr], y[tr])
        conv = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    r = dict(k=k, regime=reg, model=name, n_features=X.shape[1],
             **L.metrics(y[te], mdl.predict(X[te]), L.scores(mdl, X[te])), seconds=time.time() - t, converged=conv)
    hold.append(r)
    print('HOLDOUT', {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
H = pd.DataFrame(hold).sort_values(['k', 'regime', 'model'])
H.to_csv(f'{OUT}/model_selection_holdout.csv', index=False)
print(f'hold-out complete ({time.time()-t0:.0f}s)', flush=True)

# ------------------------------------------------------------------ CV of the best configuration per family
best = H.sort_values('roc_auc', ascending=False).groupby('model').head(1)
sel = {(int(r.k), r.regime, r.model) for r in best.itertuples()} | {(3, 'normalized', 'RF')}
print('CV on:', sorted(sel), flush=True)
skf = list(StratifiedKFold(5, shuffle=True, random_state=L.SEED).split(idx, y))
rows = []
for k in sorted({s[0] for s in sel}):
    fold_C = [L.kmer_matrix(a, k) for a, _ in skf]
    for (kk, reg, name) in sorted(s for s in sel if s[0] == k):
        t = time.time(); fm = []
        for (a, b), Cf in zip(skf, fold_C):
            X = L.design(Cf, reg)
            mdl = make(name, k).fit(X[a], y[a])
            fm.append(L.metrics(y[b], mdl.predict(X[b]), L.scores(mdl, X[b])))
        f = pd.DataFrame(fm)
        r = dict(k=k, regime=reg, model=name, **{c: f[c].mean() for c in f},
                 **{c + '_sd': f[c].std() for c in f}, seconds=time.time() - t)
        rows.append(r)
        print('CV', {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items() if not a.endswith('_sd')}, flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/model_selection_cv.csv', index=False)
print(f'all done ({time.time()-t0:.0f}s)', flush=True)
