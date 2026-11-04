"""Optuna RF retuning on the fix_maxrun feature set.

Objective: OOB recall_PC at OOB recall_lncRNA = 0.981 (threshold-free, matches the
paper's lncRNA-first operating point). Uses training data only.
Note: hexamer table is built on the full training set, so OOB rows have a slight
leakage through that single column (same compromise as the paper's Optuna step);
the winner is re-validated afterwards with proper per-fold-hexamer 5-fold CV.
"""
import json, time
import numpy as np, optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from common import build_split, all_features, hex_column
from run_exp import assemble, VARIANTS

N_TRIALS = 50
strand, extra_sel, _ = VARIANTS['fix_maxrun']
df = build_split()
Fd = all_features(df['sequence'].values, 'split')
y = df['label'].values
tr = df['split'].values == 'train'
h, _ = hex_column(Fd, 'orf_' + strand, tr, y)
X = assemble(Fd, strand, extra_sel, h)[tr]
y = y[tr]


def pc_at(yt, p, target=0.981):
    fpr, tpr, _ = roc_curve(yt, p)
    return 1 - fpr[min(np.searchsorted(tpr, target), len(fpr) - 1)]


def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int('n_estimators', 100, 500, step=50),
        max_depth=trial.suggest_categorical('max_depth', [None, 20, 30, 50]),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
        max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.15, 0.25, 0.35]),
        max_samples=trial.suggest_categorical('max_samples', [None, 0.5, 0.7]),
        class_weight={0: 1.0, 1: trial.suggest_float('lncrna_weight', 0.5, 6.0, log=True)},
    )
    t = time.time()
    clf = RandomForestClassifier(**params, oob_score=True, bootstrap=True, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    oob = clf.oob_decision_function_[:, 1]
    oob = np.nan_to_num(oob, nan=0.5)
    s = pc_at(y, oob)
    trial.set_user_attr('oob_auc', roc_auc_score(y, oob))
    trial.set_user_attr('fit_s', time.time() - t)
    print(f"trial {trial.number}: pc@981={s:.4f} auc={trial.user_attrs['oob_auc']:.4f} "
          f"{time.time()-t:.0f}s {trial.params}", flush=True)
    return s


study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
# seed with the paper's configuration so the search can only match or beat it
study.enqueue_trial(dict(n_estimators=220, max_depth=50, min_samples_split=4, min_samples_leaf=10,
                         max_features='log2', max_samples=None, lncrna_weight=5.63))
study.optimize(objective, n_trials=N_TRIALS)
best = dict(study.best_params)
json.dump(dict(best_params=best, best_value=study.best_value,
               trials=[dict(n=t.number, value=t.value, params=t.params, **t.user_attrs)
                       for t in study.trials]),
          open('results/optuna_fix_maxrun.json', 'w'), indent=2, default=str)
print('BEST', study.best_value, best, flush=True)
