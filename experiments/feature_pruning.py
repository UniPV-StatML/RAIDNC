"""
Feature-pruning ablation for RAIDNC v2 (compact hyperparameters, config_both).

For each variant (full model, drop-one for each of the 15 non-k-mer features, grouped drops)
the model is retrained in 5-fold stratified CV on the 60k/class training set (same folds for
every variant, hexamer table rebuilt in each fold). Selection evidence = paired per-fold
differences vs. the full model (ROC-AUC, PC recall at lncRNA recall 0.981).
For information only: a final model per variant is trained on all training data and the
lncRNA recall on the five NONCODE v6 species is computed at the variant's own OOF threshold.

Outputs -> ../results/feature_pruning/{folds.csv, summary.csv}
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, EXP  # noqa: E402
import os, sys, time, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, EXP)
sys.path.insert(0, f'{ROOT}/scripts')
from common import build_split, all_features, hex_column  # noqa: E402
import raidnc_features as RF  # noqa: E402
import feats as F  # noqa: E402
from train_raidnc_v2 import PRESETS  # noqa: E402

OUT = f'{ROOT}/results/feature_pruning'
os.makedirs(OUT, exist_ok=True)
SPECIES = ['ATH', 'celegans', 'fruitfly', 'mouse', 'yeast']
NAMES = RF.FEATURE_NAMES
IDX = {n: i for i, n in enumerate(NAMES)}
NON_KMER = NAMES[64:]
LOW6 = ['gc_content', 'enc', 'orf_density', 'n_orfs', 'kozak', 'utr_ratio']
CORE4 = ['hexamer_score', 'frame_maxrun_ratio', 'longest_orf', 'length']

VARIANTS = {'full': []}
VARIANTS.update({f'drop_{n}': [n] for n in NON_KMER})
VARIANTS['drop_low6'] = LOW6
VARIANTS['drop_kmers'] = NAMES[:64]
VARIANTS['only_core4'] = [n for n in NAMES if n not in CORE4]


def X_of(Fd, h):
    return np.hstack([Fd['kmer'], Fd['length'][:, None], Fd['bio_f'], Fd['extra'][:, 7:8], h[:, None]]).astype(np.float32)


def pc_at(y, p, target=0.981):
    fpr, tpr, _ = roc_curve(y, p)
    return 1 - fpr[min(np.searchsorted(tpr, target), len(fpr) - 1)]


t0 = time.time()
df = build_split()
Fd = all_features(df['sequence'].values, 'split')
y_all = df['label'].values
tr = np.where(df['split'].values == 'train')[0]
y = y_all[tr]
params = dict(PRESETS['compact'], random_state=42, n_jobs=-1)
skf = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))

# per-fold design matrices (hexamer table fitted on the fold's training part) -- shared by all variants
fold_X = []
for a, b in skf:
    m = np.zeros(len(df), bool); m[tr[a]] = True
    h, _ = hex_column(Fd, 'orf_f', m, y_all)
    fold_X.append(X_of(Fd, h)[tr])
m = np.zeros(len(df), bool); m[tr] = True
h_full, tab = hex_column(Fd, 'orf_f', m, y_all)
X_full = X_of(Fd, h_full)[tr]
bench = {}
for s in SPECIES:
    Fb = pd.read_pickle(f'{EXP}/cache/feats_bench_{s}.pkl')
    hb = np.array([F.hex_score(o, tab) for o in Fb['orf_f']])
    bench[s] = X_of(Fb, hb)
print(f'matrices ready ({time.time()-t0:.0f}s)', flush=True)

fold_rows, summ = [], []
for vname, drop in VARIANTS.items():
    keep = [i for i, n in enumerate(NAMES) if n not in set(drop)]
    oof = np.zeros(len(y))
    for f, ((a, b), X) in enumerate(zip(skf, fold_X)):
        clf = RandomForestClassifier(**params).fit(X[a][:, keep], y[a])
        oof[b] = clf.predict_proba(X[b][:, keep])[:, 1]
        fold_rows.append(dict(variant=vname, fold=f, auc=roc_auc_score(y[b], oof[b]), pc_at981=pc_at(y[b], oof[b])))
    pl = np.sort(oof[y == 1]); thr = float(pl[int(np.floor(0.019 * len(pl)))])
    clf = RandomForestClassifier(**params).fit(X_full[:, keep], y)
    sp = {s: float((clf.predict_proba(bench[s][:, keep])[:, 1] >= thr).mean()) for s in SPECIES}
    fr = pd.DataFrame([r for r in fold_rows if r['variant'] == vname])
    summ.append(dict(variant=vname, n_features=len(keep), auc=fr.auc.mean(), pc_at981=fr.pc_at981.mean(),
                     oof_auc=roc_auc_score(y, oof), threshold=thr, **{f'recall_{s}': v for s, v in sp.items()},
                     macro_species=float(np.mean(list(sp.values())))))
    print(f"[{vname}] nfeat={len(keep)} AUC={fr.auc.mean():.4f} pc@981={fr.pc_at981.mean():.4f} "
          + ' '.join(f'{s}={v:.4f}' for s, v in sp.items()) + f" ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(fold_rows).to_csv(f'{OUT}/folds.csv', index=False)
    pd.DataFrame(summ).to_csv(f'{OUT}/summary.csv', index=False)

# paired differences vs full model
fd = pd.DataFrame(fold_rows)
base = fd[fd.variant == 'full'].set_index('fold')
rows = []
for v in VARIANTS:
    d = fd[fd.variant == v].set_index('fold')
    da, dp = d.auc - base.auc, d.pc_at981 - base.pc_at981
    rows.append(dict(variant=v, d_auc_mean=da.mean(), d_auc_sd=da.std(), d_auc_folds_better=int((da > 0).sum()),
                     d_pc_mean=dp.mean(), d_pc_sd=dp.std(), d_pc_folds_better=int((dp > 0).sum())))
P = pd.DataFrame(rows).merge(pd.DataFrame(summ), on='variant')
P.to_csv(f'{OUT}/summary.csv', index=False)
pd.set_option('display.width', 250)
print(P.drop(columns=['threshold', 'oof_auc']).round(4).to_string(index=False))
print(f'all done ({time.time()-t0:.0f}s)', flush=True)
