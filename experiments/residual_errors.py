"""Characterise lncRNAs still misclassified by RAIDNC v2 (compact) in each benchmark species."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, EXP  # noqa: E402
import json, sys
import numpy as np, pandas as pd, joblib
sys.path.insert(0, f'{ROOT}/scripts')
import raidnc_features as RF

W = f'{EXP}/cache'
M = f'{ROOT}/model/compact'
clf = joblib.load(f'{M}/rf_model.joblib'); clf.n_jobs = -1
tab = joblib.load(f'{M}/hexamer_table.joblib')
thr = json.load(open(f'{M}/model_config.json'))['threshold']
cols = ['length', 'longest_orf', 'orf_coverage', 'enc', 'hexamer_score', 'frame_maxrun_ratio', 'gc_content']
ix = {n: i for i, n in enumerate(RF.FEATURE_NAMES)}
rows = []
df = pd.read_pickle(f'{EXP}/cache/split_config_both_60000.pkl')
tr = df[df.split == 'train']
base, orfs = RF.sequence_features_batch(tr['sequence'].sample(20000, random_state=0).tolist())
Xh = RF.full_matrix(base, orfs, tab)
lab = tr['label'].sample(20000, random_state=0).values
for name, mask in (('human PC (train)', lab == 0), ('human lncRNA (train)', lab == 1)):
    rows.append(dict(group=name, n=int(mask.sum()), **{c: float(np.median(Xh[mask, ix[c]])) for c in cols}))
for sp in ('ATH', 'celegans', 'fruitfly', 'mouse', 'yeast'):
    seqs = pd.read_pickle(f'{W}/bench_{sp}.pkl')['sequence'].tolist()
    base, orfs = RF.sequence_features_batch(seqs)
    X = RF.full_matrix(base, orfs, tab)
    wrong = clf.predict_proba(X)[:, 1] < thr
    for g, mk in (('misclassified', wrong), ('correct', ~wrong)):
        if mk.sum():
            rows.append(dict(group=f'{sp} {g}', n=int(mk.sum()), **{c: float(np.median(X[mk, ix[c]])) for c in cols}))
    rows.append(dict(group=f'{sp} all', n=len(seqs), frac_with_orf=float((X[:, ix['longest_orf']] > 0).mean()),
                     frac_wrong=float(wrong.mean())))
out = pd.DataFrame(rows)
out.to_csv(f'{ROOT}/results/paper/residual_errors.csv', index=False)
pd.set_option('display.width', 250)
print(out.round(3).to_string(index=False))
