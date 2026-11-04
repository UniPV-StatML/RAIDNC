"""Model size / inference speed / feature-extraction speed: paper vs tuned vs compact."""
import os, time, joblib
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from common import R, build_split, all_features, hex_column, BIO_NAMES
from run_exp import PAPER_RF, assemble, tuned_overrides
import feats as F

df = build_split()
Fd = all_features(df['sequence'].values, 'split')
y = df['label'].values
tr = df['split'].values == 'train'
h, tab = hex_column(Fd, 'orf_f', tr, y)
X = assemble(Fd, 'f', [7], h)
COMPACT = dict(n_estimators=300, max_depth=30, min_samples_split=10, min_samples_leaf=5,
               max_features=0.35, class_weight={0: 1.0, 1: 2.814662819210254})
cfgs = {'paper': PAPER_RF, 'tuned': {**PAPER_RF, **tuned_overrides()}, 'compact': COMPACT}
Xbig = np.tile(X[~tr], (10, 1))  # 60k rows
for name, p in cfgs.items():
    clf = RandomForestClassifier(**p, random_state=42, n_jobs=-1).fit(X[tr], y[tr])
    path = f'cache/model_{name}.joblib'
    joblib.dump(clf, path, compress=3)
    t = time.time(); clf.predict_proba(Xbig); tp = time.time() - t
    clf.n_jobs = 1
    t = time.time(); clf.predict_proba(Xbig[:6000]); t1 = time.time() - t
    print(f"{name:8s} size={os.path.getsize(path)/1e6:.0f}MB(compressed) "
          f"predict 60k rows: {tp:.1f}s (all cores), 6k rows 1 core: {t1:.1f}s", flush=True)

# feature extraction, single core, 3000 sequences
seqs = pd.Series(df['sequence'].values[:3000])
t = time.time()
R.create_kmer_features(pd.DataFrame({'sequence': seqs}), k=3, normalize=True, **R.BIO_FLAGS)
tab_o = R.build_hexamer_table(seqs.values[:10], np.r_[np.zeros(5), np.ones(5)])
R.compute_hexamer_score(seqs, tab_o)
t_orig = time.time() - t
t = time.time()
for s in seqs:
    F.kmer3_norm(s); _, o = F.compute_bio(s, 'f'); F.hex_score(o, tab); F._stop_runs(s.upper())
t_new = time.time() - t
print(f"feature extraction 3000 seqs, 1 core: original {t_orig:.1f}s, new {t_new:.1f}s "
      f"({t_orig/t_new:.1f}x)", flush=True)
