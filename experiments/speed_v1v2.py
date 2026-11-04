"""Feature+prediction time, 1 core: v1 pipeline (original functions) vs v2 module, on D. melanogaster."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, EXP  # noqa: E402
import sys, time, joblib
import numpy as np, pandas as pd
from Bio import SeqIO
from common import R, REPO
sys.path.insert(0, f'{ROOT}/scripts')
import raidnc_features as RF

fa = f'{REPO}/data/benchmark_fasta/noncodev6/NONCODEv6_fruitfly.fa'
seqs = pd.Series([str(r.seq) for r in SeqIO.parse(fa, 'fasta')])
print('n seqs', len(seqs), flush=True)

# v1: exactly what scripts/RAIDNC.py does (feature functions + vectorizer + hexamer), 1 core
d1 = f'{REPO}/model/config_both/k3/60000'
vect, hex1, rf1 = (joblib.load(f'{d1}/{f}') for f in ('kmer_vectorizer.joblib', 'hexamer_table.joblib', 'rf_model.joblib'))
rf1.n_jobs = 1
t = time.time()
X = R.transform_kmer_features(pd.DataFrame({'sequence': seqs}), vect, k=3, normalize=True, **R.BIO_FLAGS).toarray()
X = np.hstack([X, R.compute_hexamer_score(seqs, hex1)])
t_feat1 = time.time() - t
t = time.time(); rf1.predict_proba(X); t_pred1 = time.time() - t
print(f'v1: features {t_feat1:.1f}s  predict {t_pred1:.1f}s  total {t_feat1+t_pred1:.1f}s', flush=True)

d2 = f'{ROOT}/model/compact'
rf2, hex2 = joblib.load(f'{d2}/rf_model.joblib'), joblib.load(f'{d2}/hexamer_table.joblib')
t = time.time()
base, orfs = RF.sequence_features_batch(seqs.tolist())
X2 = RF.full_matrix(base, orfs, hex2)
t_feat2 = time.time() - t
t = time.time(); rf2.predict_proba(X2); t_pred2 = time.time() - t
print(f'v2: features {t_feat2:.1f}s  predict {t_pred2:.1f}s  total {t_feat2+t_pred2:.1f}s  '
      f'speedup {(t_feat1+t_pred1)/(t_feat2+t_pred2):.1f}x', flush=True)
