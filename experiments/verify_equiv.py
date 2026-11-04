import time
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from common import R, build_split, BIO_NAMES
import feats as F

df = build_split()
print(df.groupby(["split", "label"]).size().to_dict())
sub = df.sample(1500, random_state=0).reset_index(drop=True)
seqs = sub["sequence"]

t = time.time()
Xo, vect = R.create_kmer_features(sub, k=3, normalize=True, **R.BIO_FLAGS)
Xo = Xo.toarray()
t_orig = time.time() - t
print("vocab", len(vect.vocabulary_), "orig time", round(t_orig, 1))

t = time.time()
K = np.array([F.kmer3_norm(s) for s in seqs])
B = np.array([F.compute_bio(s, 'b')[0] for s in seqs])
t_new = time.time() - t
Xn = np.hstack([K, seqs.str.len().values[:, None].astype(float), B])
print("new time", round(t_new, 1), "speedup", round(t_orig / t_new, 1))
names = list(vect.get_feature_names_out()) + ['length'] + BIO_NAMES
d = np.abs(Xo - Xn).max(axis=0)
bad = [(n, v) for n, v in zip(names, d) if v > 1e-9]
print("max abs diff overall", d.max(), "bad:", bad)

tab_o = R.build_hexamer_table(seqs.values, sub["label"].values)
orfs = [F.compute_bio(s, 'b')[1] for s in seqs]
tab_n = F.build_hex_table(orfs, sub["label"].values)
print("hex table maxdiff", max(abs(tab_o[k] - tab_n[k]) for k in tab_o))
ho = R.compute_hexamer_score(seqs, tab_o).ravel()
hn = np.array([F.hex_score(o, tab_n) for o in orfs])
print("hex score maxdiff", np.abs(ho - hn).max())

# How often is the longest ORF on the minus strand?
import orfipy_core
for lab in (0, 1):
    ss = seqs[sub.label == lab]
    minus = 0; tot = 0
    for s in ss:
        o = orfipy_core.orfs(s.upper(), minlen=100, starts=['ATG'], stops=F.STOPS)
        if o:
            tot += 1
            minus += max(o, key=lambda x: x[1] - x[0])[2] == '-'
    print(f"label {lab}: longest ORF on minus strand {minus}/{tot} = {minus/tot:.1%}")
