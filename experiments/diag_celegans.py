"""Which C. elegans lncRNAs flip to 'coding' after the strand fix / new features, and why."""
import numpy as np, pandas as pd
from common import BIO_NAMES, build_split, all_features
import feats as F

b = pd.read_pickle('cache/bench_celegans.pkl')
Fb = pd.read_pickle('cache/feats_bench_celegans.pkl')
P = {v: np.load(f'results/{v}_bench_celegans_proba.npy') for v in ('orig', 'fix', 'fix_scalar')}
cols = ['length'] + [f'{n}_b' for n in BIO_NAMES] + [f'{n}_f' for n in BIO_NAMES] + F.EXTRA_NAMES[:14]
D = pd.DataFrame(np.hstack([Fb['length'][:, None], Fb['bio_b'], Fb['bio_f'], Fb['extra'][:, :14]]), columns=cols)
for v, p in P.items():
    D['p_' + v] = p

lost_fix = (P['orig'] >= .5) & (P['fix'] < .5)
gain_fix = (P['orig'] < .5) & (P['fix'] >= .5)
lost_sc = (P['fix'] >= .5) & (P['fix_scalar'] < .5)
gain_sc = (P['fix'] < .5) & (P['fix_scalar'] >= .5)
print(f"n={len(D)}  orig->fix: lost {lost_fix.sum()} gained {gain_fix.sum()} | "
      f"fix->fix_scalar: lost {lost_sc.sum()} gained {gain_sc.sum()}")

# reference distributions: human training lncRNA vs PC
df = build_split(); Fh = all_features(df['sequence'].values, 'split')
tr = (df['split'] == 'train').values; y = df['label'].values
H = pd.DataFrame(np.hstack([Fh['length'][:, None], Fh['bio_b'], Fh['bio_f'], Fh['extra'][:, :14]]), columns=cols)
show = ['length', 'longest_orf_b', 'longest_orf_f', 'orf_coverage_f', 'n_orfs_f', 'orf_length_ratio_f',
        'fickett_f', 'kozak_f', 'enc_f'] + F.EXTRA_NAMES[:14]
summary = pd.DataFrame({
    'human_PC': H[tr & (y == 0)][show].median(), 'human_lnc': H[tr & (y == 1)][show].median(),
    'cel_all': D[show].median(), 'cel_lost_fix': D[lost_fix][show].median(),
    'cel_lost_scalar': D[lost_sc][show].median(), 'cel_still_ok': D[P['fix_scalar'] >= .5][show].median()})
pd.set_option('display.width', 200)
print(summary.round(3).to_string())
D[lost_fix | lost_sc].assign(id=b['id'][lost_fix | lost_sc].values).to_csv('results/celegans_flips.csv', index=False)
