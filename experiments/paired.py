import json, numpy as np
from scipy.stats import binomtest
y = np.r_[np.ones(3000), np.zeros(3000)].astype(int)  # test order: lnc then PC
a = np.load('results/orig_test_proba.npy'); b = np.load('results/fix_maxrun_test_proba.npy')
ta = json.load(open('results/orig.json'))['oof_threshold']; tb = json.load(open('results/fix_maxrun.json'))['oof_threshold']
ca = (a >= ta).astype(int) == y; cb = (b >= tb).astype(int) == y
for name, m in [('lncRNA', y == 1), ('PC', y == 0), ('all', y >= 0)]:
    n01 = int((ca & ~cb & m).sum()); n10 = int((~ca & cb & m).sum())
    p = binomtest(n10, n01 + n10).pvalue if n01 + n10 else 1
    print(f"{name:6s} orig-only-correct={n01:4d} new-only-correct={n10:4d}  McNemar p={p:.2e}")
rng = np.random.default_rng(0); d = []
for _ in range(2000):
    i = rng.integers(0, 6000, 6000); d.append(cb[i].mean() - ca[i].mean())
print("acc diff 95% CI", np.percentile(d, [2.5, 97.5]).round(4))
