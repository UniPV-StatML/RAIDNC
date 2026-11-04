"""Evaluate feature-set / RF-param variants on the exact paper split.

Usage: python run_exp.py VARIANT [VARIANT ...]
Selection should use CV metrics only; test + benchmarks are reported for information.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, recall_score, precision_score, roc_auc_score,
                             matthews_corrcoef, roc_curve)
from common import R, build_split, all_features, hex_column
from build_cache import BENCH

P = R.BEST_PARAMS_DEFAULT
PAPER_RF = dict(n_estimators=P['n_estimators'], max_depth=P['max_depth'],
                min_samples_split=P['min_samples_split'], min_samples_leaf=P['min_samples_leaf'],
                max_features=P['max_features'], class_weight={0: 1.0, 1: P['lncrna_weight']})

N_EXTRA_SCALAR = 14
VARIANTS = {
    # name: (strand, extra-columns selector, rf overrides)
    'orig':        ('b', None, {}),
    'fix':         ('f', None, {}),
    'fix_scalar':  ('f', slice(0, N_EXTRA_SCALAR), {}),
    'fix_codon':   ('f', slice(N_EXTRA_SCALAR, None), {}),
    'fix_all':     ('f', slice(0, None), {}),
    'fix_maxrun':  ('f', [7], {}),  # + frame_maxrun_ratio only (ablation winner)
}
TARGET_LNC_RECALL = 0.981  # paper operating point


def oof_threshold(y, p, target=TARGET_LNC_RECALL):
    """Largest threshold whose OOF lncRNA recall is still >= target."""
    pl = np.sort(p[y == 1])
    k = int(np.floor((1 - target) * len(pl)))
    return float(pl[k])


def assemble(Fd, strand, extra_sel, hexcol):
    cols = [Fd['kmer'], Fd['length'][:, None], Fd['bio_' + strand]]
    if extra_sel is not None:
        cols.append(Fd['extra'][:, extra_sel])
    cols.append(hexcol[:, None])
    return np.hstack(cols).astype(np.float32)


def metrics(y, p, thr=0.5):
    yp = (p >= thr).astype(int)
    fpr, tpr, th = roc_curve(y, p)
    # recall_PC when recall_lncRNA (TPR) is held at the paper's 0.981
    i = np.searchsorted(tpr, 0.981)
    return dict(acc=accuracy_score(y, yp), recall_lnc=recall_score(y, yp, pos_label=1),
                recall_pc=recall_score(y, yp, pos_label=0),
                prec_lnc=precision_score(y, yp, pos_label=1), mcc=matthews_corrcoef(y, yp),
                auc=roc_auc_score(y, p), recall_pc_at_lnc981=1 - fpr[min(i, len(fpr) - 1)])


def run(name, strand, extra_sel, rf_over, out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    df = build_split()
    Fd = all_features(df['sequence'].values, 'split')
    y = df['label'].values
    tr_all = np.where(df['split'].values == 'train')[0]
    te_all = np.where(df['split'].values == 'test')[0]
    rfp = {**PAPER_RF, **rf_over, 'random_state': 42, 'n_jobs': -1}
    orf_key = 'orf_' + strand

    oof = np.zeros(len(df))
    folds = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for a, b in cv.split(tr_all, y[tr_all]):
        tr, va = tr_all[a], tr_all[b]
        m = np.zeros(len(df), bool); m[tr] = True
        h, _ = hex_column(Fd, orf_key, m, y)
        X = assemble(Fd, strand, extra_sel, h)
        clf = RandomForestClassifier(**rfp).fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
        folds.append(metrics(y[va], oof[va]))
    cvm = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}
    cvs = {k: float(np.std([f[k] for f in folds])) for k in folds[0]}

    m = np.zeros(len(df), bool); m[tr_all] = True
    h, tab = hex_column(Fd, orf_key, m, y)
    X = assemble(Fd, strand, extra_sel, h)
    t_fit = time.time()
    clf = RandomForestClassifier(**rfp).fit(X[tr_all], y[tr_all])
    t_fit = time.time() - t_fit
    pt = clf.predict_proba(X[te_all])[:, 1]
    test = metrics(y[te_all], pt)
    thr = oof_threshold(y[tr_all], oof[tr_all])
    test_thr = metrics(y[te_all], pt, thr)

    import feats as F
    bench = {}
    for bname in BENCH:
        p = os.path.join('cache', f'feats_bench_{bname}.pkl')
        if not os.path.exists(p):
            continue
        Fb = pd.read_pickle(p)
        hb = np.array([F.hex_score(o, tab) for o in Fb[orf_key]])
        pb = clf.predict_proba(assemble(Fb, strand, extra_sel, hb))[:, 1]
        bench[bname] = dict(n=len(pb), recall_lnc=float((pb >= 0.5).mean()),
                            recall_lnc_thr=float((pb >= thr).mean()))
        np.save(os.path.join(out_dir, f'{name}_bench_{bname}_proba.npy'), pb)
    np.save(os.path.join(out_dir, f'{name}_oof.npy'), oof[tr_all])
    np.save(os.path.join(out_dir, f'{name}_test_proba.npy'), pt)
    res = dict(variant=name, n_features=X.shape[1], rf={k: str(v) for k, v in rfp.items()},
               cv_mean=cvm, cv_std=cvs, test=test, oof_threshold=thr, test_at_thr=test_thr,
               bench=bench,
               fit_seconds=t_fit, total_seconds=time.time() - t0)
    json.dump(res, open(os.path.join(out_dir, f'{name}.json'), 'w'), indent=2, default=float)
    print(f"[{name}] nfeat={X.shape[1]} CV acc={cvm['acc']:.4f} rec_lnc={cvm['recall_lnc']:.4f} "
          f"rec_pc={cvm['recall_pc']:.4f} auc={cvm['auc']:.4f} pc@lnc981={cvm['recall_pc_at_lnc981']:.4f} | "
          f"TEST acc={test['acc']:.4f} rec_lnc={test['recall_lnc']:.4f} rec_pc={test['recall_pc']:.4f} "
          f"auc={test['auc']:.4f} | " + ' '.join(f"{k}={v['recall_lnc']:.3f}" for k, v in bench.items())
          + f" | {time.time()-t0:.0f}s", flush=True)
    print(f"[{name}@thr={thr:.3f}] TEST acc={test_thr['acc']:.4f} rec_lnc={test_thr['recall_lnc']:.4f} "
          f"rec_pc={test_thr['recall_pc']:.4f} mcc={test_thr['mcc']:.4f} | "
          + ' '.join(f"{k}={v['recall_lnc_thr']:.4f}" for k, v in bench.items()), flush=True)
    return res


def tuned_overrides(path='results/optuna_fix_maxrun.json'):
    bp = json.load(open(path))['best_params']
    ov = {k: v for k, v in bp.items() if k != 'lncrna_weight'}
    ov['class_weight'] = {0: 1.0, 1: bp['lncrna_weight']}
    return ov


if __name__ == '__main__':
    for v in sys.argv[1:]:
        if v == 'fix_maxrun_tuned':
            run(v, 'f', [7], tuned_overrides())
        elif v == 'fix_maxrun_compact':  # best Optuna trial with min_samples_leaf >= 5, depth 30
            run(v, 'f', [7], dict(n_estimators=300, max_depth=30, min_samples_split=10,
                                  min_samples_leaf=5, max_features=0.35, max_samples=None,
                                  class_weight={0: 1.0, 1: 2.814662819210254}))
        else:
            run(v, *VARIANTS[v])
