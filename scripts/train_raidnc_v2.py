#!/usr/bin/env python3
"""
Train RAIDNC v2 on the same data split as the paper.

Protocol (identical to RAIDNC_reproduce_ncv6.py where not stated otherwise):
  1. NONCODE v6 lncRNA / GENCODE v46 PC (config_both filter), 20 length bins.
  2. Independent test set: 3,000/class, seed 42; excluded from training.
  3. Training set: 60,000/class, seed 42.
  4. 5-fold stratified CV (seed 42), hexamer table rebuilt inside each fold.
  5. Decision threshold = largest value keeping out-of-fold lncRNA recall >= 0.981
     (the paper's operating point). Replaces the v1 practice of a 0.5 cutoff.
  6. Final RF on all training data; evaluation on the independent test set.

Usage:
  python train_raidnc_v2.py --preset compact --out ../model/compact
  python train_raidnc_v2.py --preset tuned   --out ../model/tuned
"""
import os, sys, json, time, argparse
import numpy as np
import joblib
from Bio import SeqIO
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
                             roc_auc_score, matthews_corrcoef, roc_curve)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raidnc_features as RF

SEED = 42
N_BINS = 20
TEST_PER_CLASS = 3000
TRAIN_PER_CLASS = 60000
TARGET_LNC_RECALL = 0.981
IDS_TO_REMOVE = {'NONHSAT136478.2', 'NONHSAT258554.1', 'NONHSAT258555.1',
                 'NONHSAT258556.1', 'NONHSAT258557.1'}

# Optuna (50 trials, objective: OOB recall_PC at recall_lncRNA = 0.981), see REPORT.md
PRESETS = {
    'tuned': dict(n_estimators=450, max_depth=None, min_samples_split=11, min_samples_leaf=2,
                  max_features=0.35, class_weight={0: 1.0, 1: 2.6062786672887372}),
    'compact': dict(n_estimators=300, max_depth=30, min_samples_split=10, min_samples_leaf=5,
                    max_features=0.35, class_weight={0: 1.0, 1: 2.814662819210254}),
}


def parse_pc_header(desc):
    has_utr, cds = False, 0
    for f in desc.split('|'):
        f = f.strip()
        if f.startswith('UTR5:') or f.startswith('UTR3:'):
            has_utr = True
        elif f.startswith('CDS:'):
            a, b = f[4:].split('-')
            cds = int(b) - int(a) + 1
    return has_utr, cds


def load_and_bin(path, label, pc_filter='both'):
    recs = list(SeqIO.parse(path, 'fasta'))
    if label == 1:
        recs = [r for r in recs if r.id not in IDS_TO_REMOVE]
    elif pc_filter == 'both':  # config_both: drop CDS-only < 100 nt and UTR-bearing CDS < 50 nt
        keep = []
        for r in recs:
            has_utr, cds = parse_pc_header(r.description)
            if not ((not has_utr and cds < 100) or (has_utr and cds < 50)):
                keep.append(r)
        recs = keep
    recs.sort(key=lambda r: len(r.seq))
    size = len(recs) // N_BINS
    return [recs[i * size:(i + 1) * size if i < N_BINS - 1 else len(recs)] for i in range(N_BINS)]


def sample(bins, n_total, seed, exclude=None):
    exclude = exclude or [set() for _ in bins]
    rng = np.random.default_rng(seed)
    per_bin = n_total // len(bins)
    out, chosen_sets = [], []
    for b, recs in enumerate(bins):
        avail = [i for i in range(len(recs)) if i not in exclude[b]]
        ch = rng.choice(avail, size=min(per_bin, len(avail)), replace=False)
        out += [str(recs[i].seq) for i in sorted(ch)]
        chosen_sets.append(set(int(i) for i in ch))
    return out, chosen_sets


def extract(seqs, n_jobs, chunk=500):
    parts = Parallel(n_jobs=n_jobs)(delayed(RF.sequence_features_batch)(seqs[i:i + chunk])
                                    for i in range(0, len(seqs), chunk))
    return np.vstack([p[0] for p in parts]), sum((p[1] for p in parts), [])


def metrics(y, p, thr):
    yp = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, yp), recall_lncRNA=recall_score(y, yp, pos_label=1),
                recall_pc=recall_score(y, yp, pos_label=0),
                precision_lncRNA=precision_score(y, yp, pos_label=1),
                precision_pc=precision_score(y, yp, pos_label=0),
                f1_lncRNA=f1_score(y, yp, pos_label=1), f1_pc=f1_score(y, yp, pos_label=0),
                f1_weighted=f1_score(y, yp, average='weighted'),
                mcc=matthews_corrcoef(y, yp), roc_auc=roc_auc_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(here, '..', 'data', 'training')
    ap.add_argument('--lnc', default=os.path.join(data, 'cdhit09_noncodev6_lncRNA_human.fa'))
    ap.add_argument('--pc', default=os.path.join(data, 'cdhit09_gencode_pc_human_unique.fa'))
    ap.add_argument('--preset', choices=sorted(PRESETS), default='compact')
    ap.add_argument('--pc_filter', choices=['both', 'nofilter'], default='both',
                    help='PC reference set: config_both (default) or config_nofilter')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_jobs', type=int, default=-1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    lnc_bins, pc_bins = load_and_bin(args.lnc, 1), load_and_bin(args.pc, 0, args.pc_filter)
    te_l, xl = sample(lnc_bins, TEST_PER_CLASS, SEED)
    te_p, xp = sample(pc_bins, TEST_PER_CLASS, SEED)
    tr_l, _ = sample(lnc_bins, TRAIN_PER_CLASS, SEED, xl)
    tr_p, _ = sample(pc_bins, TRAIN_PER_CLASS, SEED, xp)
    tr_seqs, te_seqs = tr_l + tr_p, te_l + te_p
    y = np.r_[np.ones(len(tr_l)), np.zeros(len(tr_p))].astype(int)
    y_te = np.r_[np.ones(len(te_l)), np.zeros(len(te_p))].astype(int)
    print(f'train {len(tr_seqs)}  test {len(te_seqs)}  ({time.time()-t0:.0f}s)', flush=True)

    Xb, orfs = extract(tr_seqs, args.n_jobs)
    Xb_te, orfs_te = extract(te_seqs, args.n_jobs)
    print(f'features done ({time.time()-t0:.0f}s)', flush=True)

    params = dict(PRESETS[args.preset], random_state=SEED, n_jobs=args.n_jobs)
    oof = np.zeros(len(y))
    fold_metrics = []
    for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xb, y), 1):
        tab = RF.build_hexamer_table([orfs[i] for i in a], y[a])
        X_a = RF.full_matrix(Xb[a], [orfs[i] for i in a], tab)
        X_b = RF.full_matrix(Xb[b], [orfs[i] for i in b], tab)
        oof[b] = RandomForestClassifier(**params).fit(X_a, y[a]).predict_proba(X_b)[:, 1]
        print(f'  fold {f}: AUC={roc_auc_score(y[b], oof[b]):.4f}', flush=True)

    pl = np.sort(oof[y == 1])
    thr = float(pl[int(np.floor((1 - TARGET_LNC_RECALL) * len(pl)))])
    for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xb, y), 1):
        fold_metrics.append(metrics(y[b], oof[b], thr))
    cv = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    cv_sd = {k: float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    fpr, tpr, _ = roc_curve(y, oof)
    print(f'threshold={thr:.4f}  CV: ' + ' '.join(f'{k}={v:.4f}' for k, v in cv.items()), flush=True)

    tab = RF.build_hexamer_table(orfs, y)
    clf = RandomForestClassifier(**params).fit(RF.full_matrix(Xb, orfs, tab), y)
    p_te = clf.predict_proba(RF.full_matrix(Xb_te, orfs_te, tab))[:, 1]
    test = metrics(y_te, p_te, thr)
    print('TEST: ' + ' '.join(f'{k}={v:.4f}' for k, v in test.items()), flush=True)

    clf.n_jobs = 1  # inference parallelism is handled by RAIDNC.py
    joblib.dump(clf, os.path.join(args.out, 'rf_model.joblib'), compress=3)
    joblib.dump(tab, os.path.join(args.out, 'hexamer_table.joblib'), compress=3)
    np.save(os.path.join(args.out, 'cv_oof_proba.npy'), oof)
    np.save(os.path.join(args.out, 'test_proba.npy'), p_te)
    imp = sorted(zip(RF.FEATURE_NAMES, clf.feature_importances_), key=lambda t: -t[1])
    with open(os.path.join(args.out, 'feature_importances.csv'), 'w') as fh:
        fh.write('feature,importance\n' + ''.join(f'{n},{v}\n' for n, v in imp))
    cfg = dict(version='2.0', preset=args.preset, pc_filter=f'config_{args.pc_filter}', n_features=len(RF.FEATURE_NAMES),
               feature_names=RF.FEATURE_NAMES, orf_strand='sense', min_orf_nt=RF.MIN_ORF,
               threshold=thr, threshold_rule=f'OOF lncRNA recall >= {TARGET_LNC_RECALL}',
               rf_params={k: (str(v) if k == 'class_weight' else v) for k, v in params.items()},
               cv_mean=cv, cv_std=cv_sd, test=test,
               train_per_class=TRAIN_PER_CLASS, test_per_class=TEST_PER_CLASS, seed=SEED)
    json.dump(cfg, open(os.path.join(args.out, 'model_config.json'), 'w'), indent=2)
    print(f'saved to {args.out}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
