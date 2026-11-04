"""Shared helpers: load original RAIDNC module, reproduce exact data split, cache features."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, EXP  # noqa: E402
import os, importlib.util
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO, WORK = ROOT, EXP
CACHE = os.path.join(WORK, "cache")
os.makedirs(CACHE, exist_ok=True)
N_JOBS = 28

_spec = importlib.util.spec_from_file_location(
    "raidnc", os.path.join(EXP, "legacy_v1", "RAIDNC_reproduce_ncv6.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
R.FASTA_LNC = os.path.join(REPO, "data", "training", "cdhit09_noncodev6_lncRNA_human.fa")
R.FASTA_PC = os.path.join(REPO, "data", "training", "cdhit09_gencode_pc_human_unique.fa")

BIO_NAMES = ['orf_coverage', 'longest_orf', 'gc_content', 'fickett', 'utr_ratio', 'kozak',
             'kmer_entropy', 'kmer_entropy_orf', 'n_orfs', 'orf_density', 'orf_length_ratio', 'enc']


def sample_with_ids(bins, n_total, label, random_state, exclude_indices=None):
    """Exact mirror of R.sample_from_bins, but also returns record ids/descriptions."""
    n_bins = len(bins)
    if exclude_indices is None:
        exclude_indices = [set() for _ in range(n_bins)]
    rng = np.random.default_rng(random_state)
    n_per_bin = n_total // n_bins
    rows, sel = [], [set() for _ in range(n_bins)]
    for b in range(n_bins):
        available = [i for i in range(len(bins[b])) if i not in exclude_indices[b]]
        k = min(n_per_bin, len(available))
        chosen = rng.choice(available, size=k, replace=False)
        for i in sorted(chosen):
            r = bins[b][i]
            rows.append((r.id, r.description, str(r.seq), label, b))
            sel[b].add(i)
    return pd.DataFrame(rows, columns=["id", "desc", "sequence", "label", "bin"]), sel


def build_split(train_n=R.TRAIN_SEQ_LIMIT, filter_config=R.FILTER_CONFIG):
    """Reproduce paper split: test 3k/class (seed 42) then train n/class excluding test."""
    path = os.path.join(CACHE, f"split_{filter_config}_{train_n}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    lnc_bins = R.load_and_bin(R.FASTA_LNC, label=1, n_bins=R.N_BINS)
    pc_bins = R.load_and_bin(R.FASTA_PC, label=0, n_bins=R.N_BINS, filter_config=filter_config)
    t_l, lx = sample_with_ids(lnc_bins, R.TEST_SEQ_LIMIT, 1, R.TEST_RANDOM_STATE)
    t_p, px = sample_with_ids(pc_bins, R.TEST_SEQ_LIMIT, 0, R.TEST_RANDOM_STATE)
    tr_l, _ = sample_with_ids(lnc_bins, train_n, 1, R.RANDOM_STATE, lx)
    tr_p, _ = sample_with_ids(pc_bins, train_n, 0, R.RANDOM_STATE, px)
    test = pd.concat([t_l, t_p], ignore_index=True); test["split"] = "test"
    train = pd.concat([tr_l, tr_p], ignore_index=True); train["split"] = "train"
    df = pd.concat([train, test], ignore_index=True)
    df.to_pickle(path)
    return df


def _chunk_feats(seqs):
    import feats as F
    K, Bb, Bf, E, Ob, Of = [], [], [], [], [], []
    for s in seqs:
        K.append(F.kmer3_norm(s))
        b, ob = F.compute_bio(s, 'b')
        f, of = F.compute_bio(s, 'f')
        Bb.append(b); Bf.append(f); Ob.append(ob); Of.append(of)
        E.append(F.compute_extra(s))
    return (np.array(K), np.array(Bb), np.array(Bf), np.array(E, dtype=float), Ob, Of)


def all_features(seqs, tag, chunk=500):
    """Per-sequence (fold-independent) features for every variant; cached by tag."""
    path = os.path.join(CACHE, f"feats_{tag}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    seqs = list(seqs)
    parts = Parallel(n_jobs=N_JOBS)(delayed(_chunk_feats)(seqs[i:i + chunk])
                                    for i in range(0, len(seqs), chunk))
    out = {
        'kmer': np.vstack([p[0] for p in parts]),
        'length': np.array([len(s) for s in seqs], float),
        'bio_b': np.vstack([p[1] for p in parts]),
        'bio_f': np.vstack([p[2] for p in parts]),
        'extra': np.vstack([p[3] for p in parts]),
        'orf_b': sum((p[4] for p in parts), []),
        'orf_f': sum((p[5] for p in parts), []),
    }
    pd.to_pickle(out, path)
    return out


def hex_column(F_, orf_key, train_mask, labels):
    import feats as F
    tab = F.build_hex_table([o for o, m in zip(F_[orf_key], train_mask) if m], labels[train_mask])
    return np.array([F.hex_score(o, tab) for o in F_[orf_key]]), tab
