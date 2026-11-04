"""Shared helpers: data loading, the training/test split used for the released models, feature cache."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, EXP  # noqa: E402
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO, WORK = ROOT, EXP
CACHE = os.path.join(WORK, "cache")
os.makedirs(CACHE, exist_ok=True)
N_JOBS = 28

FASTA_LNC = os.path.join(REPO, "data", "training", "cdhit09_noncodev6_lncRNA_human.fa")
FASTA_PC = os.path.join(REPO, "data", "training", "cdhit09_gencode_pc_human_unique.fa")
N_BINS = 20
TEST_SEQ_LIMIT = 3000
TRAIN_SEQ_LIMIT = 60_000
TEST_RANDOM_STATE = RANDOM_STATE = 42
FILTER_CONFIG = "config_both"
IDS_TO_REMOVE = {"NONHSAT136478.2", "NONHSAT258554.1", "NONHSAT258555.1", "NONHSAT258556.1", "NONHSAT258557.1"}


def _pc_header(desc):
    has_utr, cds = False, 0
    for f in desc.split("|"):
        f = f.strip()
        if f.startswith("UTR5:") or f.startswith("UTR3:"):
            has_utr = True
        elif f.startswith("CDS:"):
            a, b = f[4:].split("-")
            cds = int(b) - int(a) + 1
    return has_utr, cds


def load_and_bin(path, label, n_bins=N_BINS, filter_config=None):
    """Read a FASTA, filter (lncRNA exclusion list / PC config_both), sort by length, split into n_bins bins."""
    from Bio import SeqIO
    recs = list(SeqIO.parse(path, "fasta"))
    if label == 1:
        recs = [r for r in recs if r.id not in IDS_TO_REMOVE]
    elif filter_config == "config_both":
        recs = [r for r in recs if not ((lambda u, c: (not u and c < 100) or (u and c < 50))(*_pc_header(r.description)))]
    recs.sort(key=lambda r: len(r.seq))
    size = len(recs) // n_bins
    return [recs[i * size:(i + 1) * size if i < n_bins - 1 else len(recs)] for i in range(n_bins)]


def sample_with_ids(bins, n_total, label, random_state, exclude_indices=None):
    """Length-stratified sampling (uniform across bins); also returns record ids/descriptions."""
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


def build_split(train_n=TRAIN_SEQ_LIMIT, filter_config=FILTER_CONFIG):
    """Reproduce paper split: test 3k/class (seed 42) then train n/class excluding test."""
    path = os.path.join(CACHE, f"split_{filter_config}_{train_n}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    lnc_bins = load_and_bin(FASTA_LNC, label=1, n_bins=N_BINS)
    pc_bins = load_and_bin(FASTA_PC, label=0, n_bins=N_BINS, filter_config=filter_config)
    t_l, lx = sample_with_ids(lnc_bins, TEST_SEQ_LIMIT, 1, TEST_RANDOM_STATE)
    t_p, px = sample_with_ids(pc_bins, TEST_SEQ_LIMIT, 0, TEST_RANDOM_STATE)
    tr_l, _ = sample_with_ids(lnc_bins, train_n, 1, RANDOM_STATE, lx)
    tr_p, _ = sample_with_ids(pc_bins, train_n, 0, RANDOM_STATE, px)
    test = pd.concat([t_l, t_p], ignore_index=True); test["split"] = "test"
    train = pd.concat([tr_l, tr_p], ignore_index=True); train["split"] = "train"
    df = pd.concat([train, test], ignore_index=True)
    df.to_pickle(path)
    return df


def _chunk_feats(seqs):
    import feats as F
    K, Bf, E, Of = [], [], [], []
    for s in seqs:
        K.append(F.kmer3_norm(s))
        f, of = F.compute_bio(s, 'f')
        Bf.append(f); Of.append(of)
        E.append(F.compute_extra(s))
    return (np.array(K), np.array(Bf), np.array(E, dtype=float), Of)


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
        'bio_f': np.vstack([p[1] for p in parts]),
        'extra': np.vstack([p[2] for p in parts]),
        'orf_f': sum((p[3] for p in parts), []),
    }
    pd.to_pickle(out, path)
    return out


def hex_column(F_, orf_key, train_mask, labels):
    import feats as F
    tab = F.build_hex_table([o for o, m in zip(F_[orf_key], train_mask) if m], labels[train_mask])
    return np.array([F.hex_score(o, tab) for o in F_[orf_key]]), tab
