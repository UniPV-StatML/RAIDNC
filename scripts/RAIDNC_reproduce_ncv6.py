#!/usr/bin/env python3
"""
RAIDNC_reproduce_ncv6_ncv6.py
============================
End-to-end reproducibility script for:

    Bonomo et al., "RAIDNC: Random forest Assisted Identification and
    Discrimination of Non-Coding transcripts"

Aligned with the final paper configuration (NONCODE v6, CD-HIT 90%,
k=3 + 12 biological features, Optuna Fβ2 tuning, 60k sequences/class).

Steps
-----
  1. Model selection    -- 5-fold CV comparing RF, SVM, LogReg, NB
  2. Sample-size study  -- 5k/15k/30k/60k × config_both  5-fold CV
  3. Final model        -- 60k/class, Optuna params, bio features,
                          5-fold CV + independent test

Data
----
  lncRNA : NONCODE v6 human (CD-HIT 90% identity)
           data/cdhit_NONCODEv6/cdhit09_noncodev6_lncRNA_human.fa
  PC     : GENCODE v46 human (CD-HIT 90% identity)
           data/cdhit09_gencode_pc_human_unique.fa

Usage
-----
  python RAIDNC_reproduce_ncv6.py           # runs all steps
  python RAIDNC_reproduce_ncv6.py --step 1
  python RAIDNC_reproduce_ncv6.py --step 2 3

Requirements
------------
  pip install -r requirements.txt   (includes optuna, orfipy)

Notes
-----
  - Data is loaded once (via load_and_bin, 20 bins) and shared across all steps.
  - The independent test set is sampled first (TEST_RANDOM_STATE=42) and kept
    excluded from all training.
  - Step 4 mirrors run_optuna_experiment in RAIDNC_experiments_v5_mar16_noncodev6.py.
  - Optuna paper defaults (BEST_PARAMS_DEFAULT) are used for steps 2 and 3.
  - random_state = 42 throughout for full reproducibility.
"""

import os
import sys
import time
import json
import argparse
import joblib

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import (
    StratifiedKFold, train_test_split, cross_val_predict
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score, fbeta_score,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import orfipy_core
from itertools import product as itertools_product

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────

RANDOM_STATE      = 42
TEST_RANDOM_STATE = 42
K                 = 3       # final k-mer size (tri-nucleotides)
N_BINS            = 20      # length bins for stratified sampling
N_FOLDS           = 5       # CV folds for final evaluation
N_JOBS            = -1
TEST_SEQ_LIMIT    = 3000    # sequences per class in independent test set
TRAIN_SEQ_LIMIT   = 60_000  # sequences per class for final model
FILTER_CONFIG     = 'config_both'  # PC filter for final model

OPTUNA_N_TRIALS      = 200
OPTUNA_CV_FOLDS      = 3
OPTUNA_BETA          = 2.0
OPTUNA_MIN_RECALL_PC = 0.78

# Biological feature flags used in the final model (all 12)
BIO_FLAGS = dict(
    orf_coverage=True, gc_content=True, fickett=True,
    utr_structure=True, kozak=True, kmer_entropy=True,
    kmer_entropy_orf=True, orf_stats=True, enc=True,
)
# Note: hexamer is handled separately (per-fold table construction)

# Paper-reported Optuna defaults (used when step 2 is skipped)
BEST_PARAMS_DEFAULT = {
    'n_estimators':      220,
    'max_depth':         50,
    'min_samples_split': 4,
    'min_samples_leaf':  10,
    'max_features':      'log2',
    'lncrna_weight':     5.63,
}

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results_reproduce_ncv6')
MODEL_DIR   = os.path.join(SCRIPT_DIR, 'model')
MODEL_FILE  = os.path.join(MODEL_DIR, 'rf_model_ncv6_ncv6.joblib')
VECT_FILE   = os.path.join(MODEL_DIR, 'kmer_vectorizer_ncv6_ncv6.joblib')
HEX_FILE    = os.path.join(MODEL_DIR, 'hexamer_table_ncv6_ncv6.joblib')

FASTA_LNC = os.path.join(
    DATA_DIR, 'cdhit_NONCODEv6', 'cdhit09_noncodev6_lncRNA_human.fa')
FASTA_PC  = os.path.join(
    DATA_DIR, 'cdhit09_gencode_pc_human_unique.fa')

# lncRNA IDs to exclude (confirmed problematic sequences)
IDS_TO_REMOVE = {
    'NONHSAT136478.2',
    'NONHSAT258554.1',
    'NONHSAT258555.1',
    'NONHSAT258556.1',
    'NONHSAT258557.1',
}

# ── PC Header Parsing and Filtering ──────────────────────────────────

def parse_pc_header(description):
    """
    Parse a GENCODE protein-coding FASTA header.

    Header format (pipe-separated):
      ENST|ENSG|OTTHUMG|OTTHUMT|name|gene|length|[UTR5:s-e|][CDS:s-e|][UTR3:s-e|]

    Returns dict with keys: has_utr5, has_utr3, cds_length.
    """
    fields = description.split('|')
    has_utr5, has_utr3, cds_length = False, False, 0
    for field in fields:
        field = field.strip()
        if field.startswith('UTR5:'):
            has_utr5 = True
        elif field.startswith('UTR3:'):
            has_utr3 = True
        elif field.startswith('CDS:'):
            start, end = field[4:].split('-')
            cds_length = int(end) - int(start) + 1
    return {'has_utr5': has_utr5, 'has_utr3': has_utr3, 'cds_length': cds_length}


def apply_pc_filter(records, config_name):
    """
    Filter protein-coding records based on CDS/UTR criteria.

    config_nofilter  : no filtering
    config_both      : remove CDS-only with CDS < 100 nt AND
                       UTR-bearing with CDS < 50 nt
    """
    if config_name == 'config_nofilter':
        return records, 0
    filtered, n_removed = [], 0
    for rec in records:
        info = parse_pc_header(rec.description)
        has_utr = info['has_utr5'] or info['has_utr3']
        cds_len = info['cds_length']
        remove = False
        if config_name == 'config_both':
            if not has_utr and cds_len < 100:
                remove = True
            elif has_utr and cds_len < 50:
                remove = True
        if remove:
            n_removed += 1
        else:
            filtered.append(rec)
    return filtered, n_removed


# ── Data Loading and Binning ─────────────────────────────────────────

def load_and_bin(fasta_path, label, n_bins=N_BINS, filter_config=None):
    """
    Load sequences from FASTA, optionally filter, sort by length,
    and split into n_bins equal-size bins.

    Returns list of n_bins lists of SeqRecord, sorted by length (ascending).
    """
    records = list(SeqIO.parse(fasta_path, 'fasta'))
    if label == 1:
        before = len(records)
        records = [r for r in records if r.id not in IDS_TO_REMOVE]
        n_excl = before - len(records)
        if n_excl > 0:
            print(f"    Excluded {n_excl} lncRNA IDs (IDS_TO_REMOVE)")
    if label == 0 and filter_config is not None:
        records, n_removed = apply_pc_filter(records, filter_config)
        print(f"    PC filter '{filter_config}': removed {n_removed}, "
              f"remaining {len(records)}")
    records.sort(key=lambda r: len(r.seq))
    total = len(records)
    bin_size = total // n_bins
    bins = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else total
        bins.append(records[start:end])
    return bins


def sample_from_bins(bins, n_total, label, random_state, exclude_indices=None):
    """
    Sample n_total sequences uniformly across bins.

    Returns (DataFrame with ['sequence','label'], list of per-bin selected index sets).
    """
    n_bins = len(bins)
    if exclude_indices is None:
        exclude_indices = [set() for _ in range(n_bins)]
    rng = np.random.default_rng(random_state)
    n_per_bin = n_total // n_bins
    all_seqs = []
    selected_indices = [set() for _ in range(n_bins)]
    for b in range(n_bins):
        available = [i for i in range(len(bins[b]))
                     if i not in exclude_indices[b]]
        k = min(n_per_bin, len(available))
        if k < n_per_bin:
            print(f"    WARNING: bin {b+1} has only {len(available)} "
                  f"available (requested {n_per_bin})")
        chosen = rng.choice(available, size=k, replace=False)
        for i in sorted(chosen):
            all_seqs.append(str(bins[b][i].seq))
            selected_indices[b].add(i)
    df = pd.DataFrame({'sequence': all_seqs, 'label': [label] * len(all_seqs)})
    return df, selected_indices


def sample_test_set(lnc_bins, pc_bins):
    """
    Sample the independent test set: TEST_SEQ_LIMIT per class, stratified.

    Returns (test_df, lnc_excl, pc_excl) — exclusion index sets to pass
    to subsequent training sampling calls.
    """
    print(f"\n  Sampling independent test set ({TEST_SEQ_LIMIT}/class, "
          f"{TEST_SEQ_LIMIT // N_BINS}/bin)...")
    df_lnc, lnc_excl = sample_from_bins(
        lnc_bins, TEST_SEQ_LIMIT, label=1, random_state=TEST_RANDOM_STATE)
    df_pc, pc_excl = sample_from_bins(
        pc_bins, TEST_SEQ_LIMIT, label=0, random_state=TEST_RANDOM_STATE)
    test_df = pd.concat([df_lnc, df_pc], ignore_index=True)
    print(f"    Test set: {len(df_lnc)} lncRNA + {len(df_pc)} PC = "
          f"{len(test_df)} total")
    return test_df, lnc_excl, pc_excl


# ── K-mer Feature Engineering ─────────────────────────────────────────

def get_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def compute_orf_features(sequences):
    """ORF coverage (longest_ORF / seq_length) and longest ORF length."""
    results = []
    for seq in sequences:
        seq_upper = str(seq).upper()
        seq_len = len(seq_upper)
        if seq_len == 0:
            results.append((0.0, 0.0))
            continue
        orfs = orfipy_core.orfs(seq_upper, minlen=100, starts=['ATG'],
                                stops=['TAA', 'TAG', 'TGA'])
        if orfs:
            longest = max(stop - start for start, stop, _, _ in orfs)
            orf_cov = longest / seq_len
        else:
            longest, orf_cov = 0, 0.0
        results.append((orf_cov, float(longest)))
    return np.array(results, dtype=float)


def compute_gc_content(sequences):
    """GC content = (G+C) / seq_length."""
    results = []
    for seq in sequences:
        seq_upper = str(seq).upper()
        seq_len = len(seq_upper)
        if seq_len == 0:
            results.append(0.0)
            continue
        results.append((seq_upper.count('G') + seq_upper.count('C')) / seq_len)
    return np.array(results, dtype=float).reshape(-1, 1)


def compute_fickett_score(sequences):
    """
    Fickett TESTCODE score (Fickett 1982, lookup tables from CPC2/CPAT).
    """
    position_parameter = [1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 0.0]
    content_parameter  = [0.33, 0.31, 0.29, 0.27, 0.25, 0.23, 0.21, 0.19, 0.17, 0.0]
    position_probability = {
        'A': [0.51, 0.55, 0.57, 0.52, 0.48, 0.58, 0.57, 0.54, 0.50, 0.36],
        'C': [0.29, 0.44, 0.55, 0.49, 0.52, 0.60, 0.60, 0.56, 0.51, 0.38],
        'G': [0.62, 0.67, 0.74, 0.65, 0.61, 0.62, 0.52, 0.41, 0.31, 0.17],
        'T': [0.51, 0.60, 0.69, 0.64, 0.62, 0.67, 0.58, 0.48, 0.39, 0.24],
    }
    position_weight = {'A': 0.062, 'C': 0.093, 'G': 0.205, 'T': 0.154}
    content_probability = {
        'A': [0.40, 0.55, 0.58, 0.58, 0.52, 0.48, 0.45, 0.45, 0.38, 0.19],
        'C': [0.50, 0.63, 0.59, 0.50, 0.46, 0.45, 0.47, 0.56, 0.59, 0.33],
        'G': [0.21, 0.40, 0.47, 0.50, 0.52, 0.56, 0.57, 0.52, 0.44, 0.23],
        'T': [0.30, 0.49, 0.56, 0.53, 0.48, 0.48, 0.52, 0.57, 0.60, 0.51],
    }
    content_weight = {'A': 0.084, 'C': 0.076, 'G': 0.081, 'T': 0.055}

    def _lookup(value, params, probs, weight, base):
        if value < 0:
            return 0.0
        for idx, threshold in enumerate(params):
            if value >= threshold:
                return probs[base][idx] * weight[base]
        return 0.0

    results = []
    for seq in sequences:
        dna = str(seq).upper()
        total_base = len(dna)
        if total_base < 2:
            results.append(0.0)
            continue
        phase_0, phase_1, phase_2 = dna[::3], dna[1::3], dna[2::3]
        score = 0.0
        for base in ('A', 'C', 'G', 'T'):
            counts = [phase_0.count(base), phase_1.count(base), phase_2.count(base)]
            position_val = max(counts) / (min(counts) + 1.0)
            score += _lookup(position_val, position_parameter,
                             position_probability, position_weight, base)
            content_val = sum(counts) / total_base
            score += _lookup(content_val, content_parameter,
                             content_probability, content_weight, base)
        results.append(score)
    return np.array(results, dtype=float).reshape(-1, 1)


def compute_utr_structure(sequences):
    """
    UTR asymmetry ratio = post_orf / (pre_orf + post_orf).
    Returns 0.5 (neutral) if no ORF found.
    """
    results = []
    for seq in sequences:
        seq_upper = str(seq).upper()
        seq_len = len(seq_upper)
        if seq_len == 0:
            results.append(0.5)
            continue
        orfs = orfipy_core.orfs(seq_upper, minlen=100, starts=['ATG'],
                                stops=['TAA', 'TAG', 'TGA'])
        if not orfs:
            results.append(0.5)
            continue
        best = max(orfs, key=lambda x: x[1] - x[0])
        pre_orf = best[0]
        post_orf = seq_len - best[1]
        flanking = pre_orf + post_orf
        results.append(post_orf / flanking if flanking > 0 else 0.5)
    return np.array(results, dtype=float).reshape(-1, 1)


# Kozak PWM log-odds matrix (GENCODE v46, n=55,808 PC transcripts)
_KOZAK_LOG_ODDS = np.array([
    [-0.2854, -0.1659,  0.6542, -0.4852],  # -6
    [-0.3456,  0.3160,  0.2373, -0.3411],  # -5
    [-0.0523,  0.5763,  0.0536, -0.9805],  # -4
    [ 0.8632, -1.2388,  0.5537, -1.7891],  # -3 (critical)
    [ 0.2840,  0.5275, -0.3519, -0.8432],  # -2
    [-0.3745,  0.8008,  0.2129, -1.6102],  # -1
    [ 0.0000,  0.0000,  0.0000,  0.0000],  # A(0) fixed
    [ 0.0000,  0.0000,  0.0000,  0.0000],  # T(+1) fixed
    [ 0.0000,  0.0000,  0.0000,  0.0000],  # G(+2) fixed
    [-0.1624, -0.7221,  0.9515, -0.8202],  # +3 (critical)
    [ 0.1167,  0.6584, -0.4815, -0.6870],  # +4
    [-0.6315,  0.0471,  0.5582, -0.2364],  # +5
])
_KOZAK_BASE_IDX  = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
_KOZAK_SCORE_POS = [0, 1, 2, 3, 4, 5, 9, 10, 11]  # exclude ATG positions


def compute_kozak_score(sequences):
    """Kozak context log-odds score (9 informative positions, ATG excluded)."""
    results = []
    for seq in sequences:
        seq_upper = str(seq).upper()
        seq_len = len(seq_upper)
        if seq_len == 0:
            results.append(0.0)
            continue
        orfs = orfipy_core.orfs(seq_upper, minlen=100, starts=['ATG'],
                                stops=['TAA', 'TAG', 'TGA'])
        if not orfs:
            results.append(0.0)
            continue
        atg_pos = max(orfs, key=lambda x: x[1] - x[0])[0]
        w_start, w_end = atg_pos - 6, atg_pos + 6
        if w_start < 0 or w_end > seq_len:
            results.append(0.0)
            continue
        window = seq_upper[w_start:w_end]
        score = sum(_KOZAK_LOG_ODDS[p, _KOZAK_BASE_IDX[window[p]]]
                    for p in _KOZAK_SCORE_POS if window[p] in _KOZAK_BASE_IDX)
        results.append(score)
    return np.array(results, dtype=float).reshape(-1, 1)


def compute_kmer_entropy(sequences, k=3):
    """Shannon entropy of k-mer frequency distribution (whole transcript)."""
    results = []
    for seq in sequences:
        seq_upper = str(seq).upper()
        if len(seq_upper) < k:
            results.append(0.0)
            continue
        kmers = [seq_upper[i:i+k] for i in range(len(seq_upper) - k + 1)]
        counts = {}
        for km in kmers:
            counts[km] = counts.get(km, 0) + 1
        total = len(kmers)
        entropy = -sum((c / total) * np.log2(c / total) for c in counts.values())
        results.append(entropy)
    return np.array(results, dtype=float).reshape(-1, 1)


def compute_kmer_entropy_orf(sequences, k=3):
    """Shannon entropy of in-frame k-mer distribution within the longest ORF."""
    results = []
    for seq in sequences:
        seq_upper = str(seq).upper()
        if len(seq_upper) < k:
            results.append(0.0)
            continue
        orfs = orfipy_core.orfs(seq_upper, minlen=100, starts=['ATG'],
                                stops=['TAA', 'TAG', 'TGA'])
        if not orfs:
            results.append(0.0)
            continue
        best = max(orfs, key=lambda x: x[1] - x[0])
        orf_seq = seq_upper[best[0]:best[1]]
        kmers = [orf_seq[i:i+k] for i in range(0, len(orf_seq) - k + 1, 3)]
        if not kmers:
            results.append(0.0)
            continue
        counts = {}
        for km in kmers:
            counts[km] = counts.get(km, 0) + 1
        total = len(kmers)
        entropy = -sum((c / total) * np.log2(c / total) for c in counts.values())
        results.append(entropy)
    return np.array(results, dtype=float).reshape(-1, 1)


def compute_orf_stats(sequences):
    """
    ORF statistics: (n_orfs, orf_density [/kb], orf_length_ratio).
    Returns array of shape (n, 3).
    """
    results = []
    for seq in sequences:
        seq_upper = str(seq).upper()
        seq_len = len(seq_upper)
        if seq_len == 0:
            results.append((0.0, 0.0, 0.0))
            continue
        orfs = orfipy_core.orfs(seq_upper, minlen=100, starts=['ATG'],
                                stops=['TAA', 'TAG', 'TGA'])
        n_orfs = len(orfs)
        orf_density = n_orfs / seq_len * 1000
        if n_orfs >= 2:
            lens = sorted([s - st for st, s, _, _ in orfs], reverse=True)
            orf_length_ratio = lens[0] / lens[1] if lens[1] > 0 else 0.0
        else:
            orf_length_ratio = 0.0
        results.append((float(n_orfs), orf_density, orf_length_ratio))
    return np.array(results, dtype=float)


def compute_enc(sequences):
    """
    Effective Number of Codons (Wright 1990).
    Scale 20 (max bias) to 61 (no bias). Default 61.0 for ORF < 20 codons.
    """
    FAMILIES_2FOLD = [['TTT','TTC'],['TAT','TAC'],['CAT','CAC'],['CAA','CAG'],
                      ['AAT','AAC'],['AAA','AAG'],['GAT','GAC'],['GAA','GAG'],
                      ['TGT','TGC']]
    FAMILIES_3FOLD = [['ATT','ATC','ATA']]
    FAMILIES_4FOLD = [['GTT','GTC','GTA','GTG'],['CCT','CCC','CCA','CCG'],
                      ['ACT','ACC','ACA','ACG'],['GCT','GCC','GCA','GCG'],
                      ['GGT','GGC','GGA','GGG']]
    FAMILIES_6FOLD = [['TTA','TTG','CTT','CTC','CTA','CTG'],
                      ['TCT','TCC','TCA','TCG','AGT','AGC'],
                      ['CGT','CGC','CGA','CGG','AGA','AGG']]
    deg_classes = [(FAMILIES_2FOLD,9,2),(FAMILIES_3FOLD,1,3),
                   (FAMILIES_4FOLD,5,4),(FAMILIES_6FOLD,3,6)]

    def _corrected_f(codon_counts, codons):
        counts = [codon_counts.get(c, 0) for c in codons]
        n = sum(counts)
        if n <= 1:
            return None
        return (n * sum((c/n)**2 for c in counts) - 1) / (n - 1)

    results = []
    for seq in sequences:
        seq_upper = str(seq).upper()
        orfs = orfipy_core.orfs(seq_upper, minlen=100, starts=['ATG'],
                                stops=['TAA','TAG','TGA'])
        if not orfs:
            results.append(61.0)
            continue
        best = max(orfs, key=lambda x: x[1] - x[0])
        orf_seq = seq_upper[best[0]:best[1]]
        n_codons = len(orf_seq) // 3
        if n_codons < 20:
            results.append(61.0)
            continue
        codon_counts = {}
        for i in range(0, n_codons * 3, 3):
            codon = orf_seq[i:i+3]
            if all(c in 'ACGT' for c in codon):
                codon_counts[codon] = codon_counts.get(codon, 0) + 1
        enc = 2.0
        for families, n_expected, deg in deg_classes:
            f_values = [f for codons in families
                        for f in [_corrected_f(codon_counts, codons)]
                        if f is not None]
            if f_values:
                avg_f = max(sum(f_values) / len(f_values), 1e-10)
                enc += n_expected / avg_f
            else:
                enc += n_expected * deg
        results.append(max(20.0, min(61.0, enc)))
    return np.array(results, dtype=float).reshape(-1, 1)


def build_hexamer_table(sequences, labels):
    """
    Build hexamer log-likelihood ratio table (CPC2 approach).
    In-frame hexamers (step=3) from longest ORF, Laplace-smoothed.
    """
    coding_counts, noncoding_counts = {}, {}
    coding_total = noncoding_total = 0
    for seq, label in zip(sequences, labels):
        seq_upper = str(seq).upper()
        orfs = orfipy_core.orfs(seq_upper, minlen=100, starts=['ATG'],
                                stops=['TAA','TAG','TGA'])
        if not orfs:
            continue
        orf_seq = seq_upper[max(orfs, key=lambda x: x[1]-x[0])[0]:
                             max(orfs, key=lambda x: x[1]-x[0])[1]]
        for i in range(0, len(orf_seq) - 5, 3):
            hexamer = orf_seq[i:i+6]
            if len(hexamer) == 6 and all(c in 'ACGT' for c in hexamer):
                if label == 0:
                    coding_counts[hexamer] = coding_counts.get(hexamer, 0) + 1
                    coding_total += 1
                else:
                    noncoding_counts[hexamer] = noncoding_counts.get(hexamer, 0) + 1
                    noncoding_total += 1
    n_possible = 4096
    hex_table = {}
    for combo in itertools_product('ACGT', repeat=6):
        hexamer = ''.join(combo)
        p_c  = (coding_counts.get(hexamer, 0)    + 1) / (coding_total    + n_possible)
        p_nc = (noncoding_counts.get(hexamer, 0) + 1) / (noncoding_total + n_possible)
        hex_table[hexamer] = np.log(p_c / p_nc)
    return hex_table


def compute_hexamer_score(sequences, hex_table):
    """Mean log-likelihood of in-frame hexamers within longest ORF."""
    results = []
    for seq in sequences:
        seq_upper = str(seq).upper()
        orfs = orfipy_core.orfs(seq_upper, minlen=100, starts=['ATG'],
                                stops=['TAA','TAG','TGA'])
        if not orfs:
            results.append(0.0)
            continue
        orf_seq = seq_upper[max(orfs, key=lambda x: x[1]-x[0])[0]:
                             max(orfs, key=lambda x: x[1]-x[0])[1]]
        scores = [hex_table[orf_seq[i:i+6]]
                  for i in range(0, len(orf_seq) - 5, 3)
                  if orf_seq[i:i+6] in hex_table]
        results.append(np.mean(scores) if scores else 0.0)
    return np.array(results, dtype=float).reshape(-1, 1)


# ── Feature Matrix Construction ───────────────────────────────────────

def create_kmer_features(df, k, normalize=True,
                         orf_coverage=False, gc_content=False,
                         fickett=False, utr_structure=False,
                         kozak=False, kmer_entropy=False,
                         kmer_entropy_orf=False, orf_stats=False,
                         enc=False):
    """
    Build feature matrix (fit + transform).
    Hexamer NOT included here — added per-fold in step 4.
    """
    seqs = df['sequence']
    seqs_k = seqs.apply(lambda s: ' '.join(get_kmers(s, k)))
    vect = CountVectorizer(ngram_range=(1, 1))
    X_kmer = vect.fit_transform(seqs_k).astype(float)
    if normalize:
        row_sums = np.array(X_kmer.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        X_kmer = X_kmer.multiply(1.0 / row_sums[:, None])
        lengths = seqs.str.len().values.astype(float).reshape(-1, 1)
        cols = [X_kmer, sparse.csr_matrix(lengths)]
        if orf_coverage:
            cols.append(sparse.csr_matrix(compute_orf_features(seqs)))
        if gc_content:
            cols.append(sparse.csr_matrix(compute_gc_content(seqs)))
        if fickett:
            cols.append(sparse.csr_matrix(compute_fickett_score(seqs)))
        if utr_structure:
            cols.append(sparse.csr_matrix(compute_utr_structure(seqs)))
        if kozak:
            cols.append(sparse.csr_matrix(compute_kozak_score(seqs)))
        if kmer_entropy:
            cols.append(sparse.csr_matrix(compute_kmer_entropy(seqs, k=3)))
        if kmer_entropy_orf:
            cols.append(sparse.csr_matrix(compute_kmer_entropy_orf(seqs, k=3)))
        if orf_stats:
            cols.append(sparse.csr_matrix(compute_orf_stats(seqs)))
        if enc:
            cols.append(sparse.csr_matrix(compute_enc(seqs)))
        X = sparse.hstack(cols, format='csr')
    else:
        X = X_kmer
    return X, vect


def transform_kmer_features(df, vect, k, normalize=True,
                             orf_coverage=False, gc_content=False,
                             fickett=False, utr_structure=False,
                             kozak=False, kmer_entropy=False,
                             kmer_entropy_orf=False, orf_stats=False,
                             enc=False):
    """Transform sequences using an already-fitted vectorizer."""
    seqs = df['sequence']
    seqs_k = seqs.apply(lambda s: ' '.join(get_kmers(s, k)))
    X_kmer = vect.transform(seqs_k).astype(float)
    if normalize:
        row_sums = np.array(X_kmer.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1
        X_kmer = X_kmer.multiply(1.0 / row_sums[:, None])
        lengths = seqs.str.len().values.astype(float).reshape(-1, 1)
        cols = [X_kmer, sparse.csr_matrix(lengths)]
        if orf_coverage:
            cols.append(sparse.csr_matrix(compute_orf_features(seqs)))
        if gc_content:
            cols.append(sparse.csr_matrix(compute_gc_content(seqs)))
        if fickett:
            cols.append(sparse.csr_matrix(compute_fickett_score(seqs)))
        if utr_structure:
            cols.append(sparse.csr_matrix(compute_utr_structure(seqs)))
        if kozak:
            cols.append(sparse.csr_matrix(compute_kozak_score(seqs)))
        if kmer_entropy:
            cols.append(sparse.csr_matrix(compute_kmer_entropy(seqs, k=3)))
        if kmer_entropy_orf:
            cols.append(sparse.csr_matrix(compute_kmer_entropy_orf(seqs, k=3)))
        if orf_stats:
            cols.append(sparse.csr_matrix(compute_orf_stats(seqs)))
        if enc:
            cols.append(sparse.csr_matrix(compute_enc(seqs)))
        X = sparse.hstack(cols, format='csr')
    else:
        X = X_kmer
    return X


# ── Metrics and Plots ─────────────────────────────────────────────────

def compute_and_print_metrics(y_true, y_pred, y_prob, label=""):
    prec = precision_score(y_true, y_pred, average=None)
    rec  = recall_score(y_true, y_pred, average=None)
    f1s  = f1_score(y_true, y_pred, average=None)
    acc  = accuracy_score(y_true, y_pred)
    f1w  = f1_score(y_true, y_pred, average='weighted')
    roc  = roc_auc_score(y_true, y_prob)
    metrics = {
        'accuracy': acc, 'f1_weighted': f1w,
        'precision_pc': prec[0], 'precision_lncRNA': prec[1],
        'recall_pc': rec[0], 'recall_lncRNA': rec[1],
        'f1_pc': f1s[0], 'f1_lncRNA': f1s[1], 'roc_auc': roc,
    }
    if label:
        print(f"  {label}")
        print(f"    Accuracy:     {acc:.3f}  F1w: {f1w:.3f}")
        print(f"    Precision PC: {prec[0]:.3f}  Precision lncRNA: {prec[1]:.3f}")
        print(f"    Recall PC:    {rec[0]:.3f}  Recall lncRNA:    {rec[1]:.3f}")
        print(f"    ROC-AUC:      {roc:.3f}")
    return metrics


def save_confusion_matrix(cm, title, filepath):
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['PC', 'lncRNA'], yticklabels=['PC', 'lncRNA'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def save_roc_curve(fpr, tpr, auc_score, title, filepath):
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(title); plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def print_feature_importances(clf, vect, top_n=20, out_dir=None):
    feature_names = list(vect.get_feature_names_out()) + [
        'length', 'orf_coverage', 'longest_orf', 'gc_content', 'fickett',
        'utr_ratio', 'kozak', 'kmer_entropy', 'kmer_entropy_orf',
        'n_orfs', 'orf_density', 'orf_length_ratio', 'enc', 'hexamer']
    importances = clf.feature_importances_
    df_imp = (pd.DataFrame({'feature': feature_names, 'importance': importances})
              .sort_values('importance', ascending=False)
              .reset_index(drop=True))
    print(f"\n  --- Top {top_n} Feature Importances ---")
    for _, row in df_imp.head(top_n).iterrows():
        bar = '#' * int(row['importance'] * 400)
        print(f"    {row['feature']:<22}  {row['importance']:.6f}  {bar}")
    if out_dir:
        df_imp.to_csv(os.path.join(out_dir, 'feature_importances.csv'), index=False)
    return df_imp


# ── Optuna Objective ──────────────────────────────────────────────────

def _optuna_objective(trial, X_train, y_train,
                      cv_folds=OPTUNA_CV_FOLDS, beta=OPTUNA_BETA,
                      min_recall_pc=OPTUNA_MIN_RECALL_PC):
    """
    Optuna objective: maximise Fβ score (β=2 default) for lncRNA class.
    Rejects degenerate solutions: accuracy < 0.80 or recall_pc < min_recall_pc.
    """
    n_estimators      = trial.suggest_int('n_estimators', 50, 400, step=10)
    max_depth         = trial.suggest_categorical('max_depth',
                            [None, 20, 30, 50, 100])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf  = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features      = trial.suggest_categorical('max_features',
                            ['sqrt', 'log2', 0.2, 0.3, 0.5])
    lncrna_weight     = trial.suggest_float('lncrna_weight', 0.5, 6.0, log=True)

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight={0: 1.0, 1: lncrna_weight},
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                         random_state=RANDOM_STATE)
    fbeta_scores, accs, recall_pc_scores = [], [], []
    for tr_idx, val_idx in cv.split(X_train, y_train):
        clf.fit(X_train[tr_idx], y_train[tr_idx])
        ypred = clf.predict(X_train[val_idx])
        fbeta_scores.append(
            fbeta_score(y_train[val_idx], ypred, beta=beta, pos_label=1))
        accs.append(accuracy_score(y_train[val_idx], ypred))
        recall_pc_scores.append(
            recall_score(y_train[val_idx], ypred, pos_label=0))

    if float(np.mean(accs)) < 0.80:
        return 0.0
    if float(np.mean(recall_pc_scores)) < min_recall_pc:
        return 0.0
    return float(np.mean(fbeta_scores))


def _save_optuna_results(study, optuna_dir):
    """Save Optuna artifacts: CSV, history plot, param importance plot."""
    os.makedirs(optuna_dir, exist_ok=True)
    trials_df = study.trials_dataframe(
        attrs=('number', 'value', 'params', 'state'))
    trials_df.to_csv(os.path.join(optuna_dir, 'optuna_trials.csv'), index=False)
    best_info = {
        'best_value_fbeta_lncrna': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
    }
    with open(os.path.join(optuna_dir, 'optuna_best_params.json'), 'w') as fh:
        json.dump(best_info, fh, indent=2)
    # History plot
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    nums   = [t.number for t in completed]
    values = [t.value  for t in completed]
    running_max = [max(values[:i+1]) for i in range(len(values))]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(nums, values, alpha=0.4, s=20, label='Trial Fβ')
    ax.plot(nums, running_max, 'r-', lw=2, label='Best so far')
    ax.set_xlabel('Trial'); ax.set_ylabel(f'Fβ{OPTUNA_BETA:.0f} lncRNA')
    ax.set_title('Optuna optimisation history'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(optuna_dir, 'optuna_history.png'), dpi=150)
    plt.close()
    try:
        importances = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(8, 5))
        params_list = list(importances.keys())
        imps_list   = list(importances.values())
        ax.barh(params_list[::-1], imps_list[::-1])
        ax.set_xlabel('Importance'); ax.set_title('Hyperparameter importance')
        plt.tight_layout()
        plt.savefig(os.path.join(optuna_dir, 'optuna_param_importances.png'),
                    dpi=150)
        plt.close()
    except Exception:
        pass
    print(f"    Best Fβ{OPTUNA_BETA:.0f}: {study.best_value:.4f}")
    print(f"    Best params: {study.best_params}")
    print(f"    Artifacts saved to: {optuna_dir}")


def load_best_params(results_dir):
    """Load best Optuna params from step 2 output; fall back to paper defaults."""
    params_path = os.path.join(
        results_dir, 'step2_optuna', 'optuna_best_params.json')
    if os.path.isfile(params_path):
        with open(params_path) as fh:
            data = json.load(fh)
        print(f"  Loaded Optuna params from: {params_path}")
        return data['best_params']
    print(f"  Optuna params not found at {params_path}")
    print(f"  Using paper defaults: {BEST_PARAMS_DEFAULT}")
    return BEST_PARAMS_DEFAULT.copy()


# ── Step 1: Model Selection ──────────────────────────────────────────

def step1_model_selection(lnc_bins, pc_bins, test_df, lnc_excl, pc_excl):
    """
    5-fold CV comparing RF, SGD-SVM, LogReg, NB on 5k sequences/class,
    k=3 normalized k-mer features + 11 biological features (no hexamer).
    """
    print("\n" + "=" * 70)
    print("STEP 1: Model Selection")
    print("=" * 70)

    out_dir = os.path.join(RESULTS_DIR, 'step1_model_selection')
    os.makedirs(out_dir, exist_ok=True)

    df_lnc, _ = sample_from_bins(lnc_bins, 5000, label=1,
                                  random_state=RANDOM_STATE,
                                  exclude_indices=lnc_excl)
    df_pc, _  = sample_from_bins(pc_bins,  5000, label=0,
                                  random_state=RANDOM_STATE,
                                  exclude_indices=pc_excl)
    data = pd.concat([df_lnc, df_pc], ignore_index=True)
    print(f"  Dataset: {len(df_lnc)} lncRNA + {len(df_pc)} PC")

    X, _ = create_kmer_features(data, k=K, normalize=True, **BIO_FLAGS)
    y = data['label'].values
    print(f"  Features: {X.shape[1]} (k={K} + 11 bio features)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {
        'RF':     RandomForestClassifier(
                      n_estimators=90, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        'SVM':    SGDClassifier(
                      loss='hinge', max_iter=5000,
                      random_state=RANDOM_STATE, n_jobs=N_JOBS),
        'LogReg': LogisticRegression(
                      max_iter=5000, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        'NB':     MultinomialNB(),
    }

    results = []
    for name, model in models.items():
        print(f"\n  Evaluating {name}...")
        t0 = time.time()
        # NB requires non-negative raw counts only
        if isinstance(model, MultinomialNB):
            seqs_k = data['sequence'].apply(lambda s: ' '.join(get_kmers(s, K)))
            vect_nb = CountVectorizer(ngram_range=(1, 1))
            X_nb = vect_nb.fit_transform(seqs_k)
            y_pred  = cross_val_predict(model, X_nb, y, cv=cv, n_jobs=N_JOBS)
            y_score = None
            roc_val = float('nan')
        elif hasattr(model, 'predict_proba'):
            y_pred  = cross_val_predict(model, X, y, cv=cv, n_jobs=N_JOBS)
            y_score = cross_val_predict(model, X, y, cv=cv,
                                         method='predict_proba', n_jobs=N_JOBS)[:, 1]
            roc_val = roc_auc_score(y, y_score)
        else:
            y_pred  = cross_val_predict(model, X, y, cv=cv, n_jobs=N_JOBS)
            y_score = cross_val_predict(model, X, y, cv=cv,
                                         method='decision_function', n_jobs=N_JOBS)
            roc_val = roc_auc_score(y, y_score)

        prec = precision_score(y, y_pred, average=None)
        rec  = recall_score(y, y_pred, average=None)
        f1s  = f1_score(y, y_pred, average=None)
        acc  = accuracy_score(y, y_pred)
        f1w  = f1_score(y, y_pred, average='weighted')
        elapsed = time.time() - t0

        results.append({
            'Model': name,
            'Accuracy': round(acc, 3), 'F1_weighted': round(f1w, 3),
            'Precision_PC': round(prec[0], 3), 'Precision_lncRNA': round(prec[1], 3),
            'Recall_PC': round(rec[0], 3), 'Recall_lncRNA': round(rec[1], 3),
            'F1_PC': round(f1s[0], 3), 'F1_lncRNA': round(f1s[1], 3),
            'ROC_AUC': round(roc_val, 3) if not np.isnan(roc_val) else 'N/A',
            'Time_s': round(elapsed, 1),
        })
        cm = confusion_matrix(y, y_pred)
        save_confusion_matrix(cm, f'CM: {name} (k={K})',
                              os.path.join(out_dir, f'cm_{name}.png'))
        if y_score is not None and not np.isnan(roc_val):
            fpr, tpr, _ = roc_curve(y, y_score)
            save_roc_curve(fpr, tpr, roc_val, f'ROC: {name}',
                           os.path.join(out_dir, f'roc_{name}.png'))
        print(f"    Acc={acc:.3f}  Recall_lncRNA={rec[1]:.3f}  "
              f"ROC-AUC={roc_val:.3f}  ({elapsed:.1f}s)")

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, 'model_comparison_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to {csv_path}")
    print(df.to_string(index=False))
    return df


# ── (internal) Optuna Hyperparameter Tuning ──────────────────────────

def step2_optuna_tuning(lnc_bins, pc_bins, lnc_excl, pc_excl):
    """
    Optuna hyperparameter optimisation: 200 trials, Fβ2 lncRNA objective,
    3-fold CV during search, 60k sequences/class, config_both,
    k=3 + 11 bio features (hexamer on full training set for Optuna speed).
    """
    print("\n" + "=" * 70)
    print("Optuna Hyperparameter Tuning")
    print("=" * 70)

    if not _OPTUNA_AVAILABLE:
        print("  ERROR: optuna not installed. Run: pip install optuna")
        return None

    out_dir = os.path.join(RESULTS_DIR, 'step2_optuna')
    os.makedirs(out_dir, exist_ok=True)

    print(f"  Loading {TRAIN_SEQ_LIMIT:,} seqs/class...")
    df_lnc, _ = sample_from_bins(lnc_bins, TRAIN_SEQ_LIMIT, label=1,
                                  random_state=RANDOM_STATE,
                                  exclude_indices=lnc_excl)
    df_pc, _  = sample_from_bins(pc_bins,  TRAIN_SEQ_LIMIT, label=0,
                                  random_state=RANDOM_STATE,
                                  exclude_indices=pc_excl)
    train_data = pd.concat([df_lnc, df_pc], ignore_index=True)

    print(f"  Building features (k={K} + 11 bio features)...")
    X_base, _ = create_kmer_features(train_data, k=K, normalize=True, **BIO_FLAGS)
    y_train = train_data['label'].values

    # Add hexamer on full set (slight leakage for Optuna speed; ranking preserved)
    print("  Adding hexamer on full training set...")
    hex_table = build_hexamer_table(train_data['sequence'].values, y_train)
    hex_scores = compute_hexamer_score(train_data['sequence'], hex_table)
    X_optuna = sparse.hstack([X_base, sparse.csr_matrix(hex_scores)], format='csr')
    X_dense = X_optuna.toarray()
    print(f"  Feature matrix: {X_dense.shape}")

    print(f"\n  Running Optuna ({OPTUNA_N_TRIALS} trials, "
          f"F{OPTUNA_BETA:.0f} lncRNA objective)...")
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: _optuna_objective(
            trial, X_dense, y_train,
            cv_folds=OPTUNA_CV_FOLDS, beta=OPTUNA_BETA,
            min_recall_pc=OPTUNA_MIN_RECALL_PC),
        n_trials=OPTUNA_N_TRIALS,
        show_progress_bar=False,
    )
    _save_optuna_results(study, out_dir)
    return study.best_params


# ── Step 2: Sample Size Comparison ───────────────────────────────────

def step2_sample_size_comparison(lnc_bins, pc_bins, test_df,
                                  lnc_excl, pc_excl):
    """
    RF performance across 5k/15k/30k/60k sequences/class, config_both,
    k=3 + 12 bio features (hexamer per fold), 5-fold CV.
    Default RF hyperparameters (n_estimators=90).
    """
    print("\n" + "=" * 70)
    print("STEP 2: Sample Size Comparison")
    print("=" * 70)

    out_dir = os.path.join(RESULTS_DIR, 'step2_sample_size')
    os.makedirs(out_dir, exist_ok=True)

    rf_params = dict(
        n_estimators = 90,
        random_state = RANDOM_STATE,
        n_jobs       = N_JOBS,
    )

    sample_sizes = [5_000, 15_000, 30_000, 60_000]
    summary = []

    for n in sample_sizes:
        cfg_dir = os.path.join(out_dir, f'k{K}_{n}')
        os.makedirs(cfg_dir, exist_ok=True)
        print(f"\n  Sample size: {n:,}")

        df_lnc, _ = sample_from_bins(lnc_bins, n, label=1,
                                      random_state=RANDOM_STATE,
                                      exclude_indices=lnc_excl)
        df_pc, _  = sample_from_bins(pc_bins,  n, label=0,
                                      random_state=RANDOM_STATE,
                                      exclude_indices=pc_excl)
        train_data = pd.concat([df_lnc, df_pc], ignore_index=True)
        print(f"    {len(df_lnc)} lncRNA + {len(df_pc)} PC")

        X_base, _ = create_kmer_features(
            train_data, k=K, normalize=True, **BIO_FLAGS)
        y = train_data['label'].values

        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)
        fold_metrics = []
        for fold, (tr_idx, te_idx) in enumerate(cv.split(X_base, y), 1):
            Xtr, Xte = X_base[tr_idx], X_base[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]
            # Per-fold hexamer
            hex_table = build_hexamer_table(
                train_data.iloc[tr_idx]['sequence'].values, ytr)
            hex_tr = compute_hexamer_score(
                train_data.iloc[tr_idx]['sequence'], hex_table)
            hex_te = compute_hexamer_score(
                train_data.iloc[te_idx]['sequence'], hex_table)
            Xtr = sparse.hstack([Xtr, sparse.csr_matrix(hex_tr)], format='csr')
            Xte = sparse.hstack([Xte, sparse.csr_matrix(hex_te)], format='csr')

            clf = RandomForestClassifier(**rf_params)
            clf.fit(Xtr, ytr)
            yp    = clf.predict(Xte)
            yprob = clf.predict_proba(Xte)[:, 1]

            prec = precision_score(yte, yp, average=None)
            rec  = recall_score(yte, yp, average=None)
            f1s  = f1_score(yte, yp, average=None)
            fold_metrics.append({
                'fold': fold,
                'accuracy': accuracy_score(yte, yp),
                'f1_weighted': f1_score(yte, yp, average='weighted'),
                'precision_pc': prec[0], 'precision_lncRNA': prec[1],
                'recall_pc': rec[0], 'recall_lncRNA': rec[1],
                'f1_pc': f1s[0], 'f1_lncRNA': f1s[1],
                'roc_auc': roc_auc_score(yte, yprob),
            })
            print(f"    Fold {fold}: Acc={accuracy_score(yte,yp):.3f}  "
                  f"Recall_lncRNA={rec[1]:.3f}")

        df_folds = pd.DataFrame(fold_metrics)
        mean_vals = df_folds.select_dtypes(include=[np.number]).mean()
        mean_row = mean_vals.to_dict()
        mean_row['fold'] = 'Mean'
        df_folds = pd.concat(
            [df_folds, pd.DataFrame([mean_row])], ignore_index=True)
        df_folds.round(4).to_csv(
            os.path.join(cfg_dir, 'rf_metrics_cv.csv'), index=False)

        print(f"    Mean: Acc={mean_vals['accuracy']:.3f}  "
              f"Recall_lncRNA={mean_vals['recall_lncRNA']:.3f}  "
              f"ROC-AUC={mean_vals['roc_auc']:.3f}")
        summary.append({
            'Sample_size': n, 'k': K,
            'Accuracy': round(mean_vals['accuracy'], 3),
            'F1_weighted': round(mean_vals['f1_weighted'], 3),
            'Recall_PC': round(mean_vals['recall_pc'], 3),
            'Recall_lncRNA': round(mean_vals['recall_lncRNA'], 3),
            'ROC_AUC': round(mean_vals['roc_auc'], 3),
        })

    df_summary = pd.DataFrame(summary)
    csv_path = os.path.join(out_dir, 'sample_size_comparison.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"\n  Summary saved to {csv_path}")
    print(df_summary.to_string(index=False))
    return df_summary


# ── Step 3: Final Model + Independent Test ───────────────────────────

def step3_final_model(lnc_bins, pc_bins, test_df, lnc_excl, pc_excl):
    """
    Final model training following run_optuna_experiment logic from
    RAIDNC_experiments_v5_mar16_noncodev6.py.

    Protocol:
      1. Load 60k training sequences/class (stratified, excluding test set)
      2. Build feature matrix: k=3 + 11 bio features (hexamer excluded at this stage)
      3. Build hexamer table on full training set
         (per-fold table used in CV for correctness)
      4. Load Optuna paper-default params
      5. 5-fold stratified CV with best params + per-fold hexamer
      6. Train final model on ALL training data (best params + full-set hexamer)
      7. Evaluate on independent test set
      8. Save model artefacts
    """
    print("\n" + "=" * 70)
    print("STEP 3: Final Model + Independent Test Set")
    print("=" * 70)

    out_dir = os.path.join(RESULTS_DIR, 'step3_final_model')
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ── 1. Load training data ──────────────────────────────────────────
    print(f"\n  Loading {TRAIN_SEQ_LIMIT:,} seqs/class ({FILTER_CONFIG})...")
    df_lnc, _ = sample_from_bins(lnc_bins, TRAIN_SEQ_LIMIT, label=1,
                                  random_state=RANDOM_STATE,
                                  exclude_indices=lnc_excl)
    df_pc, _  = sample_from_bins(pc_bins,  TRAIN_SEQ_LIMIT, label=0,
                                  random_state=RANDOM_STATE,
                                  exclude_indices=pc_excl)
    train_data = pd.concat([df_lnc, df_pc], ignore_index=True)
    print(f"  Training: {len(df_lnc)} lncRNA + {len(df_pc)} PC = "
          f"{len(train_data)}")

    # ── 2. Build base feature matrix ──────────────────────────────────
    print(f"\n  Building features (k={K} + 11 bio features)...")
    X_train, vect = create_kmer_features(
        train_data, k=K, normalize=True, **BIO_FLAGS)
    y_train = train_data['label'].values
    print(f"  Base feature matrix: {X_train.shape} (hexamer added per fold)")

    # ── 3. Hexamer table on full training set ─────────────────────────
    print("\n  Building hexamer table on full training set...")
    hex_table_full = build_hexamer_table(
        train_data['sequence'].values, y_train)
    hex_train_full = compute_hexamer_score(
        train_data['sequence'], hex_table_full)
    X_train_full = sparse.hstack(
        [X_train, sparse.csr_matrix(hex_train_full)], format='csr')
    print(f"  Full feature matrix: {X_train_full.shape}")

    # ── 4. Load Optuna hyperparameters ────────────────────────────────
    best_params = load_best_params(RESULTS_DIR)
    best_rf_params = dict(
        n_estimators      = best_params['n_estimators'],
        max_depth         = best_params['max_depth'],
        min_samples_split = best_params['min_samples_split'],
        min_samples_leaf  = best_params['min_samples_leaf'],
        max_features      = best_params['max_features'],
        class_weight      = {0: 1.0, 1: best_params['lncrna_weight']},
        random_state      = RANDOM_STATE,
        n_jobs            = N_JOBS,
    )
    print(f"\n  Hyperparameters: {best_params}")

    # ── 5. 5-fold CV with best params + per-fold hexamer ─────────────
    print("\n  --- 5-Fold Cross-Validation ---")
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    fold_metrics = []
    cms, all_yte, all_proba = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X_train, y_train), 1):
        Xtr, Xte = X_train[tr_idx], X_train[te_idx]
        ytr, yte = y_train[tr_idx], y_train[te_idx]

        # Per-fold hexamer (avoids data leakage)
        hex_table_fold = build_hexamer_table(
            train_data.iloc[tr_idx]['sequence'].values, ytr)
        hex_tr = compute_hexamer_score(
            train_data.iloc[tr_idx]['sequence'], hex_table_fold)
        hex_te = compute_hexamer_score(
            train_data.iloc[te_idx]['sequence'], hex_table_fold)
        Xtr = sparse.hstack([Xtr, sparse.csr_matrix(hex_tr)], format='csr')
        Xte = sparse.hstack([Xte, sparse.csr_matrix(hex_te)], format='csr')

        clf = RandomForestClassifier(**best_rf_params)
        clf.fit(Xtr, ytr)
        yp    = clf.predict(Xte)
        yprob = clf.predict_proba(Xte)[:, 1]

        cm = confusion_matrix(yte, yp)
        cms.append(cm)
        all_yte.extend(yte)
        all_proba.extend(yprob)

        roc_auc_val = roc_auc_score(yte, yprob)
        fpr, tpr, _ = roc_curve(yte, yprob)
        save_roc_curve(fpr, tpr, roc_auc_val,
                       f'ROC Fold {fold} (k={K})',
                       os.path.join(out_dir, f'roc_fold{fold}.png'))
        save_confusion_matrix(cm, f'CM Fold {fold} (k={K})',
                              os.path.join(out_dir, f'cm_fold{fold}.png'))

        prec = precision_score(yte, yp, average=None)
        rec  = recall_score(yte, yp, average=None)
        f1s  = f1_score(yte, yp, average=None)
        fold_metrics.append({
            'fold': fold,
            'accuracy': accuracy_score(yte, yp),
            'f1_weighted': f1_score(yte, yp, average='weighted'),
            'precision_pc': prec[0], 'precision_lncRNA': prec[1],
            'recall_pc': rec[0], 'recall_lncRNA': rec[1],
            'f1_pc': f1s[0], 'f1_lncRNA': f1s[1],
            'roc_auc': roc_auc_val,
        })
        print(f"    Fold {fold}: Acc={accuracy_score(yte,yp):.3f}  "
              f"Recall_lncRNA={rec[1]:.3f}  ROC-AUC={roc_auc_val:.3f}")
        print(classification_report(yte, yp, target_names=['PC', 'lncRNA']))

    # Aggregated CV
    cm_agg = np.sum(cms, axis=0)
    save_confusion_matrix(cm_agg, 'Aggregated CV CM',
                          os.path.join(out_dir, 'cm_aggregated.png'))
    all_yte_arr   = np.array(all_yte)
    all_proba_arr = np.array(all_proba)
    agg_auc = roc_auc_score(all_yte_arr, all_proba_arr)
    fpr, tpr, _ = roc_curve(all_yte_arr, all_proba_arr)
    save_roc_curve(fpr, tpr, agg_auc, 'Aggregated CV ROC',
                   os.path.join(out_dir, 'roc_aggregated.png'))

    df_cv = pd.DataFrame(fold_metrics)
    mean_vals = df_cv.select_dtypes(include=[np.number]).mean()
    mean_row = mean_vals.to_dict()
    mean_row['fold'] = 'Mean'
    df_cv = pd.concat([df_cv, pd.DataFrame([mean_row])], ignore_index=True)
    df_cv.round(4).to_csv(
        os.path.join(out_dir, 'rf_metrics_cv.csv'), index=False)

    compute_and_print_metrics(all_yte_arr,
                               (all_proba_arr >= 0.5).astype(int),
                               all_proba_arr,
                               label="Mean CV metrics:")
    print(f"  Aggregated CV ROC-AUC: {agg_auc:.3f}")

    # ── 6. Train final model on ALL training data ─────────────────────
    print("\n  --- Training Final Model on Full Training Set ---")
    clf_final = RandomForestClassifier(**best_rf_params)
    clf_final.fit(X_train_full, y_train)

    # ── 7. Save model artefacts ────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf_final,    MODEL_FILE)
    joblib.dump(vect,         VECT_FILE)
    joblib.dump(hex_table_full, HEX_FILE)
    # Copy to step results folder
    joblib.dump(clf_final,    os.path.join(out_dir, 'rf_model_ncv6_ncv6.joblib'))
    joblib.dump(vect,         os.path.join(out_dir, 'kmer_vectorizer_ncv6_ncv6.joblib'))
    joblib.dump(hex_table_full, os.path.join(out_dir, 'hexamer_table_ncv6_ncv6.joblib'))
    print(f"\n  Model saved to:      {MODEL_FILE}")
    print(f"  Vectorizer saved to: {VECT_FILE}")
    print(f"  Hexamer table saved: {HEX_FILE}")

    # Feature importances
    print_feature_importances(clf_final, vect, top_n=20, out_dir=out_dir)

    # ── 8. Independent test set evaluation ────────────────────────────
    print("\n  --- Independent Test Set Evaluation ---")
    X_test_base = transform_kmer_features(
        test_df, vect, k=K, normalize=True, **BIO_FLAGS)
    hex_test = compute_hexamer_score(test_df['sequence'], hex_table_full)
    X_test = sparse.hstack(
        [X_test_base, sparse.csr_matrix(hex_test)], format='csr')

    y_test = test_df['label'].values
    y_pred = clf_final.predict(X_test)
    y_prob = clf_final.predict_proba(X_test)[:, 1]

    ind_metrics = compute_and_print_metrics(
        y_test, y_pred, y_prob, label="INDEPENDENT TEST SET:")
    print(classification_report(y_test, y_pred, target_names=['PC', 'lncRNA']))

    cm_test = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm_test, 'Independent Test CM',
                          os.path.join(out_dir, 'cm_independent.png'))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    save_roc_curve(fpr, tpr, roc_auc_score(y_test, y_prob),
                   'Independent Test ROC',
                   os.path.join(out_dir, 'roc_independent.png'))

    pd.DataFrame([ind_metrics]).round(4).to_csv(
        os.path.join(out_dir, 'rf_metrics_independent.csv'), index=False)

    elapsed = time.time() - t0
    print(f"\n  Step 3 completed in {elapsed / 60:.1f} minutes")
    return df_cv, pd.DataFrame([ind_metrics])


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAIDNC reproducibility pipeline — NONCODE v6, k=3, "
                    "12 bio features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  Model selection
  2  Sample-size comparison
  3  Final model + independent test
        """)
    parser.add_argument(
        '--step', type=int, nargs='+', choices=[1, 2, 3],
        help='Run only specific step(s). Default: all steps 1-3.')
    args = parser.parse_args()
    steps = args.step or [1, 2, 3]

    for fasta, desc in [(FASTA_LNC, 'lncRNA (NONCODE v6)'), (FASTA_PC, 'PC (GENCODE v46)')]:
        if not os.path.isfile(fasta):
            print(f"ERROR: {desc} FASTA not found:\n  {fasta}")
            sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    t_start = time.time()
    print("=" * 70)
    print("RAIDNC Reproducibility Pipeline ncv6 (NONCODE v6 + GENCODE v46)")
    print("=" * 70)
    print(f"  lncRNA FASTA : {FASTA_LNC}")
    print(f"  PC FASTA     : {FASTA_PC}")
    print(f"  k={K}, filter={FILTER_CONFIG}, train={TRAIN_SEQ_LIMIT:,}/class")
    print(f"  Steps        : {steps}")
    print(f"  Results      : {RESULTS_DIR}")

    # ── Load and bin data (shared across all steps) ──
    print("\nLoading lncRNA sequences (NONCODE v6)...")
    lnc_bins = load_and_bin(FASTA_LNC, label=1, n_bins=N_BINS)
    lnc_total = sum(len(b) for b in lnc_bins)
    print(f"  {lnc_total:,} sequences in {N_BINS} bins")

    print(f"\nLoading PC sequences ({FILTER_CONFIG})...")
    pc_bins = load_and_bin(FASTA_PC, label=0, n_bins=N_BINS,
                            filter_config=FILTER_CONFIG)
    pc_total = sum(len(b) for b in pc_bins)
    print(f"  {pc_total:,} sequences in {N_BINS} bins")

    # ── Sample independent test set ──
    test_df, lnc_excl, pc_excl = sample_test_set(lnc_bins, pc_bins)

    # ── Run steps ──
    if 1 in steps:
        step1_model_selection(lnc_bins, pc_bins, test_df, lnc_excl, pc_excl)
    if 2 in steps:
        step2_sample_size_comparison(lnc_bins, pc_bins, test_df,
                                     lnc_excl, pc_excl)
    if 3 in steps:
        step3_final_model(lnc_bins, pc_bins, test_df, lnc_excl, pc_excl)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"All done!  Total time: {elapsed / 60:.1f} minutes")
    print(f"Results in: {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
