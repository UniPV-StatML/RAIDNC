#!/usr/bin/env python3
"""
RAIDNC -- RandomForest Assisted Identification and Discrimination
           of Non-Coding transcripts

Predict transcript type (Protein_Coding or lncRNA) from FASTA files
using a pre-trained Random Forest model.

Usage examples:
    python RAIDNC.py --fasta input.fa
    python RAIDNC.py --fasta input.fa --output result.tsv
    python RAIDNC.py --batch_dir ./fastas/ --output batch_out.tsv -t 4
    python RAIDNC.py --model_dir ./other_model --fasta input.fa

Output:
    TSV file with columns:
      Transcript_ID, Prediction, Prob_Protein_Coding, Prob_lncRNA, FASTA_File
"""

import sys
import os
import argparse
import time
from functools import partial
from glob import glob
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import pandas as pd
import numpy as np
from Bio import SeqIO
from scipy import sparse
import joblib
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'model', 'config_both', 'k3', '60000')

MODEL_CONFIG = {
    "k":               3,
    "log_length":      False,
    "orf_coverage":    True,
    "gc_content":      True,
    "fickett":         True,
    "utr_structure":   True,
    "kozak":           True,
    "kmer_entropy":    True,
    "kmer_entropy_orf": True,
    "orf_stats":       True,
    "enc":             True,
    "hexamer":         True,
    "rfecv":           False,
}


# ── K-mer extraction ────────────────────────────────────────────────

def get_kmers(sequence, k):
    """Return all overlapping k-mers of length k."""
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


# ── Feature pipeline ────────────────────────────────────────────────

def fasta_to_features_full(fasta_path, vect, hex_table):
    """
    Read a FASTA file and build the full feature matrix matching
    the training pipeline in RAIDNC_experiments.py.
    """
    from RAIDNC_experiments import (
        compute_orf_features, compute_gc_content, compute_fickett_score,
        compute_utr_structure, compute_kozak_score, compute_kmer_entropy,
        compute_kmer_entropy_orf, compute_orf_stats, compute_enc,
        compute_hexamer_score,
    )

    seqs_list, names = [], []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seqs_list.append(str(rec.seq))
        names.append(rec.id)

    seqs = pd.Series(seqs_list)
    k = MODEL_CONFIG['k']

    # K-mer features
    seqs_k = seqs.apply(lambda s: ' '.join(get_kmers(s, k)))
    X_kmer = vect.transform(seqs_k).astype(float)
    row_sums = np.array(X_kmer.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1
    X_kmer = X_kmer.multiply(1.0 / row_sums[:, None])

    # Length (raw)
    lengths = seqs.str.len().values.astype(float).reshape(-1, 1)
    cols = [X_kmer, sparse.csr_matrix(lengths)]

    # Biological features (fixed order matching training pipeline)
    cols.append(sparse.csr_matrix(compute_orf_features(seqs)))
    cols.append(sparse.csr_matrix(compute_gc_content(seqs)))
    cols.append(sparse.csr_matrix(compute_fickett_score(seqs)))
    cols.append(sparse.csr_matrix(compute_utr_structure(seqs)))
    cols.append(sparse.csr_matrix(compute_kozak_score(seqs)))
    cols.append(sparse.csr_matrix(compute_kmer_entropy(seqs, k=3)))
    cols.append(sparse.csr_matrix(compute_kmer_entropy_orf(seqs, k=3)))
    cols.append(sparse.csr_matrix(compute_orf_stats(seqs)))
    cols.append(sparse.csr_matrix(compute_enc(seqs)))

    X = sparse.hstack(cols, format='csr')

    # Hexamer log-likelihood (appended last)
    hex_scores = compute_hexamer_score(seqs, hex_table)
    X = sparse.hstack([X, sparse.csr_matrix(hex_scores)], format='csr')

    return X, names


# ── Model loading ───────────────────────────────────────────────────

def load_model(model_dir):
    """
    Load model artifacts from model_dir.

    Returns (rf_model, vect, hex_table).
    """
    if not os.path.isdir(model_dir):
        print(f"\nERROR: Model directory not found: {model_dir}\n")
        sys.exit(1)

    model_path = os.path.join(model_dir, 'rf_model.joblib')
    vect_path = os.path.join(model_dir, 'kmer_vectorizer.joblib')

    missing = []
    if not os.path.isfile(model_path):
        missing.append('rf_model.joblib')
    if not os.path.isfile(vect_path):
        missing.append('kmer_vectorizer.joblib')
    if missing:
        print(f"\nERROR: Missing files in {model_dir}: "
              f"{', '.join(missing)}")
        print("Export a model first with --export-model in "
              "RAIDNC_experiments.py.\n")
        sys.exit(1)

    rf_model = joblib.load(model_path)
    vect = joblib.load(vect_path)

    hex_path = os.path.join(model_dir, 'hexamer_table.joblib')
    if not os.path.isfile(hex_path):
        print(f"\nERROR: Missing hexamer_table.joblib in {model_dir}\n")
        sys.exit(1)
    hex_table = joblib.load(hex_path)

    return rf_model, vect, hex_table


# ── Prediction ──────────────────────────────────────────────────────

def predict_fasta(fasta_file, rf_model, vect, hex_table, threshold=0.5):
    """Run predictions on a single FASTA file."""
    try:
        X, names = fasta_to_features_full(fasta_file, vect, hex_table)

        probs = rf_model.predict_proba(X)
        preds = (probs[:, 1] >= threshold).astype(int)
        class_map = {0: "Protein_Coding", 1: "lncRNA"}
        return pd.DataFrame({
            "Transcript_ID":      names,
            "Prediction":         [class_map[p] for p in preds],
            "Prob_Protein_Coding": probs[:, 0],
            "Prob_lncRNA":         probs[:, 1],
            "FASTA_File":          os.path.basename(fasta_file),
        })
    except Exception as e:
        print(f"[WARNING] Skipping {fasta_file}: {e}")
        return pd.DataFrame(columns=[
            "Transcript_ID", "Prediction",
            "Prob_Protein_Coding", "Prob_lncRNA", "FASTA_File"])


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAIDNC: Predict transcript type "
                    "(Protein_Coding or lncRNA) from FASTA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument(
        "--model_dir", type=str, default=DEFAULT_MODEL_DIR,
        help="Directory with model artifacts "
             f"(default: {DEFAULT_MODEL_DIR})")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fasta", type=str,
                       help="Single FASTA file to predict")
    group.add_argument("--batch_dir", type=str,
                       help="Directory of FASTA files to predict")

    parser.add_argument("--output", type=str, default="predictions.tsv",
                        help="Output TSV path (default: predictions.tsv)")
    parser.add_argument("--threads", "-t", type=int, default=None,
                        help="Parallel threads for batch mode")

    args = parser.parse_args()

    rf_model, vect, hex_table = load_model(args.model_dir)

    print(f"Model: k={MODEL_CONFIG['k']}, features: orf_coverage, gc_content, "
          f"fickett, utr_structure, kozak, kmer_entropy, kmer_entropy_orf, "
          f"orf_stats, enc, hexamer")

    threshold = 0.5

    if args.fasta:
        start = time.time()
        result_df = predict_fasta(args.fasta, rf_model, vect, hex_table,
                                  threshold=threshold)
        result_df.to_csv(args.output, sep='\t', index=False)
        print(f"Predictions saved to {args.output}")
        print(f"Time elapsed: {time.time() - start:.2f}s")

    elif args.batch_dir:
        fasta_files = sorted(
            glob(os.path.join(args.batch_dir, "*.fa")) +
            glob(os.path.join(args.batch_dir, "*.fasta")) +
            glob(os.path.join(args.batch_dir, "*.fna")))

        if not fasta_files:
            print(f"No FASTA files found in {args.batch_dir}")
            sys.exit(1)

        print(f"Found {len(fasta_files)} FASTA files in {args.batch_dir}")
        workers = args.threads or min(cpu_count(), len(fasta_files))
        print(f"Running batch prediction with {workers} workers...")

        predict_fn = partial(predict_fasta, rf_model=rf_model, vect=vect,
                             hex_table=hex_table, threshold=threshold)
        start = time.time()
        with ThreadPool(processes=workers) as pool:
            all_results = list(tqdm(
                pool.imap_unordered(predict_fn, fasta_files),
                total=len(fasta_files), desc="Predicting",
                file=sys.stdout, ncols=80))

        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(args.output, sep='\t', index=False)
        print(f"Batch predictions saved to {args.output}")
        print(f"Time elapsed: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
