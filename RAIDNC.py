import sys
import os
import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
from scipy import sparse
import joblib
from glob import glob
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import time

KMER_SIZE = 12  # Always use 12, as in training
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

HELPME_TEXT = """
examples:
    python RAIDNC.py --fasta myseqs.fa
    python RAIDNC.py --fasta input.fa --output result.tsv
    python RAIDNC.py --batch_dir ./fastas/ --output batch_out.tsv -t 4
    python RAIDNC.py --model_dir ./other_model --fasta myseqs.fa

output:
    TSV file with Transcript_ID, Prediction, Prob_Protein_Coding, Prob_lncRNA, FASTA_File

note: normalization of k-mers is always ON and k=12 is always used.
"""

def get_kmers(sequence, k=KMER_SIZE):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)]

def fasta_to_features(fasta_path, vect):
    seqs = []
    names = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seqs.append(str(rec.seq))
        names.append(rec.id)
    seqs_k = [' '.join(get_kmers(seq)) for seq in seqs]
    X_counts = vect.transform(seqs_k).astype(float)
    # Always normalize
    row_sums = np.array(X_counts.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1
    X_feat = X_counts.multiply(1.0 / row_sums[:, None])
    lengths = np.array([len(seq) for seq in seqs]).reshape(-1,1)
    X_len = sparse.csr_matrix(lengths)
    X = sparse.hstack([X_feat, X_len], format='csr')
    return X, names

def predict_fasta(fasta_file, rf_model, vect):
    try:
        X, names = fasta_to_features(fasta_file, vect)
        preds = rf_model.predict(X)
        probs = rf_model.predict_proba(X)
        class_map = {0: "Protein_Coding", 1: "lncRNA"}
        result_df = pd.DataFrame({
            "Transcript_ID": names,
            "Prediction": [class_map[p] for p in preds],
            "Prob_Protein_Coding": probs[:,0],
            "Prob_lncRNA": probs[:,1],
            "FASTA_File": os.path.basename(fasta_file)
        })
        return result_df
    except Exception as e:
        print(f"[WARNING] Skipping {fasta_file} due to error: {e}")
        return pd.DataFrame(columns=["Transcript_ID", "Prediction", "Prob_Protein_Coding", "Prob_lncRNA", "FASTA_File"])

def check_model_files(model_dir):
    model_path = os.path.join(model_dir, 'rf_model.joblib')
    vect_path = os.path.join(model_dir, 'kmer_vectorizer.joblib')
    missing = []
    if not os.path.isfile(model_path):
        missing.append('rf_model.joblib')
    if not os.path.isfile(vect_path):
        missing.append('kmer_vectorizer.joblib')
    if missing:
        print(f"\nERROR: The following files are missing in {model_dir}: {', '.join(missing)}")
        print("Make sure to train and save your model and vectorizer before using this script.\n")
        sys.exit(1)
    return model_path, vect_path

def main():
    parser = argparse.ArgumentParser(
        description="Predict transcript type (Protein_Coding or lncRNA) from FASTA using a trained RandomForest model.",
        epilog=HELPME_TEXT,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model_dir", type=str, default=SCRIPT_DIR,
                        help=f"Path to directory with rf_model.joblib and kmer_vectorizer.joblib (default: script's own directory)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fasta", type=str, help="Single FASTA file to predict")
    group.add_argument("--batch_dir", type=str, help="Directory containing multiple FASTA files")
    parser.add_argument("--output", type=str, default="predictions.tsv", help="Output TSV file name (default: predictions.tsv)")
    parser.add_argument("--threads", "-t", type=int, default=None, help="Number of parallel processes for batch mode")
    args = parser.parse_args()

    model_path, vect_path = check_model_files(args.model_dir)
    rf_model = joblib.load(model_path)
    vect = joblib.load(vect_path)

    if args.fasta:
        start_time = time.time()
        result_df = predict_fasta(args.fasta, rf_model, vect)
        result_df.to_csv(args.output, sep='\t', index=False)
        elapsed = time.time() - start_time
        print(f"Prediction saved to {args.output}")
        print(f"Time elapsed: {elapsed:.2f} seconds")

    elif args.batch_dir:
        fasta_files = sorted(
            glob(os.path.join(args.batch_dir, "*.fa")) +
            glob(os.path.join(args.batch_dir, "*.fasta")) +
            glob(os.path.join(args.batch_dir, "*.fna"))
        )
        if not fasta_files:
            print(f"No fasta files found in {args.batch_dir}")
            sys.exit(1)
        print(f"Found {len(fasta_files)} FASTA files in {args.batch_dir}")

        workers = args.threads or min(cpu_count(), len(fasta_files))
        print(f"Running batch prediction using {workers} workers...")

        predict_partial = partial(
            predict_fasta,
            rf_model=rf_model,
            vect=vect
        )
        start_time = time.time()
        with ThreadPool(processes=workers) as pool:
            all_results = list(tqdm(
                pool.imap_unordered(predict_partial, fasta_files),
                total=len(fasta_files),
                desc="Predicting",
                file=sys.stdout,
                ncols=80
            ))

        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(args.output, sep='\t', index=False)
        elapsed = time.time() - start_time
        print(f"All batch predictions saved to {args.output}")
        print(f"Time elapsed: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
# This script is designed to be run as a standalone module.