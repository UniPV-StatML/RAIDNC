#!/usr/bin/env python3
"""
RAIDNC v2 -- Random Forest Assisted Identification and Discrimination
             of Non-Coding transcripts

Predict transcript type (Protein_Coding or lncRNA) from FASTA files.

Usage examples:
    python RAIDNC.py --fasta input.fa
    python RAIDNC.py --fasta input.fa --output result.tsv -t 8
    python RAIDNC.py --batch_dir ./fastas/ --output batch_out.tsv
    python RAIDNC.py --model_dir ../model/tuned --fasta input.fa
    python RAIDNC.py --fasta input.fa --threshold 0.5     # balanced operating point

Output:
    TSV with columns:
      Transcript_ID, Prediction, Prob_Protein_Coding, Prob_lncRNA, FASTA_File

The default decision threshold is read from model_config.json; it was chosen on
out-of-fold training predictions so that lncRNA recall is 0.981 (paper operating point).
"""
import os
import sys
import json
import time
import argparse
from glob import glob
from itertools import islice
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import joblib
from Bio import SeqIO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import raidnc_features as RF  # noqa: E402

DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'model', 'compact')
CLASS_MAP = {0: 'Protein_Coding', 1: 'lncRNA'}
COLUMNS = ['Transcript_ID', 'Prediction', 'Prob_Protein_Coding', 'Prob_lncRNA', 'FASTA_File']


def load_model(model_dir):
    need = ['rf_model.joblib', 'hexamer_table.joblib', 'model_config.json']
    missing = [f for f in need if not os.path.isfile(os.path.join(model_dir, f))]
    if missing:
        sys.exit(f'ERROR: missing in {model_dir}: {", ".join(missing)}')
    cfg = json.load(open(os.path.join(model_dir, 'model_config.json')))
    if cfg.get('feature_names') != RF.FEATURE_NAMES:
        sys.exit('ERROR: model feature set does not match raidnc_features.py')
    clf = joblib.load(os.path.join(model_dir, 'rf_model.joblib'))
    clf.n_jobs = 1
    tab = joblib.load(os.path.join(model_dir, 'hexamer_table.joblib'))
    return clf, tab, cfg


def read_chunks(path, size):
    it = SeqIO.parse(path, 'fasta')
    while True:
        recs = list(islice(it, size))
        if not recs:
            return
        yield [r.id for r in recs], [str(r.seq) for r in recs]


def predict_file(path, clf, tab, threshold, pool, chunk=2000):
    out = []
    for ids, seqs in read_chunks(path, chunk):
        sub = max(1, len(seqs) // (pool._processes * 4) if pool else len(seqs))
        parts = [seqs[i:i + sub] for i in range(0, len(seqs), sub)]
        res = pool.map(RF.sequence_features_batch, parts) if pool else [RF.sequence_features_batch(seqs)]
        base = np.vstack([r[0] for r in res])
        orfs = sum((r[1] for r in res), [])
        prob = clf.predict_proba(RF.full_matrix(base, orfs, tab))
        pred = (prob[:, 1] >= threshold).astype(int)
        out.append(pd.DataFrame({
            'Transcript_ID': ids, 'Prediction': [CLASS_MAP[p] for p in pred],
            'Prob_Protein_Coding': prob[:, 0], 'Prob_lncRNA': prob[:, 1],
            'FASTA_File': os.path.basename(path)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLUMNS)


def main():
    ap = argparse.ArgumentParser(description='RAIDNC v2: predict Protein_Coding vs lncRNA from FASTA.',
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--model_dir', default=DEFAULT_MODEL_DIR)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--fasta', help='single FASTA file')
    g.add_argument('--batch_dir', help='directory of FASTA files (*.fa, *.fasta, *.fna)')
    ap.add_argument('--output', default='predictions.tsv')
    ap.add_argument('--threads', '-t', type=int, default=None, help='worker processes (default: all cores)')
    ap.add_argument('--threshold', type=float, default=None,
                    help='lncRNA probability cutoff (default: value stored in model_config.json)')
    args = ap.parse_args()

    clf, tab, cfg = load_model(args.model_dir)
    thr = cfg['threshold'] if args.threshold is None else args.threshold
    files = [args.fasta] if args.fasta else sorted(
        sum((glob(os.path.join(args.batch_dir, e)) for e in ('*.fa', '*.fasta', '*.fna')), []))
    if not files:
        sys.exit(f'No FASTA files found in {args.batch_dir}')
    workers = args.threads or cpu_count()
    print(f'RAIDNC v2 | model={cfg["preset"]} | threshold={thr:.4f} | files={len(files)} | workers={workers}')

    t0 = time.time()
    pool = Pool(workers) if workers > 1 else None
    try:
        dfs = []
        for f in files:
            try:
                dfs.append(predict_file(f, clf, tab, thr, pool))
            except Exception as e:  # keep batch mode going
                print(f'[WARNING] Skipping {f}: {e}')
    finally:
        if pool:
            pool.close()
            pool.join()
    res = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=COLUMNS)
    res.to_csv(args.output, sep='\t', index=False)
    n_lnc = int((res['Prediction'] == 'lncRNA').sum())
    print(f'{len(res)} transcripts -> {n_lnc} lncRNA, {len(res) - n_lnc} Protein_Coding')
    print(f'Predictions saved to {args.output}  ({time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
