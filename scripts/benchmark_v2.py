#!/usr/bin/env python3
"""
End-to-end benchmark of RAIDNC v2 through the RAIDNC.py command-line interface.

  * rebuilds the paper's independent test set (3,000 + 3,000, seed 42) as FASTA
  * runs RAIDNC.py on it and on the five NONCODE v6 cross-species sets
  * writes results/benchmark_<preset>.csv and prints a comparison with the
    values published for RAIDNC v1, CPC2 and LncDC (paper Figure 2 / Figure 3)

Usage: python benchmark_v2.py --preset compact [--threads 28]
"""
import os, sys, subprocess, argparse
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_raidnc_v2 as T  # noqa: E402

DATA = os.path.join(HERE, '..', 'data')
RES = os.path.join(HERE, '..', 'results')
SPECIES = {  # file, published recall: RAIDNC v1 (config_both), CPC2, LncDC
    'A. thaliana':     ('NONCODEv6_ATH.fa',      0.9647, 0.9815, 0.9854),
    'C. elegans':      ('NONCODEv6_celegans.fa', 0.9023, 0.8542, 0.8716),
    'D. melanogaster': ('NONCODEv6_fruitfly.fa', 0.9276, 0.9270, 0.8926),
    'M. musculus':     ('NONCODEv6_mouse.fa',    0.9843, 0.8681, 0.9432),
    'S. cerevisiae':   ('NONCODEv6_yeast.fa',    0.9455, 0.8909, 0.9455),
}
PAPER_TEST = {'RAIDNC v1': (0.8703, 0.9810, 0.7597), 'CPC2': (0.8017, 0.9357, 0.6677),
              'LncDC': (0.8316, 0.9687, 0.6941)}  # accuracy, recall lncRNA, recall PC


def run_raidnc(fasta, out, model_dir, threads):
    cmd = [sys.executable, os.path.join(HERE, 'RAIDNC.py'), '--fasta', fasta, '--output', out,
           '--model_dir', model_dir, '-t', str(threads)]
    subprocess.run(cmd, check=True)
    return pd.read_csv(out, sep='\t')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='compact')
    ap.add_argument('--threads', type=int, default=os.cpu_count())
    ap.add_argument('--species_only', action='store_true',
                    help='skip the config_both test set (e.g. for a config_nofilter model, whose own '
                         'test set metrics are stored in model_config.json)')
    a = ap.parse_args()
    model_dir = os.path.join(HERE, '..', 'model', a.preset)
    tmp = os.path.join(RES, 'tmp'); os.makedirs(tmp, exist_ok=True)

    test_fa = os.path.join(RES, 'independent_test_set.fa')
    if not os.path.exists(test_fa):
        lnc = T.load_and_bin(os.path.join(DATA, 'training', 'cdhit09_noncodev6_lncRNA_human.fa'), 1)
        pc = T.load_and_bin(os.path.join(DATA, 'training', 'cdhit09_gencode_pc_human_unique.fa'), 0)
        s_l, _ = T.sample(lnc, T.TEST_PER_CLASS, T.SEED)
        s_p, _ = T.sample(pc, T.TEST_PER_CLASS, T.SEED)
        with open(test_fa, 'w') as fh:
            for i, s in enumerate(s_l):
                fh.write(f'>test_lnc_{i}\n{s}\n')
            for i, s in enumerate(s_p):
                fh.write(f'>test_pc_{i}\n{s}\n')

    rows = []
    d = None if a.species_only else run_raidnc(test_fa, os.path.join(tmp, 'test.tsv'), model_dir, a.threads)
    if d is not None:
        y = d['Transcript_ID'].str.startswith('test_lnc').astype(int).values
        yp = (d['Prediction'] == 'lncRNA').astype(int).values
        rows.append(dict(dataset='human independent test', n=len(d), accuracy=accuracy_score(y, yp),
                     recall_lncRNA=recall_score(y, yp), recall_pc=recall_score(y, yp, pos_label=0),
                     mcc=matthews_corrcoef(y, yp), roc_auc=roc_auc_score(y, d['Prob_lncRNA'])))
    for sp, (fn, v1, cpc2, lncdc) in SPECIES.items():
        d = run_raidnc(os.path.join(DATA, 'benchmark_fasta', 'noncodev6', fn),
                       os.path.join(tmp, fn + '.tsv'), model_dir, a.threads)
        rows.append(dict(dataset=sp, n=len(d), recall_lncRNA=(d['Prediction'] == 'lncRNA').mean(),
                         paper_RAIDNC_v1=v1, paper_CPC2=cpc2, paper_LncDC=lncdc))
    df = pd.DataFrame(rows)
    sp = df[df['dataset'].isin(list(SPECIES))]
    macro = sp['recall_lncRNA'].mean()
    micro = (sp['recall_lncRNA'] * sp['n']).sum() / sp['n'].sum()
    df.to_csv(os.path.join(RES, f'benchmark_{a.preset}.csv'), index=False)
    pd.set_option('display.width', 200)
    print(df.round(4).to_string(index=False))
    print(f'macro recall {macro:.4f} (paper: v1 0.945, CPC2 0.904, LncDC 0.928) | '
          f'micro recall {micro:.4f} (paper: v1 0.969, CPC2 0.884, LncDC 0.931)')
    print('paper test (acc, rec_lnc, rec_pc):', PAPER_TEST)


if __name__ == '__main__':
    main()
