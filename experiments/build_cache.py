import os, time
import pandas as pd
from Bio import SeqIO
from common import REPO, build_split, all_features

BENCH = {
    'ATH': 'noncodev6/NONCODEv6_ATH.fa',
    'celegans': 'noncodev6/NONCODEv6_celegans.fa',
    'fruitfly': 'noncodev6/NONCODEv6_fruitfly.fa',
    'yeast': 'noncodev6/NONCODEv6_yeast.fa',
    'mouse': 'noncodev6/NONCODEv6_mouse.fa',
    'gencode_human_lnc': 'gencode/gencode.v46.lncRNA_transcripts.fa',
    'gencode_mouse_lnc': 'gencode/gencode.vM37.lncRNA_transcripts.fa',
}

if __name__ == '__main__':
    t = time.time()
    df = build_split()
    all_features(df['sequence'].values, 'split')
    print('split done', round(time.time() - t), flush=True)
    for name, rel in BENCH.items():
        p = os.path.join(REPO, 'data', 'benchmark_fasta', rel)
        if not os.path.exists(p):
            print('missing', p); continue
        recs = list(SeqIO.parse(p, 'fasta'))
        bdf = pd.DataFrame({'id': [r.id for r in recs], 'sequence': [str(r.seq) for r in recs]})
        bdf.to_pickle(os.path.join('cache', f'bench_{name}.pkl'))
        all_features(bdf['sequence'].values, f'bench_{name}')
        print(name, len(bdf), round(time.time() - t), flush=True)
