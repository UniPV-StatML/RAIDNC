"""
RAIDNC v2 feature extraction.

Feature vector (79 columns, fixed order):
    64  normalized tri-nucleotide frequencies (aaa..ttt, ACGT-only 3-mers)
     1  transcript length
    12  biological features computed on the longest sense-strand ATG ORF (>= 100 nt):
        orf_coverage, longest_orf, gc_content, fickett, utr_ratio, kozak,
        kmer_entropy, kmer_entropy_orf, n_orfs, orf_density, orf_length_ratio, enc
     1  frame_maxrun_ratio: longest stop-free run over the 3 sense frames (nt) / length
     1  hexamer log-likelihood score of the longest ORF (table fitted on training data)

Differences from v1 (RAIDNC_reproduce_ncv6.py):
  * ORFs are searched on the sense strand only (orfipy strand='f'). v1 used orfipy's
    default strand='b' and then sliced seq[start:stop] as if every ORF were on '+',
    so minus-strand ORFs produced features from the wrong sequence/frame.
  * one orfipy call per sequence instead of eight.
  * new feature frame_maxrun_ratio.
"""
import math
from itertools import product

import numpy as np
import orfipy_core

STOPS = ['TAA', 'TAG', 'TGA']
MIN_ORF = 100

KMER_NAMES = [''.join(p) for p in product('acgt', repeat=3)]
BIO_NAMES = ['orf_coverage', 'longest_orf', 'gc_content', 'fickett', 'utr_ratio', 'kozak',
             'kmer_entropy', 'kmer_entropy_orf', 'n_orfs', 'orf_density', 'orf_length_ratio', 'enc']
FEATURE_NAMES = KMER_NAMES + ['length'] + BIO_NAMES + ['frame_maxrun_ratio', 'hexamer_score']

_LUT = np.full(256, -1, dtype=np.int64)
for _i, _c in enumerate(b'ACGT'):
    _LUT[_c] = _i

# Fickett TESTCODE tables (Fickett 1982; as in CPC2/CPAT)
_POS_PAR = [1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 0.0]
_CON_PAR = [0.33, 0.31, 0.29, 0.27, 0.25, 0.23, 0.21, 0.19, 0.17, 0.0]
_POS_PROB = {'A': [0.51, 0.55, 0.57, 0.52, 0.48, 0.58, 0.57, 0.54, 0.50, 0.36],
             'C': [0.29, 0.44, 0.55, 0.49, 0.52, 0.60, 0.60, 0.56, 0.51, 0.38],
             'G': [0.62, 0.67, 0.74, 0.65, 0.61, 0.62, 0.52, 0.41, 0.31, 0.17],
             'T': [0.51, 0.60, 0.69, 0.64, 0.62, 0.67, 0.58, 0.48, 0.39, 0.24]}
_POS_W = {'A': 0.062, 'C': 0.093, 'G': 0.205, 'T': 0.154}
_CON_PROB = {'A': [0.40, 0.55, 0.58, 0.58, 0.52, 0.48, 0.45, 0.45, 0.38, 0.19],
             'C': [0.50, 0.63, 0.59, 0.50, 0.46, 0.45, 0.47, 0.56, 0.59, 0.33],
             'G': [0.21, 0.40, 0.47, 0.50, 0.52, 0.56, 0.57, 0.52, 0.44, 0.23],
             'T': [0.30, 0.49, 0.56, 0.53, 0.48, 0.48, 0.52, 0.57, 0.60, 0.51]}
_CON_W = {'A': 0.084, 'C': 0.076, 'G': 0.081, 'T': 0.055}

# Kozak PWM log-odds (GENCODE v46, n=55,808 PC transcripts), positions -6..+5
_KOZAK = np.array([
    [-0.2854, -0.1659, 0.6542, -0.4852], [-0.3456, 0.3160, 0.2373, -0.3411],
    [-0.0523, 0.5763, 0.0536, -0.9805], [0.8632, -1.2388, 0.5537, -1.7891],
    [0.2840, 0.5275, -0.3519, -0.8432], [-0.3745, 0.8008, 0.2129, -1.6102],
    [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
    [-0.1624, -0.7221, 0.9515, -0.8202], [0.1167, 0.6584, -0.4815, -0.6870],
    [-0.6315, 0.0471, 0.5582, -0.2364]])
_KB = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
_KPOS = [0, 1, 2, 3, 4, 5, 9, 10, 11]

_ENC_CLASSES = [
    ([['TTT', 'TTC'], ['TAT', 'TAC'], ['CAT', 'CAC'], ['CAA', 'CAG'], ['AAT', 'AAC'], ['AAA', 'AAG'],
      ['GAT', 'GAC'], ['GAA', 'GAG'], ['TGT', 'TGC']], 9, 2),
    ([['ATT', 'ATC', 'ATA']], 1, 3),
    ([['GTT', 'GTC', 'GTA', 'GTG'], ['CCT', 'CCC', 'CCA', 'CCG'], ['ACT', 'ACC', 'ACA', 'ACG'],
      ['GCT', 'GCC', 'GCA', 'GCG'], ['GGT', 'GGC', 'GGA', 'GGG']], 5, 4),
    ([['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'], ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
      ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG']], 3, 6)]


def _lookup(v, params, probs, w, base):
    if v < 0:
        return 0.0
    for i, t in enumerate(params):
        if v >= t:
            return probs[base][i] * w[base]
    return 0.0


def fickett(dna):
    n = len(dna)
    if n < 2:
        return 0.0
    p0, p1, p2 = dna[::3], dna[1::3], dna[2::3]
    s = 0.0
    for b in 'ACGT':
        c = [p0.count(b), p1.count(b), p2.count(b)]
        s += _lookup(max(c) / (min(c) + 1.0), _POS_PAR, _POS_PROB, _POS_W, b)
        s += _lookup(sum(c) / n, _CON_PAR, _CON_PROB, _CON_W, b)
    return s


def _entropy(kmers):
    if not kmers:
        return 0.0
    cnt = {}
    for km in kmers:
        cnt[km] = cnt.get(km, 0) + 1
    t = len(kmers)
    return -sum((c / t) * np.log2(c / t) for c in cnt.values())


def _enc(orf_seq):
    n_cod = len(orf_seq) // 3
    if n_cod < 20:
        return 61.0
    cc = {}
    for i in range(0, n_cod * 3, 3):
        c = orf_seq[i:i + 3]
        if all(x in 'ACGT' for x in c):
            cc[c] = cc.get(c, 0) + 1
    enc = 2.0
    for fams, n_exp, deg in _ENC_CLASSES:
        fv = []
        for codons in fams:
            counts = [cc.get(c, 0) for c in codons]
            n = sum(counts)
            if n > 1:
                fv.append((n * sum((c / n) ** 2 for c in counts) - 1) / (n - 1))
        enc += n_exp / max(sum(fv) / len(fv), 1e-10) if fv else n_exp * deg
    return max(20.0, min(61.0, enc))


def kmer3_norm(s):
    a = _LUT[np.frombuffer(s.encode(), dtype=np.uint8)]
    if len(a) < 3:
        return np.zeros(64)
    codes = a[:-2] * 16 + a[1:-1] * 4 + a[2:]
    ok = (a[:-2] >= 0) & (a[1:-1] >= 0) & (a[2:] >= 0)
    c = np.bincount(codes[ok], minlength=64).astype(float)
    tot = c.sum()
    return c / tot if tot > 0 else c


def frame_maxrun_ratio(s):
    run = 0
    for f in range(3):
        cur = 0
        for i in range(f, len(s) - 2, 3):
            if s[i:i + 3] in ('TAA', 'TAG', 'TGA'):
                cur = 0
            else:
                cur += 1
                if cur > run:
                    run = cur
    return run * 3 / max(len(s), 1)


def bio_features(s):
    """12 bio features on the sense strand + longest ORF sequence. `s` must be uppercase."""
    L = len(s)
    if L == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 61.0], ''
    orfs = orfipy_core.orfs(s, minlen=MIN_ORF, starts=['ATG'], stops=STOPS, strand='f')
    gc = (s.count('G') + s.count('C')) / L
    fk = fickett(s)
    ent = _entropy([s[i:i + 3] for i in range(L - 2)]) if L >= 3 else 0.0
    if not orfs:
        return [0.0, 0.0, gc, fk, 0.5, 0.0, ent, 0.0, 0.0, 0.0, 0.0, 61.0], ''
    n = len(orfs)
    best = max(orfs, key=lambda x: x[1] - x[0])
    longest = best[1] - best[0]
    orf_seq = s[best[0]:best[1]]
    pre, post = best[0], L - best[1]
    utr = post / (pre + post) if pre + post > 0 else 0.5
    w0, w1 = best[0] - 6, best[0] + 6
    if w0 < 0 or w1 > L:
        koz = 0.0
    else:
        win = s[w0:w1]
        koz = sum(_KOZAK[p, _KB[win[p]]] for p in _KPOS if win[p] in _KB)
    ent_orf = _entropy([orf_seq[i:i + 3] for i in range(0, len(orf_seq) - 2, 3)])
    if n >= 2:
        lens = sorted([b - a for a, b, _, _ in orfs], reverse=True)
        olr = lens[0] / lens[1] if lens[1] > 0 else 0.0
    else:
        olr = 0.0
    return [longest / L, float(longest), gc, fk, utr, koz, ent, ent_orf,
            float(n), n / L * 1000, olr, _enc(orf_seq)], orf_seq


def sequence_features(seq):
    """Hexamer-independent part of the feature vector (78 values) + longest ORF sequence."""
    s = str(seq).upper()
    bio, orf_seq = bio_features(s)
    row = np.empty(78)
    row[:64] = kmer3_norm(s)
    row[64] = len(s)
    row[65:77] = bio
    row[77] = frame_maxrun_ratio(s)
    return row, orf_seq


def sequence_features_batch(seqs):
    rows, orfs = [], []
    for s in seqs:
        r, o = sequence_features(s)
        rows.append(r)
        orfs.append(o)
    return np.array(rows).reshape(-1, 78), orfs


# ---------------------------------------------------------------- hexamer (CPC2-style)
def _orf_hexamers(orf_seq):
    return [orf_seq[i:i + 6] for i in range(0, len(orf_seq) - 5, 3)]


def build_hexamer_table(orf_seqs, labels):
    """In-frame hexamer log-likelihood (coding=label 0 vs non-coding=label 1), Laplace-smoothed."""
    cod, nc = {}, {}
    ct = nt = 0
    for o, y in zip(orf_seqs, labels):
        for h in _orf_hexamers(o):
            if len(h) == 6 and all(c in 'ACGT' for c in h):
                if y == 0:
                    cod[h] = cod.get(h, 0) + 1
                    ct += 1
                else:
                    nc[h] = nc.get(h, 0) + 1
                    nt += 1
    tab = {}
    for p in product('ACGT', repeat=6):
        h = ''.join(p)
        tab[h] = math.log(((cod.get(h, 0) + 1) / (ct + 4096)) / ((nc.get(h, 0) + 1) / (nt + 4096)))
    return tab


def hexamer_score(orf_seq, tab):
    if not orf_seq:
        return 0.0
    sc = [tab[h] for h in _orf_hexamers(orf_seq) if h in tab]
    return float(np.mean(sc)) if sc else 0.0


def full_matrix(base, orfs, tab):
    """Append hexamer column -> (n, 79) float32 matrix in FEATURE_NAMES order."""
    hx = np.array([hexamer_score(o, tab) for o in orfs]).reshape(-1, 1)
    return np.hstack([base, hx]).astype(np.float32)
