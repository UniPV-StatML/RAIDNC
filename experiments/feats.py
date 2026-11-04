"""Fast, single-pass RAIDNC feature extractor.

compute_bio(seq, strand='b') reproduces the 12 original bio features exactly
(strand='b' == original behaviour, including the minus-strand slicing bug);
strand='f' restricts ORF search to the sense strand.
compute_extra(seq) adds new candidate features (sense-strand ORF).
"""
import math
from itertools import product
import numpy as np
import orfipy_core

STOPS = ['TAA', 'TAG', 'TGA']
KMERS3 = [''.join(p) for p in product('acgt', repeat=3)]  # CountVectorizer vocab order (sorted, lowercase)
_LUT = np.full(256, -1, dtype=np.int64)
for i, c in enumerate(b'ACGT'):
    _LUT[c] = i
CODONS = [''.join(p) for p in product('ACGT', repeat=3)]
CODON_IDX = {c: i for i, c in enumerate(CODONS)}

# ---- original constants (copied from RAIDNC_reproduce_ncv6.py) -------------
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
        if fv:
            enc += n_exp / max(sum(fv) / len(fv), 1e-10)
        else:
            enc += n_exp * deg
    return max(20.0, min(61.0, enc))


def find_orfs(s, strand):
    return orfipy_core.orfs(s, minlen=100, starts=['ATG'], stops=STOPS, strand=strand)


def kmer3_norm(seq):
    """Normalized 3-mer frequencies (64, CountVectorizer order) over ACGT-only 3-mers."""
    a = _LUT[np.frombuffer(seq.upper().encode(), dtype=np.uint8)]
    if len(a) < 3:
        return np.zeros(64)
    codes = a[:-2] * 16 + a[1:-1] * 4 + a[2:]
    ok = (a[:-2] >= 0) & (a[1:-1] >= 0) & (a[2:] >= 0)
    c = np.bincount(codes[ok], minlength=64).astype(float)
    s = c.sum()
    return c / s if s > 0 else c


def compute_bio(seq, strand='b'):
    """12 original bio features, same order as BIO_NAMES; returns (feats, longest_orf_seq)."""
    s = seq.upper()
    L = len(s)
    if L == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 61.0], ''
    orfs = find_orfs(s, strand)
    gc = (s.count('G') + s.count('C')) / L
    fk = fickett(s) if L >= 2 else 0.0
    ent = _entropy([s[i:i + 3] for i in range(L - 2)]) if L >= 3 else 0.0
    n = len(orfs)
    if not orfs:
        return [0.0, 0.0, gc, fk, 0.5, 0.0, ent, 0.0, 0.0, 0.0, 0.0, 61.0], ''
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
    ent_orf = _entropy([orf_seq[i:i + 3] for i in range(0, len(orf_seq) - 2, 3)]) if L >= 3 else 0.0
    if n >= 2:
        lens = sorted([b - a for a, b, _, _ in orfs], reverse=True)
        olr = lens[0] / lens[1] if lens[1] > 0 else 0.0
    else:
        olr = 0.0
    return [longest / L, float(longest), gc, fk, utr, koz, ent, ent_orf,
            float(n), n / L * 1000, olr, _enc(orf_seq)], orf_seq


# ---------------------------------------------------------------- new features
EXTRA_NAMES = (['orf_fickett', 'orf_gc3', 'orf_start_rel', 'orf2_len', 'orf_rev_longest',
                'orf_fwd_rev_ratio', 'frame_maxrun_codons', 'frame_maxrun_ratio',
                'stop_density_min', 'stop_density_max', 'pep_pI', 'pep_gravy',
                'pep_instability', 'pep_aromaticity']
               + ['codon_' + c for c in CODONS])

_KD = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
       'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
       'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}


def _stop_runs(s):
    """Per frame: longest stop-free run (codons) and stop density."""
    best_run, dens = 0, []
    for f in range(3):
        cod = [s[i:i + 3] for i in range(f, len(s) - 2, 3)]
        if not cod:
            dens.append(0.0)
            continue
        run = cur = nst = 0
        for c in cod:
            if c in ('TAA', 'TAG', 'TGA'):
                nst += 1
                cur = 0
            else:
                cur += 1
                run = max(run, cur)
        best_run = max(best_run, run)
        dens.append(nst / len(cod))
    return best_run, dens


def compute_extra(seq, fwd_orf_seq=None):
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    s = seq.upper()
    L = max(len(s), 1)
    fwd = find_orfs(s, 'f')
    rev = orfipy_core.orfs(s, minlen=100, starts=['ATG'], stops=STOPS, strand='r')
    rev_longest = max((b - a for a, b, _, _ in rev), default=0)
    if fwd:
        best = max(fwd, key=lambda x: x[1] - x[0])
        orf = s[best[0]:best[1]]
        lens = sorted([b - a for a, b, _, _ in fwd], reverse=True)
        orf2 = lens[1] if len(lens) > 1 else 0
        start_rel = best[0] / L
    else:
        orf, orf2, start_rel = '', 0, -1.0
    flong = len(orf)
    ratio = flong / (flong + rev_longest) if flong + rev_longest > 0 else 0.5
    cod = [orf[i:i + 3] for i in range(0, len(orf) - 2, 3)]
    cu = np.zeros(64)
    for c in cod:
        j = CODON_IDX.get(c)
        if j is not None:
            cu[j] += 1
    if cu.sum() > 0:
        cu /= cu.sum()
    gc3 = (sum(1 for c in cod if c[2] in 'GC') / len(cod)) if cod else 0.0
    run, dens = _stop_runs(s)
    if len(orf) >= 30:
        pep = ''.join(_TRANSL.get(c, 'X') for c in cod).replace('X', '')
        try:
            pa = ProteinAnalysis(pep)
            pI = pa.isoelectric_point()
            gravy = sum(_KD.get(a, 0) for a in pep) / max(len(pep), 1)
            inst = pa.instability_index()
            aro = pa.aromaticity()
        except Exception:
            pI, gravy, inst, aro = 7.0, 0.0, 40.0, 0.0
    else:
        pI, gravy, inst, aro = 0.0, 0.0, 0.0, 0.0
    return [fickett(orf) if orf else 0.0, gc3, start_rel, float(orf2), float(rev_longest), ratio,
            float(run), run * 3 / L, min(dens), max(dens), pI, gravy, inst, aro] + list(cu)


_BASES = 'TCAG'
_AA = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
_TRANSL = {a + b + c: _AA[16 * i + 4 * j + k]
           for i, a in enumerate(_BASES) for j, b in enumerate(_BASES) for k, c in enumerate(_BASES)}


# ---------------------------------------------------------------- hexamer (fold-specific)
def orf_hexamers(orf_seq):
    return [orf_seq[i:i + 6] for i in range(0, len(orf_seq) - 5, 3)]


def build_hex_table(orf_seqs, labels):
    """Same as original build_hexamer_table but from precomputed longest-ORF sequences."""
    cod, nc = {}, {}
    ct = nt = 0
    for o, y in zip(orf_seqs, labels):
        if not o:
            continue
        for h in orf_hexamers(o):
            if len(h) == 6 and all(c in 'ACGT' for c in h):
                if y == 0:
                    cod[h] = cod.get(h, 0) + 1; ct += 1
                else:
                    nc[h] = nc.get(h, 0) + 1; nt += 1
    tab = {}
    for p in product('ACGT', repeat=6):
        h = ''.join(p)
        tab[h] = math.log(((cod.get(h, 0) + 1) / (ct + 4096)) / ((nc.get(h, 0) + 1) / (nt + 4096)))
    return tab


def hex_score(orf_seq, tab):
    if not orf_seq:
        return 0.0
    sc = [tab[h] for h in orf_hexamers(orf_seq) if h in tab]
    return float(np.mean(sc)) if sc else 0.0
