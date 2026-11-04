# 🧬 RAIDNC

### **R**andom Forest **A**ssisted **I**dentification and **D**iscrimination of **N**on-**C**oding transcripts

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18632923-blue)](https://doi.org/10.5281/zenodo.18632923)

> 🔬 Bonomo A., Rigano G., Giosa D., Giuffrè L., Romeo O., Figini S., Ballante E.
> *"RAIDNC: Random Forest Assisted Identification and Discrimination of Non-Coding transcripts"*, manuscript in preparation.

RAIDNC is an **alignment-free** Random Forest classifier that separates long non-coding RNAs (lncRNAs)
from protein-coding (PC) transcripts. It combines tri-nucleotide frequencies with thirteen biological
sequence features computed on the sense strand. It needs no reference genome and no alignment.

**Version 2.0.** See [CHANGELOG.md](CHANGELOG.md) for what changed from v1 (tags `1.1` / `v1.0`).

---

## ✨ Highlights

- 🚀 **Alignment-free**: works directly on FASTA transcripts.
- 🌿 **Biologically informed**: ORF structure, stop-free stretch ratio, hexamer log-likelihood, Fickett, Kozak, ENC, k-mer entropies.
- 🎯 **Recall-first by design**: the decision threshold is calibrated so that lncRNA recall is 0.981 on out-of-fold training predictions; `--threshold 0.5` gives a more balanced operating point.
- ⚡ **Fast**: one ORF scan per sequence and multiprocessing inside each FASTA file.
- 🔁 **Reproducible**: one script retrains the released models; one script reruns the benchmark through the command line.

---

## 📦 Installation

Tested with Python 3.12 on Linux / WSL.

```bash
pip install -r requirements.txt
```

- `orfipy` builds a C extension. On Windows, use WSL, or install it from bioconda.
- Models and FASTA files are stored with **Git LFS**: install `git-lfs` before cloning, otherwise the
  `.joblib` and `.fa` files are small pointer files and loading fails.
- The released models were saved with scikit-learn 1.7.0 (pinned in `requirements.txt`).

---

## ⚡ Quick start

```bash
cd scripts
python RAIDNC.py --fasta input.fa --output predictions.tsv              # default model, all cores
python RAIDNC.py --fasta input.fa --model_dir ../model/tuned -t 8      # another model, 8 workers
python RAIDNC.py --batch_dir fastas/ --output batch.tsv                 # every *.fa / *.fasta / *.fna
python RAIDNC.py --fasta input.fa --threshold 0.5                       # balanced operating point
```

Output: a TSV with `Transcript_ID`, `Prediction` (`lncRNA` / `Protein_Coding`),
`Prob_Protein_Coding`, `Prob_lncRNA`, `FASTA_File`.

### Models

| Model | Use | PC reference set | Threshold | Size |
|---|---|---|---|---|
| `model/compact` | **default** | config_both | 0.3005 | 50 MB |
| `model/tuned` | best Optuna trial, equivalent performance | config_both | 0.2865 | 83 MB |
| `model/compact_nofilter` | same hyperparameters | config_nofilter | 0.3042 | 50 MB |

Each folder holds the Random Forest, the hexamer table and a `model_config.json` with the ordered
feature list, the hyperparameters, the threshold and the cross-validation and test metrics.

---

## 📊 Performance

Independent human test set (3,000 lncRNA + 3,000 PC, excluded from training), default model:

| Accuracy | Recall lncRNA | Recall PC | ROC-AUC | MCC |
|---|---|---|---|---|
| 0.892 | 0.979 | 0.804 | 0.973 | 0.796 |

Cross-species lncRNA recall (NONCODE v6, model trained on human data only):

| *A. thaliana* | *C. elegans* | *D. melanogaster* | *M. musculus* | *S. cerevisiae* |
|---|---|---|---|---|
| 99.46% | 92.83% | 95.45% | 98.94% | 94.55% |

Full results: `results/` (benchmarks, cross-validation, Optuna history) and `plots/`.

---

## 🔁 Reproducing

```bash
cd scripts
bash train_all.sh python3                  # retrains compact + tuned (~17 min on 28 cores)
python3 train_raidnc_v2.py --preset compact --pc_filter nofilter --out ../model/compact_nofilter
python3 benchmark_v2.py --preset compact   # test set + NONCODE v6 species through RAIDNC.py
```

`train_raidnc_v2.py` reproduces the manuscript protocol: length-stratified sampling (20 bins),
independent test set (3,000/class, seed 42), 60,000 sequences/class for training, 5-fold CV with the
hexamer table rebuilt in each fold, threshold fixed on out-of-fold predictions.

The analyses behind the manuscript (model selection, ablation, Optuna, feature importance, residual
errors, v1 vs v2 comparison) are in [`experiments/`](experiments/README.md).

---

## 🗂️ Repository structure

```
RAIDNC/
├── requirements.txt
├── CHANGELOG.md
├── scripts/
│   ├── RAIDNC.py              # command-line prediction
│   ├── raidnc_features.py     # feature extraction (79 features)
│   ├── train_raidnc_v2.py     # training protocol
│   ├── train_all.sh
│   └── benchmark_v2.py        # end-to-end benchmark
├── model/
│   ├── compact/               # default model
│   ├── tuned/
│   └── compact_nofilter/
├── results/                   # benchmarks, logs, experiment outputs, Optuna history, test set FASTA
├── plots/                     # comparison and feature-importance figures + scripts
├── experiments/               # manuscript analyses (see experiments/README.md)
└── data/
    ├── training/
    └── benchmark_fasta/
```

---

## 🗄️ Data sources

Training data is available on [Zenodo](https://doi.org/10.5281/zenodo.18632923).

| Dataset | Source | Sequences |
|---------|--------|-----------|
| 🟢 lncRNA | [NONCODE v6](http://v6.noncode.org) | 133,031 human lncRNAs |
| 🔵 Protein-coding | [GENCODE v46](https://www.gencodegenes.org/) | 66,831 human PC transcripts |

Both datasets were deduplicated at 90% sequence identity using **CD-HIT**.

---

## 📖 Citation

If you use RAIDNC in your research, please cite:

```bibtex
@article{raidnc2025,
  author  = {Bonomo, A. and Rigano, G. and Giosa, D. and Giuffre, L.
             and Romeo, O. and Figini, S. and Ballante, E.},
  title   = {RAIDNC: Random Forest Assisted Identification and
             Discrimination of Non-Coding transcripts},
  year    = {2025},
  note    = {Manuscript in preparation},
  doi     = {10.5281/zenodo.18632923}
}
```

---

## 📜 License

This project is released under the [MIT License](LICENSE).
