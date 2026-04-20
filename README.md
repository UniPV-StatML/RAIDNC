# 🧬 RAIDNC

### **R**andom Forest **A**ssisted **I**dentification and **D**iscrimination of **N**on-**C**oding transcripts

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18632923-blue)](https://doi.org/10.5281/zenodo.18632923)

> 🔬 Bonomo A., Rigano G., Giosa D., Giuffre L., Romeo O., Figini S., Ballante E.
> *"RAIDNC: Random Forest Assisted Identification and Discrimination of Non-Coding transcripts"* — manuscript in preparation.

RAIDNC is an **alignment-free** Random Forest classifier that discriminates long non-coding RNAs (lncRNAs) from protein-coding (PC) transcripts. It combines normalized k-mer frequency profiles with a rich set of biological sequence features — no reference genome or sequence alignment required.

---

## ✨ Highlights

- 🚀 **Alignment-free** — works directly on raw FASTA sequences
- 🌿 **Biologically-informed** — ORF statistics, GC content, Kozak consensus, hexamer log-likelihood, ENC, and more
- 🎯 **High lncRNA sensitivity** with balanced precision across both classes
- ⚙️ **Flexible** — single-file or batch prediction, exportable model, full reproducibility pipeline
- 🔁 **Reproducible** — systematic experiment framework with `--resume`, `--rfecv`, `--optuna`

---

## 📦 Installation

Python >= 3.8 is required.

**With pip:**
```bash
pip install -r requirements.txt
```

**With Conda:**
```bash
conda create -n raidnc python=3.10
conda activate raidnc
conda install numpy pandas scikit-learn scipy matplotlib seaborn joblib tqdm -c defaults
conda install biopython orfipy -c conda-forge
```

---

## ⚡ Quick Start

### Classify new transcripts

```bash
python scripts/RAIDNC.py --fasta input.fa --output predictions.tsv
```

Using a custom model directory:

```bash
python scripts/RAIDNC.py --model_dir model/config_both/k3/60000/ --fasta input.fa --output predictions.tsv
```

The output is a TSV file with columns: `Transcript_ID`, `Prediction` (`lncRNA` / `Protein_Coding`), `Prob_Protein_Coding`, `Prob_lncRNA`, `FASTA_File`.

---

## 🔁 Reproducibility Pipeline

`RAIDNC_reproduce_ncv6.py` reproduces the experimental steps described in the paper.

```bash
python scripts/RAIDNC_reproduce_ncv6.py           # all steps
python scripts/RAIDNC_reproduce_ncv6.py --step 1
python scripts/RAIDNC_reproduce_ncv6.py --step 1 2 3
```

| Step | Description |
|:----:|-------------|
| 1️⃣ | Classifier comparison (SGD-SVM, Logistic Regression, Naive Bayes, Random Forest) — 5-fold CV, k=3 |
| 2️⃣ | Sample-size study (5k / 15k / 30k / 60k seq/class) + Optuna hyperparameter tuning (200 trials, Fβ=2) |
| 3️⃣ | Final model — 60k seq/class, Optuna params, 5-fold CV + independent test (3k seq/class) |

Results are written to `results_reproduce_ncv6/`.

---


## 🗂️ Repository Structure

```
RAIDNC_FINAL/
├── 📄 requirements.txt
├── 🐍 scripts/
│   ├── RAIDNC.py                        # Inference script
│   └── RAIDNC_reproduce_ncv6.py         # Reproducibility pipeline
├── 🤖 model/
│   └── config_both/k3/60000/
│       ├── rf_model.joblib              # Pre-trained Random Forest
│       ├── kmer_vectorizer.joblib       # Fitted CountVectorizer
│       ├── hexamer_table.joblib         # Hexamer log-likelihood table
│       └── model_config.json           # Feature configuration (78 features)
└── 🧬 data/
    ├── training/
    │   ├── cdhit09_noncodev6_lncRNA_human.fa
    │   └── cdhit09_gencode_pc_human_unique.fa
    ├── test_set/                        # Independent test set
    └── benchmark_fasta/                 # Multi-species + cross-dataset FASTA for benchmarking
```

---

## 🗄️ Data Sources

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
