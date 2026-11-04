# Changelog

## 2.0

Same training data, sampling and independent test set as v1.

### Fixed
- **ORF strand.** `orfipy_core.orfs()` searches both strands by default (`strand='b'`) and returns
  forward-strand coordinates for minus-strand ORFs. v1 sliced `seq[start:stop]` as if every ORF were
  on the sense strand, so the hexamer score, ENC, Kozak score, ORF entropy and UTR ratio of those
  transcripts were computed on a region that is not the ORF. The longest ORF was on the minus strand
  for 16% of protein-coding and 49% of lncRNA training transcripts. ORFs are now searched on the
  sense strand only (NONCODE and GENCODE transcripts are strand-oriented).
- **Inference script.** `scripts/RAIDNC.py` imported a module that is not in the repository
  (`RAIDNC_experiments`) and could not run. It is replaced by a self-contained command line.
- **Reproducibility script paths.** The v1 script looked for data under `scripts/data/`. The v2
  training script uses `data/training/` and accepts `--lnc` / `--pc`.
- **Model / feature consistency.** `model_config.json` now stores the ordered feature list; the
  command line refuses a model whose features do not match the code.
- **Pinned dependencies.** v1 did not pin versions (its model was saved with scikit-learn 1.7.2);
  v2 pins every package and saves models with scikit-learn 1.7.0.

### Added
- **Feature `frame_maxrun_ratio`**: the longest run of codons without a stop codon in any of the
  three sense-strand frames, divided by transcript length (79 features in total).
- **Calibrated decision threshold**: set on out-of-fold predictions so that lncRNA recall is 0.981,
  stored with each model; `--threshold` overrides it.
- **Three released models** (`compact` default, `tuned`, `compact_nofilter`).
- **Training and benchmark scripts** that reproduce the released models and metrics end to end.
- **Optuna trial history** (`results/experiments/optuna_fix_maxrun.json`) and the manuscript
  analyses in `experiments/`.

### Changed
- **Hyperparameters**, retuned with Optuna (50 trials). The objective is PC recall at a lncRNA recall
  of 0.981 (out-of-bag) instead of F-beta 2 with class reweighting and a 0.5 cutoff.
- **Speed**: one ORF scan per sequence instead of eight, and multiprocessing inside each FASTA file.
  On 42,848 *D. melanogaster* transcripts this is 2.6x faster on one core and about 9x faster with
  the command line on 28 cores.
- **Model files** are stored compressed (50 MB instead of 110 MB).

### Results (independent human test set, default model)
| | v1 | v2 |
|---|---|---|
| Accuracy | 0.870 | 0.892 |
| Recall lncRNA | 0.981 | 0.979 |
| Recall PC | 0.760 | 0.804 |
| ROC-AUC | 0.963 | 0.973 |

Cross-species lncRNA recall (NONCODE v6): *A. thaliana* 96.47 → 99.46, *C. elegans* 90.23 → 92.83,
*D. melanogaster* 92.76 → 95.45, *M. musculus* 98.43 → 98.94, *S. cerevisiae* 94.55 → 94.55.

### Compatibility
- v1 and v2 models are not interchangeable (78 vs 79 features, different ORF definition).
- The output format of `RAIDNC.py` is unchanged. Probabilities are not directly comparable: filter on
  `Prediction`, or on the threshold in `model_config.json`, rather than on `Prob_lncRNA >= 0.5`.
