# Experiments

Scripts behind the analyses in the manuscript and in `CHANGELOG.md`. Run them from this folder
(`cd experiments`) with the packages in `../requirements.txt` (plus `shap` for the SHAP analysis).
Intermediate files go to `experiments/cache/` and `experiments/results/` (not versioned); the
published outputs are in `../results/`.

| Script | What it does | Output in `../results/` |
|---|---|---|
| `build_cache.py` | features of the training split and of the NONCODE v6 benchmarks, for every variant | (cache) |
| `run_exp.py` | 5-fold CV + test + benchmarks for a variant (`orig`, `fix`, `fix_maxrun`, `fix_maxrun_tuned`, `fix_maxrun_compact`, ...) | `experiments/*.json` |
| `ablation.py` | drop-one / add-one ablation of 14 candidate features (out-of-bag) | `experiments/ablation.csv` |
| `tune.py` | Optuna search (50 trials) | `experiments/optuna_fix_maxrun.json` |
| `paper_experiments.py` | model-selection CV (Figure 1) and sample-size study | `paper/model_selection.csv`, `paper/sample_size.csv` |
| `model_selection_full.py`, `model_selection_finish.py` | 80/20 hold-out over k = 3, 6, 9, 12 x regime x model, then CV of the best configurations | `paper/model_selection_holdout.csv`, `paper/model_selection_cv.csv` |
| `importance_analysis.py` | permutation importance, grouped permutation, SHAP, v1 strand-leak check | `importance/` |
| `feature_pruning.py` | retrained drop-one ablation of the final 79 features | `feature_pruning/` |
| `residual_errors.py` | features of the lncRNAs still misclassified in each species | `paper/residual_errors.csv` |
| `diag_celegans.py` | C. elegans predictions that change between v1 and v2 | `experiments/celegans_flips.csv` |
| `verify_equiv.py`, `check_strand.py` | v2 feature code reproduces v1 exactly with `strand='b'`; orfipy strand behaviour | (console) |
| `speed_v1v2.py`, `speed.py` | timing and model size | (console) |
| `paired.py` | McNemar test v1 vs v2 on the test set | (console) |

`legacy_v1/RAIDNC_reproduce_ncv6.py` is the original v1 pipeline, kept so that v1 and v2 can be
compared on identical data (`common.py` imports its loading and sampling functions).
