# Experiments

Scripts behind the analyses in the manuscript. Run them from this folder
(`cd experiments`) with the packages in `../requirements.txt`.
Intermediate files go to `experiments/cache/` and `experiments/results/` (not versioned); the
published outputs are in `../results/`.

| Script | What it does | Output in `../results/` |
|---|---|---|
| `build_cache.py` | features of the training split and of the NONCODE v6 benchmarks, for every variant | (cache) |
| `run_exp.py` | 5-fold CV + test + benchmarks for a feature-set / hyperparameter variant (`fix`, `fix_maxrun`, `fix_maxrun_tuned`, `fix_maxrun_compact`, ...) | `experiments/*.json` |
| `ablation.py` | drop-one / add-one ablation of 14 candidate features (out-of-bag) | `experiments/ablation.csv` |
| `tune.py` | Optuna search (50 trials) | `experiments/optuna_fix_maxrun.json` |
| `paper_experiments.py` | model-selection CV (Figure 1) and sample-size study | `paper/model_selection.csv`, `paper/sample_size.csv` |
| `model_selection_full.py`, `model_selection_finish.py` | 80/20 hold-out over k = 3, 6, 9, 12 x regime x model, then CV of the best configurations | `paper/model_selection_holdout.csv`, `paper/model_selection_cv.csv` |
| `importance_analysis.py` | permutation importance, grouped permutation, SHAP | `importance/` |
| `feature_pruning.py` | retrained drop-one ablation of the final 79 features | `feature_pruning/` |
| `residual_errors.py` | features of the lncRNAs still misclassified in each species | `paper/residual_errors.csv` |

`common.py` holds the data loading and the length-stratified split shared by all scripts (the same
split as `scripts/train_raidnc_v2.py`).
