# Definitions: tasks, splits, and what target data enter training

Everything below is read from the code and the configs. No numbers here come from a model run. Line numbers refer to
the files as of 2026-11-19.

## Shared pipeline (all four domains)

- **Order.** Load the series, apply the per-domain transform if there is one, split chronologically 70 / 15 / 15
  (train / val / test), z-score every split with **train-split** mean and std, then cut sliding windows inside each
  split separately. Code: `src/data/windowing.py:load_domain_series` (transform), `src/data/prepare.py:chronological_split`,
  `src/data/prepare.py:normalize` (mean/std from `train` only, or from a given `stats`), `src/data/prepare.py:prepare_domain`,
  `src/data/windowing.py:make_windows` (stride 1; windows never cross a split boundary).
- **Source and target views.** `src/experiment.py:prepare_pair` builds three views. `src` is the source series with
  source-train stats. `tgt_src` is the target series with **source-train** stats. `tgt_tgt` is the target series with
  **target-train** stats. An assertion checks that both target views decode to the same original-unit test targets
  (`src/experiment.py:137`).
- **Metrics.** MAE and RMSE are computed in original units after de-normalising both truth and prediction with the
  same stats that normalised the input (`src/evaluate.py:evaluate_model`, `src/evaluate.py:denorm`, `src/utils/metrics.py:evaluate_all`).
- **Configs.** `src/run_experiments.py:DOMAIN_CONFIGS` (window sizes, epochs, batch size, source and target CSVs).
- **Backbone for the canonical table.** TCN (`src/models/backbone.py:TCNForecaster`). Its `get_features()` is the
  32-channel output at the last time step.

## Per-domain task

| domain | task in one sentence | source series | target series | window counts (train / val / test), source and target |
|---|---|---|---|---|
| energy | Given the previous 168 hourly loads (MW) of COMED, predict the next 24 hourly loads. | AEP hourly load, `data/processed/energy/aep_clean.csv` (`load_mw`) | COMED hourly load, `data/processed/energy/comed_clean.csv` (`load_mw`) | 46,361 / 9,785 / 9,785 for both |
| climate | Given the previous 30 daily mean temperatures (°C) of Delhi, predict the next 5 days. | Shimla daily mean temperature, `data/processed/climate/shimla_clean.csv` (`temp`) | Delhi daily mean temperature, `data/processed/climate/delhi_clean.csv` (`temp`) | 2,969 / 610 / 610 for both |
| finance | Given the previous 30 daily log-returns of Nifty Pharma, predict the next 5 daily log-returns. | Nifty IT daily log-return, `data/processed/finance/nifty_it_clean.csv` (`log_return`) | Nifty Pharma daily log-return, `data/processed/finance/nifty_pharma_clean.csv` (`log_return`) | source 1,993 / 400 / 401; target 1,992 / 400 / 401 |
| finance_vol | Given the previous 30 daily absolute log-returns of Nifty Pharma, predict the next 5 absolute log-returns. | Nifty IT \|log-return\| (same CSV, `transform: "abs"` applied before the split) | Nifty Pharma \|log-return\| (same CSV, same transform) | as finance |

Split dates (first day of the series / last day of train / last day of val / last day of test):
energy 2011-01-01 / 2016-04-23 / 2017-06-13 / 2018-08-03 (both series);
climate 2015-01-01 / 2023-03-22 / 2024-12-25 / 2026-09-30 (both series);
finance and finance_vol: source 2015-01-02 / 2023-03-20 / 2024-12-26 / 2026-09-30, target 2015-01-02 / 2023-03-17 / 2024-12-24 / 2026-09-30.

## What each split is used for

| split | source | target |
|---|---|---|
| train | Supervised training (inputs and labels) in every method except target-only. Its mean/std are the "source stats". The AdaRNN-style TDC periods are cut from its input windows. | Its mean/std are the "target-train stats". Its **input windows only** (no labels) are the target batches for CORAL and AdaRNN-style. Inputs **and labels** are used only by target-only. |
| val | Early stopping (MSE in normalised units, patience 3, min improvement 1e-5, best weights restored) and λ selection. Also the bake-off backbone choice. | Not used in any training or selection. Only the post-training label-free MMD/CORAL gap diagnostic reads target-val **inputs** (`src/adaptation/coral.py:gap_diagnostic`, called after training in `src/run_coral.py:165-166` and `src/run_adarnn.py:239-240`). |
| test | Reported as `source_test` only. | All target-test numbers. Never used for training, stopping, λ or backbone choice. |

## What target data enter training, per method

| method (canonical row) | target data used during training | how target data are used at inference | code |
|---|---|---|---|
| source-only, source stats | **None.** | Target-test inputs z-scored with source-train stats; outputs de-normalised with source-train stats. | `src/experiment.py:run_transfer` (training `:182`, eval `:199`) |
| + target-stat normalisation ("source-only, target-train stats") | **None.** Same trained model as the row above. | Target-test inputs z-scored with target-train mean/std; outputs de-normalised with the same. | `src/experiment.py:200` |
| CORAL λ=10 (and λ=1 sensitivity) | Target-**train input windows** only, z-scored with target-train stats. One target batch is drawn per source batch for the CORAL term on `get_features()`. Labels are never touched: the loader is a `TensorDataset` of inputs only. | As the target-stats row. | `src/run_coral.py:run_coral` (`:138` passes `tgt_tgt["X_train"]`), `src/adaptation/coral.py:fit_coral` (`:203-205`, `:217-224`) |
| AdaRNN-style λ=10 | Target-**train input windows** only (target-train stats), as the 4th "period" in the pairwise CORAL term. | As the target-stats row. | `src/run_adarnn.py:run_adarnn` (`:213`), `src/adaptation/adarnn.py:fit_adarnn` (`:71-73`, `:100-105`) |
| target-only (labels, reference) | Target-**train inputs and labels** (supervised). No target val: it trains for a fixed number of epochs equal to that seed's source best epoch. | Target-train stats at both ends. | `src/experiment.py:208-219` |
| faithful AdaRNN / GRU+CORAL (novelty study, not in canonical.csv) | Target-train inputs only, as above. | Target-train stats. | `src/adarnn_full/run.py:run_one` (`:134` `Xt = tgt_tgt["X_train"]`) |

**For the unsupervised methods (source-only with either stats, CORAL, AdaRNN-style), target labels, target validation
data and target test data never enter training, early stopping or λ selection.** Early stopping and λ selection use
source val only (`src/train.py:fit`, `src/adaptation/coral.py:243-250`, `src/adaptation/adarnn.py:127-136`,
`src/run_coral.py:select`).

One precise caveat about the target-train stats. They are `train.mean()` and `train.std()` over the whole raw
target-train segment (`src/data/prepare.py:23-24`). That segment is the same set of raw values the target-train input
windows are cut from, plus its last `horizon` values (5 or 24 points), which appear in no target-train input window,
only as target-train labels. No target val or test value enters these statistics.
