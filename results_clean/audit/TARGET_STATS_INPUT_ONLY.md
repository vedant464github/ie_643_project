# Target-train statistics from input-window values only

**Question.** The pipeline computes target-train stats over the whole raw target-train segment
(`src/data/prepare.py:normalize`, `train.mean()` and `train.std()`, ddof=0). The last H values of that segment appear
only as target-train labels, never in an input window. This file recomputes the stats without those H values and
re-evaluates seed 0 of "source-only, target-train stats" with them.

- **Method.** Inference only, CPU, OMP_NUM_THREADS=2 / torch 2 threads, no training.
- **Script:** `results_clean/audit/target_stats_input_only.py`. Raw output: `results_clean/audit/target_stats_input_only.json`.

## Definition of "input-only" values

Target-train window i has inputs at positions i … i+L−1 and labels at i+L … i+L+H−1
(`src/data/windowing.py:make_windows`). The last window starts at n_train − L − H, so input positions cover
0 … n_train − H − 1. The input-only set is therefore `train_raw[: n_train − H]`.

The script asserts two things:
- This set has exactly (number of target-train windows) + L − 1 values.
- Recomputing the old stats from the full segment reproduces the pipeline's stored values.

## 1. Old vs input-only statistics

Relative difference = (new − old) / |old|.

| domain | H | values: full segment → input-only | old mean | input-only mean | rel. diff mean | old std | input-only std | rel. diff std |
|---|---|---|---|---|---|---|---|---|
| energy (MW) | 24 | 46,552 → 46,528 | 11446.5446 | 11447.7030 | +1.01e-4 | 2280.9577 | 2280.9046 | −2.33e-5 |
| climate (°C) | 5 | 3,003 → 2,998 | 24.112521 | 24.120514 | +3.31e-4 | 7.156370 | 7.159605 | +4.52e-4 |
| finance (log-return) | 5 | 2,026 → 2,021 | 3.28997e-5 | 3.33115e-5 | +1.25e-2 | 0.01314132 | 0.01315408 | +9.71e-4 |
| finance_vol (\|log-return\|) | 5 | 2,026 → 2,021 | 0.00951058 | 0.00952141 | +1.14e-3 | 0.00906886 | 0.00907600 | +7.87e-4 |

The finance mean shows a large relative difference (1.25%) only because the mean is close to zero. The absolute
change is 4.1e-7, which is 3.1e-5 of the std.

## 2. Seed-0 "source-only, target-train stats" on the target test set

- **Model:** checkpoint `experiments/final_3seed/tcn/checkpoints/<domain>_seed0_source_only.pt`.
- **Input-only column:** target-test inputs are z-scored with the input-only stats, and predictions and truth are
  de-normalised with the same stats (`src/evaluate.py:evaluate_model`).
- **Stored column:** `experiments/final_3seed/tcn/runs/<domain>_seed0.json` →
  `source_only_on_target_test.target_stats`.
- **Old-stats recompute:** recomputing with the old stats reproduces the stored values exactly in every domain.
- **Rounding:** metrics are rounded to 5 decimals by `src/utils/metrics.py:evaluate_all`, as in the stored JSONs.

| domain | metric | stored (old stats) | input-only stats | change | relative change | row seed std (canonical.csv), for scale |
|---|---|---|---|---|---|---|
| energy | MAE (MW) | 561.30774 | 561.25848 | −0.04926 | −0.0088% | 8.75 |
| energy | RMSE (MW) | 828.06134 | 828.00458 | −0.05676 | −0.0069% | 10.23 |
| climate | MAE (°C) | 1.67283 | 1.67294 | +0.00011 | +0.0066% | 0.0027 |
| climate | RMSE (°C) | 2.20048 | 2.20065 | +0.00017 | +0.0077% | 0.0094 |
| finance | MAE | 0.00717 | 0.00717 | 0 at 5 decimals | – | 5.8e-6 |
| finance | RMSE | 0.00963 | 0.00963 | 0 at 5 decimals | – | 1.0e-5 |
| finance_vol | MAE | 0.00522 | 0.00522 | 0 at 5 decimals | – | 2.0e-4 |
| finance_vol | RMSE | 0.00642 | 0.00642 | 0 at 5 decimals | – | 1.2e-4 |

## Result

Dropping the H label-only values changes the target-train mean and std by at most 4.5e-4 relative in energy and
climate. In finance and finance_vol the change is at most 1.25e-2 relative on a near-zero mean and 1e-3 on the std.

The seed-0 target-test metrics of "source-only, target-train stats" change by under 0.01% in energy and climate:
−0.049 MW MAE for energy and +0.0001 °C MAE for climate. Across energy and climate MAE and RMSE, the changes are 0.6% to 4% of that row's seed std. For finance
and finance_vol the metrics are unchanged at the 5-decimal precision of the stored values.

Only seed 0 and only this row were re-evaluated, as requested. The adapted rows (CORAL, AdaRNN-style) were not
re-evaluated: their training also used target-train inputs z-scored with the old stats, so a like-for-like check for
them would need retraining.
