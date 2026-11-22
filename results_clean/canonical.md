# Canonical results (one source of truth)

All numbers are on the **target test set**, original scale, regenerated from the run JSONs by
`results_clean/scripts/build_tables.py` (machine-readable copy: `canonical.csv`, full precision).
Mean and std are over seeds; std is the sample std (ddof=1). Per-seed metrics in the JSONs are rounded
to 5 decimals (`src/utils/metrics.py`), so finance values carry only ~3 significant digits.

## Definitions

- **naive train-mean (target-train stats)**: predict the target-train mean for every horizon step (= 0 in target-normalised space).
- **Persistence / seasonal-naive** (new for the final TCN runs, CLAUDE.md amendment 2026-11-15 (g)(b)): For each target-test window with last observed step t and horizon step j=1..H: seasonal-naive with lag p predicts y_hat[t+j] = y[t + j - p*ceil(j/p)] (src.evaluate.seasonal_naive). p=1 is persistence: the last observed value y[t] repeated for all H steps. Energy (H=24): p=24 gives y[t+j-24] = same hour yesterday; p=168 gives y[t+j-168] = same hour last week. Mean of last 5 inputs: y_hat[t+j] = mean(y[t-4..t]) for all j. Windows: the same target-test windows the TCN runs score on (prepare_pair -> prepare_domain: chronological 70/15/15 split of the target series, windows built inside the test split only; inputs and targets both inside the test split). Lagged values never reach past t, and with these lags they stay inside the test split.
- **source-only, source stats**: TCN trained on source, target inputs normalised with source-train stats.
- **source-only, target-train stats**: same model, target inputs normalised with target-train stats (no target labels).
- **CORAL lambda=10**: TCN + CORAL loss on get_features(), λ=10 (pre-registered λ choice), target-train stats. `experiments/coral/`.
- **CORAL lambda=1 (sensitivity)**: same, λ=1, sensitivity check only. `experiments/coral_lambda1/`.
- **AdaRNN-style lambda=10**: AdaRNN-style (TDC periods + pairwise CORAL), λ=10, target-train stats. `experiments/adarnn/`.
- **target-only (labels, reference)**: TCN trained on target-train labels for a fixed epoch count (= source best epoch). Upper reference; uses target labels.
- **% naive** = 100·mean / mean(naive train-mean row). **% best baseline** = 100·mean / min over the naive and
  persistence/seasonal rows of the same metric (the row used is named in the CSV). **reduction vs SO-src** =
  100·(mean(SO-src) − mean) / mean(SO-src); positive = lower error.
- Units: MW (energy), °C (climate), log-return units (finance; a daily log-return, 0.01 ≈ 1%), |log-return| units (finance_vol).
  MAPE/sMAPE are not reported (meaningless for log-returns; kept out everywhere for consistency).

MAE is present in every finance JSON, so MAE and RMSE are both reported for finance.

## energy: AEP->COMED (unit: MW)

| setting | n | MAE mean ± std | RMSE mean ± std | MAE % naive | RMSE % naive | MAE % best baseline | RMSE % best baseline | MAE red. vs SO-src % | RMSE red. vs SO-src % |
|---|---|---|---|---|---|---|---|---|---|
| naive train-mean (target-train stats) | deterministic | 1720.81 | 2268.22 | 100.0 | 100.0 | 199.9 | 187.2 | -114.3 | -113.7 |
| seasonal-naive lag 24 (same hour yesterday) | deterministic | 860.85 | 1211.66 | 50.0 | 53.4 | 100.0 | 100.0 | -7.2 | -14.2 |
| seasonal-naive lag 168 (same hour last week) | deterministic | 1287.52 | 1899.17 | 74.8 | 83.7 | 149.6 | 156.7 | -60.4 | -79.0 |
| source-only, source stats | 3 | 802.88 ± 32.60 | 1061.17 ± 34.08 | 46.7 | 46.8 | 93.3 | 87.6 | 0.0 | 0.0 |
| source-only, target-train stats | 3 | 561.61 ± 8.75 | 824.40 ± 10.23 | 32.6 | 36.3 | 65.2 | 68.0 | 30.1 | 22.3 |
| CORAL lambda=10 | 3 | 554.78 ± 10.35 | 810.45 ± 14.58 | 32.2 | 35.7 | 64.4 | 66.9 | 30.9 | 23.6 |
| CORAL lambda=1 (sensitivity) | 3 | 561.10 ± 7.42 | 816.91 ± 11.82 | 32.6 | 36.0 | 65.2 | 67.4 | 30.1 | 23.0 |
| AdaRNN-style lambda=10 | 3 | 560.19 ± 2.00 | 811.97 ± 1.09 | 32.6 | 35.8 | 65.1 | 67.0 | 30.2 | 23.5 |
| target-only (labels, reference) | 3 | 524.44 ± 10.02 | 794.52 ± 6.95 | 30.5 | 35.0 | 60.9 | 65.6 | 34.7 | 25.1 |

Best baseline: MAE = seasonal-naive lag 24 (same hour yesterday); RMSE = seasonal-naive lag 24 (same hour yesterday).

## climate: Shimla->Delhi (unit: degC)

| setting | n | MAE mean ± std | RMSE mean ± std | MAE % naive | RMSE % naive | MAE % best baseline | RMSE % best baseline | MAE red. vs SO-src % | RMSE red. vs SO-src % |
|---|---|---|---|---|---|---|---|---|---|
| naive train-mean (target-train stats) | deterministic | 6.137 | 6.964 | 100.0 | 100.0 | 354.0 | 296.4 | -108.4 | -87.7 |
| persistence (last observed value repeated) | deterministic | 1.733 | 2.350 | 28.2 | 33.7 | 100.0 | 100.0 | 41.1 | 36.7 |
| source-only, source stats | 3 | 2.945 ± 0.596 | 3.709 ± 0.648 | 48.0 | 53.3 | 169.9 | 157.9 | 0.0 | 0.0 |
| source-only, target-train stats | 3 | 1.673 ± 0.003 | 2.196 ± 0.009 | 27.3 | 31.5 | 96.5 | 93.5 | 43.2 | 40.8 |
| CORAL lambda=10 | 3 | 1.698 ± 0.031 | 2.211 ± 0.010 | 27.7 | 31.8 | 98.0 | 94.1 | 42.3 | 40.4 |
| CORAL lambda=1 (sensitivity) | 3 | 1.667 ± 0.014 | 2.183 ± 0.011 | 27.2 | 31.4 | 96.2 | 92.9 | 43.4 | 41.1 |
| AdaRNN-style lambda=10 | 3 | 1.679 ± 0.005 | 2.203 ± 0.008 | 27.4 | 31.6 | 96.9 | 93.8 | 43.0 | 40.6 |
| target-only (labels, reference) | 3 | 1.674 ± 0.025 | 2.205 ± 0.026 | 27.3 | 31.7 | 96.6 | 93.8 | 43.2 | 40.6 |

Best baseline: MAE = persistence (last observed value repeated); RMSE = persistence (last observed value repeated).

## finance: Nifty IT->Nifty Pharma (log-return) (unit: log-return units)

| setting | n | MAE mean ± std | RMSE mean ± std | MAE % naive | RMSE % naive | MAE % best baseline | RMSE % best baseline | MAE red. vs SO-src % | RMSE red. vs SO-src % |
|---|---|---|---|---|---|---|---|---|---|
| naive train-mean (target-train stats) | deterministic | 0.007120 | 0.009540 | 100.0 | 100.0 | 100.0 | 100.0 | 0.7 | 0.8 |
| persistence (last observed value repeated) | deterministic | 0.010350 | 0.013640 | 145.4 | 143.0 | 145.4 | 143.0 | -44.4 | -41.8 |
| mean of last 5 inputs (repeated) | deterministic | 0.007960 | 0.010580 | 111.8 | 110.9 | 111.8 | 110.9 | -11.0 | -10.0 |
| source-only, source stats | 3 | 0.007170 ± 0.000000 | 0.009617 ± 0.000012 | 100.7 | 100.8 | 100.7 | 100.8 | 0.0 | 0.0 |
| source-only, target-train stats | 3 | 0.007173 ± 0.000006 | 0.009620 ± 0.000010 | 100.7 | 100.8 | 100.7 | 100.8 | -0.0 | -0.0 |
| CORAL lambda=10 | 3 | 0.007177 ± 0.000029 | 0.009627 ± 0.000021 | 100.8 | 100.9 | 100.8 | 100.9 | -0.1 | -0.1 |
| CORAL lambda=1 (sensitivity) | 3 | 0.007170 ± 0.000010 | 0.009623 ± 0.000006 | 100.7 | 100.9 | 100.7 | 100.9 | 0.0 | -0.1 |
| AdaRNN-style lambda=10 | 3 | 0.007183 ± 0.000012 | 0.009633 ± 0.000031 | 100.9 | 101.0 | 100.9 | 101.0 | -0.2 | -0.2 |
| target-only (labels, reference) | 3 | 0.007157 ± 0.000038 | 0.009603 ± 0.000049 | 100.5 | 100.7 | 100.5 | 100.7 | 0.2 | 0.1 |

Best baseline: MAE = naive train-mean (target-train stats); RMSE = naive train-mean (target-train stats).

## finance_vol: Nifty IT->Nifty Pharma (|log-return|) (unit: |log-return| units)

| setting | n | MAE mean ± std | RMSE mean ± std | MAE % naive | RMSE % naive | MAE % best baseline | RMSE % best baseline | MAE red. vs SO-src % | RMSE red. vs SO-src % |
|---|---|---|---|---|---|---|---|---|---|
| naive train-mean (target-train stats) | deterministic | 0.005720 | 0.006780 | 100.0 | 100.0 | 115.1 | 102.1 | -12.2 | -6.2 |
| persistence (last observed value repeated) | deterministic | 0.006220 | 0.008370 | 108.7 | 123.5 | 125.2 | 126.1 | -22.0 | -31.1 |
| mean of last 5 inputs (repeated) | deterministic | 0.004970 | 0.006640 | 86.9 | 97.9 | 100.0 | 100.0 | 2.5 | -4.0 |
| source-only, source stats | 3 | 0.005100 ± 0.000197 | 0.006383 ± 0.000110 | 89.2 | 94.1 | 102.6 | 96.1 | 0.0 | 0.0 |
| source-only, target-train stats | 3 | 0.005157 ± 0.000203 | 0.006410 ± 0.000115 | 90.2 | 94.5 | 103.8 | 96.5 | -1.1 | -0.4 |
| CORAL lambda=10 | 3 | 0.005207 ± 0.000031 | 0.006430 ± 0.000010 | 91.0 | 94.8 | 104.8 | 96.8 | -2.1 | -0.7 |
| CORAL lambda=1 (sensitivity) | 3 | 0.005180 ± 0.000062 | 0.006413 ± 0.000029 | 90.6 | 94.6 | 104.2 | 96.6 | -1.6 | -0.5 |
| AdaRNN-style lambda=10 | 3 | 0.005310 ± 0.000131 | 0.006490 ± 0.000082 | 92.8 | 95.7 | 106.8 | 97.7 | -4.1 | -1.7 |
| target-only (labels, reference) | 3 | 0.005267 ± 0.000123 | 0.006460 ± 0.000075 | 92.1 | 95.3 | 106.0 | 97.3 | -3.3 | -1.2 |

Best baseline: MAE = mean of last 5 inputs (repeated); RMSE = mean of last 5 inputs (repeated).

## Window alignment check (baselines vs model test windows)

```
[energy] model test windows (prepare_pair, target-train stats): 9785 x H=24; JSON n_target_test_windows = 9785; equal: True
[energy] seasonal-naive lag 24 (same hour yesterday): windows 9785 x 24 vs model 9785 x 24; max |y_true(baseline) - y_true(model windows)| = 4.95e-04; ALIGNED: True
[energy] seasonal-naive lag 168 (same hour last week): windows 9785 x 24 vs model 9785 x 24; max |y_true(baseline) - y_true(model windows)| = 4.95e-04; ALIGNED: True
[energy] recomputed naive train-mean (target-train stats): MAE 1720.8085 RMSE 2268.2182 | summary.json MAE 1720.80859 RMSE 2268.21826 | match: True
[energy] recomputed seasonal-naive lag 24 (same hour yesterday): MAE 860.85319 RMSE 1211.66104 | summary.json MAE 860.85319 RMSE 1211.66104 | match: True
[energy] recomputed seasonal-naive lag 168 (same hour last week): MAE 1287.52001 RMSE 1899.17242 | summary.json MAE 1287.52001 RMSE 1899.17242 | match: True
[climate] model test windows (prepare_pair, target-train stats): 610 x H=5; JSON n_target_test_windows = 610; equal: True
[climate] persistence (last observed value repeated): windows 610 x 5 vs model 610 x 5; max |y_true(baseline) - y_true(model windows)| = 8.02e-07; ALIGNED: True
[climate] recomputed naive train-mean (target-train stats): MAE 6.13669 RMSE 6.96401 | summary.json MAE 6.13669 RMSE 6.96401 | match: True
[climate] recomputed persistence (last observed value repeated): MAE 1.73338 RMSE 2.34981 | summary.json MAE 1.73338 RMSE 2.34981 | match: True
[finance] model test windows (prepare_pair, target-train stats): 401 x H=5; JSON n_target_test_windows = 401; equal: True
[finance] persistence (last observed value repeated): windows 401 x 5 vs model 401 x 5; max |y_true(baseline) - y_true(model windows)| = 1.51e-09; ALIGNED: True
[finance] mean of last 5 inputs (repeated): windows 401 x 5 vs model 401 x 5; max |y_true(baseline) - y_true(model windows)| = 0.00e+00; ALIGNED: True
[finance] recomputed naive train-mean (target-train stats): MAE 0.00712 RMSE 0.00954 | summary.json MAE 0.00712 RMSE 0.00954 | match: True
[finance] recomputed persistence (last observed value repeated): MAE 0.01035 RMSE 0.01364 | summary.json MAE 0.01035 RMSE 0.01364 | match: True
[finance] mean of last 5 inputs (repeated): not in summary.json (new baseline, computed here)
[finance_vol] model test windows (prepare_pair, target-train stats): 401 x H=5; JSON n_target_test_windows = 401; equal: True
[finance_vol] persistence (last observed value repeated): windows 401 x 5 vs model 401 x 5; max |y_true(baseline) - y_true(model windows)| = 1.05e-09; ALIGNED: True
[finance_vol] mean of last 5 inputs (repeated): windows 401 x 5 vs model 401 x 5; max |y_true(baseline) - y_true(model windows)| = 0.00e+00; ALIGNED: True
[finance_vol] recomputed naive train-mean (target-train stats): MAE 0.00572 RMSE 0.00678 | summary.json MAE 0.00572 RMSE 0.00678 | match: True
[finance_vol] recomputed mean of last 5 inputs (repeated): MAE 0.00497 RMSE 0.00664 | summary.json MAE 0.00497 RMSE 0.00664 | match: True
[finance_vol] persistence (last observed value repeated): not in summary.json (new baseline, computed here)
```

