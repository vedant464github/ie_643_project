# Scale audit: normalisation round trip and metric recomputation

Produced by `results_clean/audit/audit_compute.py` (inference only, CPU, 2 threads), with all numbers in
`results_clean/audit/audit_numbers.json`. Normalisation is `z = (x − m) / s` (`src/data/prepare.py:apply_norm`), and
the inverse is `x = z·s + m` (`src/evaluate.py:denorm`). Here m and s are the mean and std (numpy `std`, ddof=0) of the
raw **train** split of the series that owns the stats.

## 1. Round trip: normalise, then convert back

For each domain, each split, and each series/stats pairing the pipeline uses, the check normalises the raw values and
converts them back. It compares the result with the raw values. It runs twice: once in float64, and once through a
float32 cast of the z-scores. The float32 path is what the pipeline does, because `make_windows` stores windows as
float32 (`src/data/windowing.py:17`). The last two rows per domain use the actual `Y_test` window arrays from
`prepare_pair`.

**Result.**
- **float64: passes the 1e-6 tolerance everywhere.** The largest error is 9.1e-13 MW, on energy.
- **float32 pipeline arrays: does not pass an absolute 1e-6 tolerance for energy or climate.** The largest errors are
  9.8e-4 MW for energy and 3.1e-6 °C for climate. These equal float32 rounding of z: about 6e-8 relative, times s.
  For the finance domains the error is at most 8.6e-9.
- Relative to the data range, every error is below 1e-7.
- Effect on reported metrics: none at the 5-decimal rounding of the JSONs. Section 2 shows the float32 pipeline path
  reproduces every JSON value exactly. The float64 path differs by at most 1.2e-4 MW (energy) and 5e-6 in the other
  domains.

### energy
source-train mean 15265.50127, std 2509.35527; target-train mean 11446.54458, std 2280.957696

| series / stats / split | n | max abs err, float64 | max abs err, via float32 arrays | pass 1e-6 (float64) | pass 1e-6 (float32) |
|---|---|---|---|---|---|
| source series, source-train stats | train | 46552 | 0.00e+00 | 0.00e+00 | yes | yes |
| source series, source-train stats | val | 9976 | 0.00e+00 | 0.00e+00 | yes | yes |
| source series, source-train stats | test | 9976 | 0.00e+00 | 0.00e+00 | yes | yes |
| target series, source-train stats | train | 46552 | 9.09e-13 | 4.88e-04 | yes | NO |
| target series, source-train stats | val | 9976 | 9.09e-13 | 4.88e-04 | yes | NO |
| target series, source-train stats | test | 9976 | 9.09e-13 | 4.88e-04 | yes | NO |
| target series, target-train stats | train | 46552 | 0.00e+00 | 9.77e-04 | yes | NO |
| target series, target-train stats | val | 9976 | 0.00e+00 | 9.77e-04 | yes | NO |
| target series, target-train stats | test | 9976 | 0.00e+00 | 9.77e-04 | yes | NO |
| target test windows Y, target-train stats (pipeline arrays) | 234840 | – | 4.95e-04 | – | NO |
| target test windows Y, source-train stats (pipeline arrays) | 234840 | – | 2.99e-04 | – | NO |
### climate
source-train mean 11.11165501, std 5.993225504; target-train mean 24.11252081, std 7.156369851

| series / stats / split | n | max abs err, float64 | max abs err, via float32 arrays | pass 1e-6 (float64) | pass 1e-6 (float32) |
|---|---|---|---|---|---|
| source series, source-train stats | train | 3003 | 2.66e-15 | 1.53e-06 | yes | NO |
| source series, source-train stats | val | 644 | 1.33e-15 | 1.53e-06 | yes | NO |
| source series, source-train stats | test | 644 | 1.33e-15 | 1.53e-06 | yes | NO |
| target series, source-train stats | train | 3003 | 7.11e-15 | 3.05e-06 | yes | NO |
| target series, source-train stats | val | 644 | 7.11e-15 | 3.05e-06 | yes | NO |
| target series, source-train stats | test | 644 | 7.11e-15 | 3.05e-06 | yes | NO |
| target series, target-train stats | train | 3003 | 1.78e-15 | 3.05e-06 | yes | NO |
| target series, target-train stats | val | 644 | 1.78e-15 | 3.05e-06 | yes | NO |
| target series, target-train stats | test | 644 | 1.78e-15 | 3.05e-06 | yes | NO |
| target test windows Y, target-train stats (pipeline arrays) | 3050 | – | 8.02e-07 | – | yes |
| target test windows Y, source-train stats (pipeline arrays) | 3050 | – | 1.39e-06 | – | NO |
### finance
source-train mean 0.0004590950663, std 0.01318658488; target-train mean 3.2899698e-05, std 0.01314131698

| series / stats / split | n | max abs err, float64 | max abs err, via float32 arrays | pass 1e-6 (float64) | pass 1e-6 (float32) |
|---|---|---|---|---|---|
| source series, source-train stats | train | 2027 | 6.94e-18 | 8.14e-09 | yes | yes |
| source series, source-train stats | val | 434 | 1.73e-18 | 4.01e-09 | yes | yes |
| source series, source-train stats | test | 435 | 3.47e-18 | 4.30e-09 | yes | yes |
| target series, source-train stats | train | 2026 | 3.47e-18 | 6.35e-09 | yes | yes |
| target series, source-train stats | val | 434 | 3.47e-18 | 2.41e-09 | yes | yes |
| target series, source-train stats | test | 435 | 3.47e-18 | 2.09e-09 | yes | yes |
| target series, target-train stats | train | 2026 | 3.47e-18 | 5.21e-09 | yes | yes |
| target series, target-train stats | val | 434 | 1.73e-18 | 1.54e-09 | yes | yes |
| target series, target-train stats | test | 435 | 3.47e-18 | 2.16e-09 | yes | yes |
| target test windows Y, target-train stats (pipeline arrays) | 2005 | – | 1.51e-09 | – | yes |
| target test windows Y, source-train stats (pipeline arrays) | 2005 | – | 1.52e-09 | – | yes |
### finance_vol
source-train mean 0.009349418359, std 0.00931048685; target-train mean 0.0095105799, std 0.009068856842

| series / stats / split | n | max abs err, float64 | max abs err, via float32 arrays | pass 1e-6 (float64) | pass 1e-6 (float32) |
|---|---|---|---|---|---|
| source series, source-train stats | train | 2027 | 1.39e-17 | 4.93e-09 | yes | yes |
| source series, source-train stats | val | 434 | 6.94e-18 | 4.05e-09 | yes | yes |
| source series, source-train stats | test | 435 | 6.94e-18 | 4.83e-09 | yes | yes |
| target series, source-train stats | train | 2026 | 1.39e-17 | 5.50e-09 | yes | yes |
| target series, source-train stats | val | 434 | 3.47e-18 | 1.45e-09 | yes | yes |
| target series, source-train stats | test | 435 | 6.94e-18 | 2.01e-09 | yes | yes |
| target series, target-train stats | train | 2026 | 6.94e-18 | 8.56e-09 | yes | yes |
| target series, target-train stats | val | 434 | 3.47e-18 | 1.53e-09 | yes | yes |
| target series, target-train stats | test | 435 | 3.47e-18 | 2.22e-09 | yes | yes |
| target test windows Y, target-train stats (pipeline arrays) | 2005 | – | 1.05e-09 | – | yes |
| target test windows Y, source-train stats (pipeline arrays) | 2005 | – | 8.37e-10 | – | yes |

## 2. Recomputation of seed-0 MAE and RMSE from raw predictions, per canonical row

For every canonical row, seed 0 is reloaded from its saved checkpoint. The script predicts on the target-test windows
with the row's input stats, de-normalises with the row's output stats, and computes MAE and RMSE in original units
two ways:

- **pipeline recompute (float32):** exactly `src/evaluate.py:evaluate_model`. Truth is `denorm(Y_test)` with the
  same stats.
- **float64 from raw series:** truth is cut directly from the raw target CSV at the test-window indices
  (`test_start = int(n·0.85)`, offset `input_len + j`), and the prediction is cast to float64 before de-normalising.
  This path does not use the normalised truth at all.

Deterministic baselines are recomputed from the raw series and compared with `results_clean/canonical.csv`. Their
model-free inputs need no stats, except the naive train-mean (the forecast is z = 0, de-normalised with target-train
stats) and mean-of-last-5 (target-train z-scores inverted, then averaged).

**Result.**
- The pipeline recompute matches the JSON MAE and RMSE **exactly**, to all 5 stored decimals, for all 24 model rows
  (6 rows × 4 domains, seed 0).
- The float64 recompute agrees to within 1.2e-4 MW (energy, about 2e-7 relative) and within 5e-6 elsewhere.
- Every baseline matches canonical.csv at its stored precision.
- JSON sources: `experiments/final_3seed/tcn/runs/<d>_seed0.json` (source-only and target-only),
  `experiments/coral/<d>/seed0.json`, `experiments/coral_lambda1/<d>/seed0.json`,
  `experiments/adarnn/<d>/seed0.json`.

### Which mean and std are used at each end, per row

| canonical row | input end (normalise target-test inputs) | output end (de-normalise predictions and truth) |
|---|---|---|
| naive train-mean (target-train stats) | none: the forecast is the constant z = 0 | target-train m, s, so the forecast is the target-train mean |
| persistence / seasonal-naive lag 24 / lag 168 | none: copied from the raw series | none: raw units |
| mean of last 5 inputs | target-train (inputs inverted back to raw units first) | none: raw units |
| source-only, source stats | **source-train** m, s | **source-train** m, s |
| source-only, target-train stats | target-train m, s | target-train m, s |
| CORAL λ=10, CORAL λ=1, AdaRNN-style λ=10 | target-train m, s (training also used source-train stats for source windows and target-train stats for target inputs) | target-train m, s |
| target-only (labels, reference) | target-train m, s (trained in the same space) | target-train m, s |

### Numbers

### energy (MW)

| canonical row | seed | input end: z = (x − m)/s with | output end: ŷ = z·s + m with | MAE: JSON/CSV | MAE: pipeline recompute (float32) | MAE: float64 from raw series | RMSE: JSON/CSV | RMSE: pipeline recompute (float32) | RMSE: float64 from raw series |
|---|---|---|---|---|---|---|---|---|---|
| naive train-mean (target-train stats) | deterministic | – (forecast is z = 0) | target-train (m=11446.5, s=2280.96) | 1720.8085 | – | 1720.8085 | 2268.2182 | – | 2268.21819 |
| seasonal-naive lag 24 (same hour yesterday) | deterministic | none (raw series) | none (raw series) | 860.85319 | – | 860.853185 | 1211.66104 | – | 1211.66104 |
| seasonal-naive lag 168 (same hour last week) | deterministic | none (raw series) | none (raw series) | 1287.52001 | – | 1287.52001 | 1899.17242 | – | 1899.17242 |
| source-only, source stats | 0 | source-train (m=15265.5, s=2509.36) | source-train (m=15265.5, s=2509.36) | 771.13586 | 771.13586 | 771.135977 | 1029.55933 | 1029.55933 | 1029.55942 |
| source-only, target-train stats | 0 | target-train (m=11446.5, s=2280.96) | target-train (m=11446.5, s=2280.96) | 561.30774 | 561.30774 | 561.307662 | 828.06134 | 828.06134 | 828.061332 |
| CORAL lambda=10 | 0 | target-train (m=11446.5, s=2280.96) | target-train (m=11446.5, s=2280.96) | 551.59674 | 551.59674 | 551.596747 | 804.0824 | 804.0824 | 804.082403 |
| CORAL lambda=1 (sensitivity) | 0 | target-train (m=11446.5, s=2280.96) | target-train (m=11446.5, s=2280.96) | 557.76215 | 557.76215 | 557.762109 | 812.08289 | 812.08289 | 812.082883 |
| AdaRNN-style lambda=10 | 0 | target-train (m=11446.5, s=2280.96) | target-train (m=11446.5, s=2280.96) | 562.24323 | 562.24323 | 562.243194 | 812.13245 | 812.13245 | 812.132418 |
| target-only (labels, reference) | 0 | target-train (m=11446.5, s=2280.96) | target-train (m=11446.5, s=2280.96) | 517.73853 | 517.73853 | 517.738501 | 790.23773 | 790.23773 | 790.237644 |

### climate (°C)

| canonical row | seed | input end: z = (x − m)/s with | output end: ŷ = z·s + m with | MAE: JSON/CSV | MAE: pipeline recompute (float32) | MAE: float64 from raw series | RMSE: JSON/CSV | RMSE: pipeline recompute (float32) | RMSE: float64 from raw series |
|---|---|---|---|---|---|---|---|---|---|
| naive train-mean (target-train stats) | deterministic | – (forecast is z = 0) | target-train (m=24.1125, s=7.15637) | 6.13669 | – | 6.13668953 | 6.96401 | – | 6.96401031 |
| persistence (last observed value repeated) | deterministic | none (raw series) | none (raw series) | 1.73338 | – | 1.73337705 | 2.34981 | – | 2.34981408 |
| source-only, source stats | 0 | source-train (m=11.1117, s=5.99323) | source-train (m=11.1117, s=5.99323) | 2.66901 | 2.66901 | 2.6690075 | 3.41855 | 3.41855 | 3.41854655 |
| source-only, target-train stats | 0 | target-train (m=24.1125, s=7.15637) | target-train (m=24.1125, s=7.15637) | 1.67283 | 1.67283 | 1.67283309 | 2.20048 | 2.20048 | 2.20047536 |
| CORAL lambda=10 | 0 | target-train (m=24.1125, s=7.15637) | target-train (m=24.1125, s=7.15637) | 1.69168 | 1.69168 | 1.69167502 | 2.20896 | 2.20896 | 2.20896037 |
| CORAL lambda=1 (sensitivity) | 0 | target-train (m=24.1125, s=7.15637) | target-train (m=24.1125, s=7.15637) | 1.6785 | 1.6785 | 1.67849916 | 2.19596 | 2.19596 | 2.19596286 |
| AdaRNN-style lambda=10 | 0 | target-train (m=24.1125, s=7.15637) | target-train (m=24.1125, s=7.15637) | 1.6765 | 1.6765 | 1.6764985 | 2.21226 | 2.21226 | 2.21226314 |
| target-only (labels, reference) | 0 | target-train (m=24.1125, s=7.15637) | target-train (m=24.1125, s=7.15637) | 1.6553 | 1.6553 | 1.65530234 | 2.19131 | 2.19131 | 2.19131176 |

### finance (log-return units)

| canonical row | seed | input end: z = (x − m)/s with | output end: ŷ = z·s + m with | MAE: JSON/CSV | MAE: pipeline recompute (float32) | MAE: float64 from raw series | RMSE: JSON/CSV | RMSE: pipeline recompute (float32) | RMSE: float64 from raw series |
|---|---|---|---|---|---|---|---|---|---|
| naive train-mean (target-train stats) | deterministic | – (forecast is z = 0) | target-train (m=3.28997e-05, s=0.0131413) | 0.00712 | – | 0.00711971502 | 0.00954 | – | 0.00953970024 |
| persistence (last observed value repeated) | deterministic | none (raw series) | none (raw series) | 0.01035 | – | 0.0103514051 | 0.01364 | – | 0.0136433762 |
| mean of last 5 inputs (repeated) | deterministic | target-train (m=3.28997e-05, s=0.0131413), inverted before averaging | none (forecast built in raw units) | 0.00796 | – | 0.00796377869 | 0.01058 | – | 0.0105787972 |
| source-only, source stats | 0 | source-train (m=0.000459095, s=0.0131866) | source-train (m=0.000459095, s=0.0131866) | 0.00717 | 0.00717 | 0.00717320163 | 0.00963 | 0.00963 | 0.00962656677 |
| source-only, target-train stats | 0 | target-train (m=3.28997e-05, s=0.0131413) | target-train (m=3.28997e-05, s=0.0131413) | 0.00717 | 0.00717 | 0.00717148891 | 0.00963 | 0.00963 | 0.00962990793 |
| CORAL lambda=10 | 0 | target-train (m=3.28997e-05, s=0.0131413) | target-train (m=3.28997e-05, s=0.0131413) | 0.00716 | 0.00716 | 0.00715953609 | 0.00962 | 0.00962 | 0.0096191947 |
| CORAL lambda=1 (sensitivity) | 0 | target-train (m=3.28997e-05, s=0.0131413) | target-train (m=3.28997e-05, s=0.0131413) | 0.00717 | 0.00717 | 0.00717244978 | 0.00963 | 0.00963 | 0.00963013877 |
| AdaRNN-style lambda=10 | 0 | target-train (m=3.28997e-05, s=0.0131413) | target-train (m=3.28997e-05, s=0.0131413) | 0.00719 | 0.00719 | 0.0071915143 | 0.00966 | 0.00966 | 0.00965749557 |
| target-only (labels, reference) | 0 | target-train (m=3.28997e-05, s=0.0131413) | target-train (m=3.28997e-05, s=0.0131413) | 0.0072 | 0.0072 | 0.00719824586 | 0.00966 | 0.00966 | 0.00966361137 |

### finance_vol (|log-return| units)

| canonical row | seed | input end: z = (x − m)/s with | output end: ŷ = z·s + m with | MAE: JSON/CSV | MAE: pipeline recompute (float32) | MAE: float64 from raw series | RMSE: JSON/CSV | RMSE: pipeline recompute (float32) | RMSE: float64 from raw series |
|---|---|---|---|---|---|---|---|---|---|
| naive train-mean (target-train stats) | deterministic | – (forecast is z = 0) | target-train (m=0.00951058, s=0.00906886) | 0.00572 | – | 0.00572271589 | 0.00678 | – | 0.00678427523 |
| persistence (last observed value repeated) | deterministic | none (raw series) | none (raw series) | 0.00622 | – | 0.00621949705 | 0.00837 | – | 0.00837223799 |
| mean of last 5 inputs (repeated) | deterministic | target-train (m=0.00951058, s=0.00906886), inverted before averaging | none (forecast built in raw units) | 0.00497 | – | 0.00496693093 | 0.00664 | – | 0.00663736653 |
| source-only, source stats | 0 | source-train (m=0.00934942, s=0.00931049) | source-train (m=0.00934942, s=0.00931049) | 0.00516 | 0.00516 | 0.0051637628 | 0.00639 | 0.00639 | 0.00639047637 |
| source-only, target-train stats | 0 | target-train (m=0.00951058, s=0.00906886) | target-train (m=0.00951058, s=0.00906886) | 0.00522 | 0.00522 | 0.00521708899 | 0.00642 | 0.00642 | 0.00642120522 |
| CORAL lambda=10 | 0 | target-train (m=0.00951058, s=0.00906886) | target-train (m=0.00951058, s=0.00906886) | 0.00524 | 0.00524 | 0.00523623621 | 0.00644 | 0.00644 | 0.00643578257 |
| CORAL lambda=1 (sensitivity) | 0 | target-train (m=0.00951058, s=0.00906886) | target-train (m=0.00951058, s=0.00906886) | 0.00523 | 0.00523 | 0.00522529214 | 0.00643 | 0.00643 | 0.00642590402 |
| AdaRNN-style lambda=10 | 0 | target-train (m=0.00951058, s=0.00906886) | target-train (m=0.00951058, s=0.00906886) | 0.00543 | 0.00543 | 0.00543197592 | 0.00656 | 0.00656 | 0.0065566279 |
| target-only (labels, reference) | 0 | target-train (m=0.00951058, s=0.00906886) | target-train (m=0.00951058, s=0.00906886) | 0.0053 | 0.0053 | 0.0052993702 | 0.00647 | 0.00647 | 0.00647490718 |

