# Summary: TCN adaptation staircase, all domains

Target test set, original scale, mean ± sample std (ddof=1) over seeds; n = seeds in the cell; 'not run' = no result on disk. All adaptation rows normalise target inputs with target-train stats and use target-train INPUTS only (no target labels). Target-only uses target labels (reference only). CORAL λ=10 is the pre-registered final; CORAL λ=1 is a sensitivity check and does not replace it. finance is raw log-returns (no learnable signal) and finance_vol is a negative control: lead with RMSE there. MAPE/sMAPE are not reported.

AdaRNN-style hard stop 2026-10-25T22:00:00: all 12 runs finished before the stop.

## finance

| setting | MAE | RMSE |
|---|---|---|
| naive train-mean (target-train stats) | 0.00712 (n=1) | 0.00954 (n=1) |
| source-only, source stats | 0.00717 ± 0 (n=3) | 0.0096167 ± 1.2e-05 (n=3) |
| source-only, target-train stats | 0.0071733 ± 5.8e-06 (n=3) | 0.00962 ± 1e-05 (n=3) |
| CORAL λ=10 | 0.0071767 ± 2.9e-05 (n=3) | 0.0096267 ± 2.1e-05 (n=3) |
| CORAL λ=1 (sensitivity) | 0.00717 ± 1e-05 (n=3) | 0.0096233 ± 5.8e-06 (n=3) |
| AdaRNN-style λ=10 | 0.0071833 ± 1.2e-05 (n=3) | 0.0096333 ± 3.1e-05 (n=3) |
| target-only (uses target labels) | 0.0071567 ± 3.8e-05 (n=3) | 0.0096033 ± 4.9e-05 (n=3) |

MMD² (RBF, median heuristic) between source-val and target-val features, label-free:

| model | MMD² |
|---|---|
| source-only (before adaptation) | 0.040841 ± 0.011 (n=3) |
| CORAL λ=10 | 0.040955 ± 0.012 (n=3) |
| CORAL λ=1 (sensitivity) | 0.041781 ± 0.0086 (n=3) |
| AdaRNN-style λ=10 | 0.046951 ± 0.012 (n=3) |

## finance_vol

| setting | RMSE | MAE |
|---|---|---|
| naive train-mean (target-train stats) | 0.00678 (n=1) | 0.00572 (n=1) |
| source-only, source stats | 0.0063833 ± 0.00011 (n=3) | 0.0051 ± 0.0002 (n=3) |
| source-only, target-train stats | 0.00641 ± 0.00012 (n=3) | 0.0051567 ± 0.0002 (n=3) |
| CORAL λ=10 | 0.00643 ± 1e-05 (n=3) | 0.0052067 ± 3.1e-05 (n=3) |
| CORAL λ=1 (sensitivity) | 0.0064133 ± 2.9e-05 (n=3) | 0.00518 ± 6.2e-05 (n=3) |
| AdaRNN-style λ=10 | 0.00649 ± 8.2e-05 (n=3) | 0.00531 ± 0.00013 (n=3) |
| target-only (uses target labels) | 0.00646 ± 7.5e-05 (n=3) | 0.0052667 ± 0.00012 (n=3) |

MMD² (RBF, median heuristic) between source-val and target-val features, label-free:

| model | MMD² |
|---|---|
| source-only (before adaptation) | 0.078864 ± 0.016 (n=3) |
| CORAL λ=10 | 0.066136 ± 0.013 (n=3) |
| CORAL λ=1 (sensitivity) | 0.077945 ± 0.011 (n=3) |
| AdaRNN-style λ=10 | 0.063636 ± 0.0075 (n=3) |

## climate

| setting | MAE | RMSE |
|---|---|---|
| naive train-mean (target-train stats) | 6.1367 (n=1) | 6.964 (n=1) |
| source-only, source stats | 2.9452 ± 0.6 (n=3) | 3.7093 ± 0.65 (n=3) |
| source-only, target-train stats | 1.6731 ± 0.0027 (n=3) | 2.196 ± 0.0094 (n=3) |
| CORAL λ=10 | 1.6981 ± 0.031 (n=3) | 2.2112 ± 0.01 (n=3) |
| CORAL λ=1 (sensitivity) | 1.6673 ± 0.014 (n=3) | 2.1833 ± 0.011 (n=3) |
| AdaRNN-style λ=10 | 1.679 ± 0.0055 (n=3) | 2.2031 ± 0.0084 (n=3) |
| target-only (uses target labels) | 1.6736 ± 0.025 (n=3) | 2.205 ± 0.026 (n=3) |

MMD² (RBF, median heuristic) between source-val and target-val features, label-free:

| model | MMD² |
|---|---|
| source-only (before adaptation) | 0.059411 ± 0.0053 (n=3) |
| CORAL λ=10 | 0.051916 ± 0.0083 (n=3) |
| CORAL λ=1 (sensitivity) | 0.051827 ± 0.0015 (n=3) |
| AdaRNN-style λ=10 | 0.053531 ± 0.0087 (n=3) |

## energy

| setting | MAE | RMSE |
|---|---|---|
| naive train-mean (target-train stats) | 1720.8 (n=1) | 2268.2 (n=1) |
| source-only, source stats | 802.88 ± 33 (n=3) | 1061.2 ± 34 (n=3) |
| source-only, target-train stats | 561.61 ± 8.7 (n=3) | 824.4 ± 10 (n=3) |
| CORAL λ=10 | 554.78 ± 10 (n=3) | 810.45 ± 15 (n=3) |
| CORAL λ=1 (sensitivity) | 561.1 ± 7.4 (n=3) | 816.91 ± 12 (n=3) |
| AdaRNN-style λ=10 | 560.19 ± 2 (n=3) | 811.97 ± 1.1 (n=3) |
| target-only (uses target labels) | 524.44 ± 10 (n=3) | 794.52 ± 7 (n=3) |

MMD² (RBF, median heuristic) between source-val and target-val features, label-free:

| model | MMD² |
|---|---|
| source-only (before adaptation) | 0.030985 ± 0.0086 (n=3) |
| CORAL λ=10 | 0.026017 ± 0.0044 (n=3) |
| CORAL λ=1 (sensitivity) | 0.024891 ± 0.0053 (n=3) |
| AdaRNN-style λ=10 | 0.031283 ± 0.0059 (n=3) |

## energy: paired differences vs source-only (target-train stats)

Descriptive only; not a success criterion. Per seed, method minus source-only (target-train stats) on the same seed's target test set (negative = lower error than the target-stats row); mean and sample std (ddof=1) of the per-seed differences.

| method | metric | seed 0 | seed 1 | seed 2 | mean ± std of diff |
|---|---|---|---|---|---|
| CORAL λ=10 | MAE | -9.711 | -4.172 | -6.632 | -6.838 ± 2.78 (n=3) |
| CORAL λ=10 | RMSE | -23.98 | -5.171 | -12.72 | -13.96 ± 9.46 (n=3) |
| AdaRNN-style λ=10 | MAE | +0.9355 | -12.26 | +7.058 | -1.422 ± 9.87 (n=3) |
| AdaRNN-style λ=10 | RMSE | -15.93 | -19.33 | -2.032 | -12.43 ± 9.16 (n=3) |

Figures: report/figures/. Per-method details: experiments/coral/summary.md (λ=10), experiments/coral/lambda_selection.json, experiments/adarnn/tdc.json, experiments/adarnn/status.json.
