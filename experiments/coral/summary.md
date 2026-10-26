# CORAL (TCN), λ = 10

Target test set, original scale. mean ± sample std (ddof=1) over seeds; n = number of seeds in the cell. Naive train-mean is deterministic (n=1). Per-window metrics are rounded to 5 decimals before averaging (src.utils.metrics), so finance stds near 1e-5 are at the rounding floor. MAPE/sMAPE are not reported. finance_vol is a negative control: lead with RMSE there (naive median beats every model on MAE).

CORAL used target-TRAIN inputs only (target-train stats), no target labels, val or test. λ chosen on seed-0 SOURCE val MSE (experiments/coral/lambda_selection.json). Target-only uses target labels and is a reference, not an adaptation method.

## finance

| setting | MAE | RMSE |
|---|---|---|
| naive train-mean (target-train stats) | 0.00712 (n=1) | 0.00954 (n=1) |
| source-only (source stats) | 0.00717 ± 0 (n=3) | 0.0096167 ± 1.2e-05 (n=3) |
| source-only (target-train stats) | 0.0071733 ± 5.8e-06 (n=3) | 0.00962 ± 1e-05 (n=3) |
| target-only (upper reference, uses target labels) | 0.0071567 ± 3.8e-05 (n=3) | 0.0096033 ± 4.9e-05 (n=3) |
| CORAL λ=10 (target-train stats) | 0.0071767 ± 2.9e-05 (n=3) | 0.0096267 ± 2.1e-05 (n=3) |

- MAE: CORAL − target-stats = +3.333e-06 (max seed std 2.9e-05) → within seed std
- RMSE: CORAL − target-stats = +6.667e-06 (max seed std 2.1e-05) → within seed std

Label-free gap, source-val vs target-val features (MMD² RBF, median heuristic; CORAL distance):

| model | MMD² | CORAL distance |
|---|---|---|
| source-only | 0.040841 ± 0.011 (n=3) | 0.00033922 ± 0.00029 (n=3) |
| CORAL λ=10 | 0.040955 ± 0.012 (n=3) | 0.00023184 ± 0.00019 (n=3) |

Source-val MSE (norm): source-only 0.80686 ± 0.0031 (n=3), CORAL 0.80556 ± 0.0036 (n=3). CORAL best epochs [1, 4, 2], wall clock [2.81, 5.89, 3.46] s.

## finance_vol

| setting | RMSE | MAE |
|---|---|---|
| naive train-mean (target-train stats) | 0.00678 (n=1) | 0.00572 (n=1) |
| source-only (source stats) | 0.0063833 ± 0.00011 (n=3) | 0.0051 ± 0.0002 (n=3) |
| source-only (target-train stats) | 0.00641 ± 0.00012 (n=3) | 0.0051567 ± 0.0002 (n=3) |
| target-only (upper reference, uses target labels) | 0.00646 ± 7.5e-05 (n=3) | 0.0052667 ± 0.00012 (n=3) |
| CORAL λ=10 (target-train stats) | 0.00643 ± 1e-05 (n=3) | 0.0052067 ± 3.1e-05 (n=3) |

- RMSE: CORAL − target-stats = +2e-05 (max seed std 0.00012) → within seed std
- MAE: CORAL − target-stats = +5e-05 (max seed std 0.0002) → within seed std

Label-free gap, source-val vs target-val features (MMD² RBF, median heuristic; CORAL distance):

| model | MMD² | CORAL distance |
|---|---|---|
| source-only | 0.078864 ± 0.016 (n=3) | 0.0010855 ± 0.0012 (n=3) |
| CORAL λ=10 | 0.066136 ± 0.013 (n=3) | 0.00066142 ± 0.00073 (n=3) |

Source-val MSE (norm): source-only 0.82042 ± 0.004 (n=3), CORAL 0.81865 ± 0.0041 (n=3). CORAL best epochs [1, 2, 1], wall clock [2.78, 3.49, 2.89] s.

## climate

| setting | MAE | RMSE |
|---|---|---|
| naive train-mean (target-train stats) | 6.1367 (n=1) | 6.964 (n=1) |
| source-only (source stats) | 2.9452 ± 0.6 (n=3) | 3.7093 ± 0.65 (n=3) |
| source-only (target-train stats) | 1.6731 ± 0.0027 (n=3) | 2.196 ± 0.0094 (n=3) |
| target-only (upper reference, uses target labels) | 1.6736 ± 0.025 (n=3) | 2.205 ± 0.026 (n=3) |
| CORAL λ=10 (target-train stats) | 1.6981 ± 0.031 (n=3) | 2.2112 ± 0.01 (n=3) |

- MAE: CORAL − target-stats = +0.02502 (max seed std 0.031) → within seed std
- RMSE: CORAL − target-stats = +0.01523 (max seed std 0.01) → worse by more than seed std

Label-free gap, source-val vs target-val features (MMD² RBF, median heuristic; CORAL distance):

| model | MMD² | CORAL distance |
|---|---|---|
| source-only | 0.059411 ± 0.0053 (n=3) | 0.0013991 ± 0.00011 (n=3) |
| CORAL λ=10 | 0.051916 ± 0.0083 (n=3) | 0.00025232 ± 0.00013 (n=3) |

Source-val MSE (norm): source-only 0.071854 ± 0.00051 (n=3), CORAL 0.073555 ± 0.0009 (n=3). CORAL best epochs [7, 4, 3], wall clock [10.17, 7.52, 6.31] s.

## energy

| setting | MAE | RMSE |
|---|---|---|
| naive train-mean (target-train stats) | 1720.8 (n=1) | 2268.2 (n=1) |
| source-only (source stats) | 802.88 ± 33 (n=3) | 1061.2 ± 34 (n=3) |
| source-only (target-train stats) | 561.61 ± 8.7 (n=3) | 824.4 ± 10 (n=3) |
| target-only (upper reference, uses target labels) | 524.44 ± 10 (n=3) | 794.52 ± 7 (n=3) |
| CORAL λ=10 (target-train stats) | 554.78 ± 10 (n=3) | 810.45 ± 15 (n=3) |

- MAE: CORAL − target-stats = -6.838 (max seed std 10) → within seed std
- RMSE: CORAL − target-stats = -13.96 (max seed std 15) → within seed std

Label-free gap, source-val vs target-val features (MMD² RBF, median heuristic; CORAL distance):

| model | MMD² | CORAL distance |
|---|---|---|
| source-only | 0.030985 ± 0.0086 (n=3) | 0.0057908 ± 0.00076 (n=3) |
| CORAL λ=10 | 0.026017 ± 0.0044 (n=3) | 9.7543e-05 ± 2.4e-05 (n=3) |

Source-val MSE (norm): source-only 0.077996 ± 0.0029 (n=3), CORAL 0.080012 ± 0.0032 (n=3). CORAL best epochs [10, 21, 11], wall clock [412.86, 757.4, 451.41] s.

## λ selection (seed 0, source-val MSE ratio to λ=0)

| domain | protocol | λ=0.1 | λ=1 | λ=10 |
|---|---|---|---|---|
| finance | full (all windows, <=20 epochs) | 1.0004 | 1.0003 | 0.9989 |
| finance_vol | full (all windows, <=20 epochs) | 0.9998 | 0.9992 | 0.9952 |
| climate | full (all windows, <=20 epochs) | 1.0071 | 0.9945 | 1.0019 |
| energy | bake-off (every 4th window, <=15 epochs) | 1.0500 ✗ | 0.9962 | 1.0260 |

Qualifying: [1.0, 10.0]; chosen λ = 10.

## Staircase criteria (CLAUDE.md Decisions)

CORAL must beat the source-only target-stats row; energy improvement must exceed seed std; climate and finance must be no worse than seed noise. Seed std = max of the two rows' sample stds. Paired deltas are per seed (CORAL − target-stats, same seed).

- **finance** (no worse than seed noise): MAE Δ +3.333e-06 vs std 2.9e-05, paired per seed [-1e-05, 3e-05, -1e-05] → met; RMSE Δ +6.667e-06 vs std 2.1e-05, paired per seed [-1e-05, 3e-05, 0.0] → met
- **finance_vol** (no worse than seed noise): MAE Δ +5e-05 vs std 0.0002, paired per seed [2e-05, -0.00014, 0.00027] → met; RMSE Δ +2e-05 vs std 0.00012, paired per seed [2e-05, -0.0001, 0.00014] → met
- **climate** (no worse than seed noise): MAE Δ +0.02502 vs std 0.031, paired per seed [0.01885, 0.06109, -0.00489] → met; RMSE Δ +0.01523 vs std 0.01, paired per seed [0.00848, 0.03742, -0.0002] → NOT met
- **energy** (improvement > seed std): MAE Δ -6.838 vs std 10, paired per seed [-9.711, -4.17182, -6.63196] → NOT met; RMSE Δ -13.96 vs std 15, paired per seed [-23.97894, -5.1712, -12.71631] → NOT met
