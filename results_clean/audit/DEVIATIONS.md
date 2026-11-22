# Deviations: rows with large seed spread, or rows a reader could flag

Source: `results_clean/canonical.csv`, the per-seed JSONs it cites, and `results_clean/audit/audit_numbers.json`
(computed by `results_clean/audit/audit_compute.py`). The std is the sample std (ddof=1) over seeds 0-2. "Best epoch"
is the epoch whose weights were restored. "Stopping epoch" is the last epoch trained. Early stopping uses patience 3
on source-val MSE.

## Which rows qualify

Seed std as a percentage of the mean, for every 3-seed row in canonical.csv (MAE std / MAE mean, RMSE std / RMSE mean):

| domain | row | MAE std % | RMSE std % | above 3%? |
|---|---|---|---|---|
| energy | source-only, source stats | 4.06 | 3.21 | yes |
| energy | source-only, target-train stats | 1.56 | 1.24 | no |
| energy | CORAL lambda=10 | 1.87 | 1.80 | no |
| energy | CORAL lambda=1 (sensitivity) | 1.32 | 1.45 | no |
| energy | AdaRNN-style lambda=10 | 0.36 | 0.13 | no |
| energy | target-only (labels, reference) | 1.91 | 0.88 | no |
| climate | source-only, source stats | 20.25 | 17.48 | yes |
| climate | source-only, target-train stats | 0.16 | 0.43 | no |
| climate | CORAL lambda=10 | 1.82 | 0.47 | no |
| climate | CORAL lambda=1 (sensitivity) | 0.81 | 0.51 | no |
| climate | AdaRNN-style lambda=10 | 0.33 | 0.38 | no |
| climate | target-only (labels, reference) | 1.47 | 1.19 | no |
| finance | source-only, source stats | 0.00 | 0.12 | no |
| finance | source-only, target-train stats | 0.08 | 0.10 | no |
| finance | CORAL lambda=10 | 0.40 | 0.22 | no |
| finance | CORAL lambda=1 (sensitivity) | 0.14 | 0.06 | no |
| finance | AdaRNN-style lambda=10 | 0.16 | 0.32 | no |
| finance | target-only (labels, reference) | 0.53 | 0.51 | no |
| finance_vol | source-only, source stats | 3.86 | 1.73 | yes |
| finance_vol | source-only, target-train stats | 3.93 | 1.80 | yes |
| finance_vol | CORAL lambda=10 | 0.59 | 0.16 | no |
| finance_vol | CORAL lambda=1 (sensitivity) | 1.21 | 0.45 | no |
| finance_vol | AdaRNN-style lambda=10 | 2.47 | 1.26 | no |
| finance_vol | target-only (labels, reference) | 2.34 | 1.17 | no |

Rows included below:
- **Above 3%:** energy source-only/source stats; climate source-only/source stats; finance_vol source-only/source
  stats; finance_vol source-only/target-train stats.
- **Named as reader-flaggable:** energy and climate source-stats rows (already above) and climate CORAL λ=10 (1.82%
  MAE, 0.47% RMSE, below 3%).

The |z| > 3 share counts every value in the target-test input windows (`X_test`, overlapping windows included),
z-scored with the stats that row feeds the model. The inputs do not depend on the seed, so the share is the same for
all three seeds. Two context columns, for the same inputs:

| domain | stats | share \|z\|>3 | mean z | max \|z\| | source-train input z range | share of target-test inputs outside that range |
|---|---|---|---|---|---|---|
| energy | source-train | 0.71% | −1.53 | 3.19 | [−2.23, 3.78] | 22.99% |
| energy | target-train | 1.13% | – | 4.34 | (same) | 0.11% |
| climate | source-train | 41.50% | +2.43 | 4.44 | [−2.70, 2.08] | 66.97% |
| climate | target-train | 0.00% | – | 2.03 | (same) | 0.00% |
| finance_vol | source-train | 0.25% | −0.22 | 3.41 | [−1.00, 9.81] | 0.00% |
| finance_vol | target-train | 0.25% | – | 3.49 | (same) | 2.99% (all below the lower end: zero returns map to z ≈ −1.05 under target stats) |

## Per-row evidence

Source-only rows with source stats and with target-train stats come from **the same trained model per seed**, so they
share best and stopping epochs and source-val MSE. Only the inference stats differ.

#### energy / source-only, source stats

| seed | MAE | RMSE | best epoch | stopping epoch | epoch cap | source-val MSE (norm) | share of target-test input values with \|z\|>3 (source-train stats) |
|---|---|---|---|---|---|---|---|
| 0 | 771.13586 | 1029.55933 | 20 | 23 | 40 | 0.07549 | 0.71% (same for every seed) |
| 1 | 801.23871 | 1056.67297 | 14 | 17 | 40 | 0.08115 | 0.71% (same for every seed) |
| 2 | 836.27594 | 1097.2644 | 20 | 23 | 40 | 0.07735 | 0.71% (same for every seed) |

#### climate / source-only, source stats

| seed | MAE | RMSE | best epoch | stopping epoch | epoch cap | source-val MSE (norm) | share of target-test input values with \|z\|>3 (source-train stats) |
|---|---|---|---|---|---|---|---|
| 0 | 2.66901 | 3.41855 | 6 | 9 | 20 | 0.07169 | 41.50% (same for every seed) |
| 1 | 3.62953 | 4.45214 | 7 | 10 | 20 | 0.07243 | 41.50% (same for every seed) |
| 2 | 2.53704 | 3.25719 | 5 | 8 | 20 | 0.07144 | 41.50% (same for every seed) |

#### climate / CORAL lambda=10

| seed | MAE | RMSE | best epoch | stopping epoch | epoch cap | source-val MSE (norm) | share of target-test input values with \|z\|>3 (target-train stats) |
|---|---|---|---|---|---|---|---|
| 0 | 1.69168 | 2.20896 | 7 | 10 | 20 | 0.07258 | 0.00% (same for every seed) |
| 1 | 1.73159 | 2.22257 | 4 | 7 | 20 | 0.07434 | 0.00% (same for every seed) |
| 2 | 1.67095 | 2.20214 | 3 | 6 | 20 | 0.07374 | 0.00% (same for every seed) |

#### finance_vol / source-only, source stats

| seed | MAE | RMSE | best epoch | stopping epoch | epoch cap | source-val MSE (norm) | share of target-test input values with \|z\|>3 (source-train stats) |
|---|---|---|---|---|---|---|---|
| 0 | 0.00516 | 0.00639 | 1 | 4 | 20 | 0.81799 | 0.25% (same for every seed) |
| 1 | 0.00526 | 0.00649 | 2 | 5 | 20 | 0.82509 | 0.25% (same for every seed) |
| 2 | 0.00488 | 0.00627 | 2 | 5 | 20 | 0.81819 | 0.25% (same for every seed) |

#### finance_vol / source-only, target-train stats

| seed | MAE | RMSE | best epoch | stopping epoch | epoch cap | source-val MSE (norm) | share of target-test input values with \|z\|>3 (target-train stats) |
|---|---|---|---|---|---|---|---|
| 0 | 0.00522 | 0.00642 | 1 | 4 | 20 | 0.81799 | 0.25% (same for every seed) |
| 1 | 0.00532 | 0.00652 | 2 | 5 | 20 | 0.82509 | 0.25% (same for every seed) |
| 2 | 0.00493 | 0.00629 | 2 | 5 | 20 | 0.81819 | 0.25% (same for every seed) |

## What the evidence supports

**energy, source-only / source stats (MAE std 4.06%, RMSE std 3.21%).**
- Supported: under AEP stats, the COMED test inputs sit on average 1.53 std below the source-train mean. 23% of input
  values fall outside the range of source-train inputs, but only 0.71% have |z| > 3.
- Supported: the same three models evaluated with target-train stats have a 1.56% MAE std (561.3, 570.5, 553.0 MW). So
  the extra spread appears only when the inputs are shifted.
- Supported (`ERROR_STRUCTURE.md`): the spread across seeds is largest at night (hours 22-1, std 85-104 MW) and
  smallest around hours 3-5 (std 25-30 MW).
- Supported: seed order on target MAE (seed 0 < 1 < 2) does not follow seed order on source-val MSE (0 < 2 < 1).
- **Unknown:** why one seed's model responds worse than another's to shifted inputs. Nothing in the files measures
  this.

**climate, source-only / source stats (MAE std 20.3%, RMSE std 17.5%).**
- Supported: under Shimla stats, 41.5% of Delhi test input values have |z| > 3, and 67% lie outside the source-train
  input range [−2.70, 2.08]. Under target-train stats, 0% do.
- Supported: seed 1 is the outlier (3.63 °C vs 2.67 and 2.54). The three seeds have almost equal source-val MSE
  (0.0717, 0.0724, 0.0714), so source validation did not predict which seed extrapolates worst.
- Supported (`ERROR_STRUCTURE.md`): the error and its seed spread are concentrated in March-September. The April-June
  MAE is 4.2-4.8 °C, with std 0.9-1.1. These months are also the warmest Delhi months, the ones furthest above the
  Shimla range.
- Supported: with target-train stats, the same models have a 0.16% MAE std.
- **Unknown:** the mechanism of the seed difference beyond "the inputs are out of range". It is not measured.

**climate, CORAL λ=10 (MAE std 1.82%, RMSE std 0.47%; below the 3% threshold).**
- Per seed vs source-only/target stats (same seed): +0.019, +0.061, −0.005 °C MAE. The mean difference (+0.025 °C)
  is driven mainly by seed 1.
- CORAL source-val MSE is higher than the source-only model's on every seed: 0.07258 / 0.07434 / 0.07374 vs
  0.07169 / 0.07243 / 0.07144.
- CORAL stopped earlier: best epochs 7 / 4 / 3 vs 6 / 7 / 5.
- No input-range issue: 0% of inputs have |z| > 3 under target-train stats.
- λ=10 was chosen by the pre-registered rule, which requires **every** domain's seed-0 source-val MSE to be within 5%
  of λ=0 (`src/run_coral.py:select`). For climate the λ=10 ratio was 1.0019 (`experiments/coral/lambda_selection.json`).
- Seed 0 is the reused grid run (`src/run_coral.py:248-257`).
- **Unknown:** why seed 1 is worse. There is no per-seed diagnostic beyond the gap metrics in the JSONs.

**finance_vol, source-only / source stats and / target-train stats (MAE std 3.86% and 3.93%; RMSE std 1.73% and 1.80%).**
- Supported: best epochs are 1, 2, 2 and training stopped at epochs 4, 5, 5. Every seed's source-val MSE (0.818,
  0.825, 0.818) is above the naive train-mean val MSE (0.8051, `experiments/bakeoff/finance_vol_val_baselines.json`),
  consistent with the documented negative-control status (CLAUDE.md Decisions and 2026-10-23).
- Supported: almost no out-of-range input (0.25% with |z| > 3 under either stats).
- Supported: seed 2 has the lowest error under both stats (0.00488 / 0.00493).
- Supported: the spread is about 4% of a quantity on which no model beats the mean-of-last-5 baseline MAE (0.00497).
- **Unknown:** why seed 2 differs. Models stopped after 1-2 useful epochs, and the files do not show what they
  learned.
