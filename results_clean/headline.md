# Headline: source-only vs adapted

'Adapted' = target-train-stat normalisation + CORAL λ=10 (the loss chosen by the pre-registered rule;
CLAUDE.md amendment 2026-11-15 (g)(a): this framing is a post-results presentation choice). Target test set,
mean over seeds 0-2, original scale. All numbers from `canonical.csv`.

Notation: A = source-only, source stats; B = source-only, target-train stats; C = adapted method
(CORAL λ=10 or AdaRNN-style λ=10, both with target-train stats); T = target-only (uses target labels).

- total reduction % = 100·(A − C)/A
- normalisation part = A − B; method part = B − C; total = A − C = (A − B) + (B − C)
- share of total drop: norm = (A − B)/(A − C), method = (B − C)/(A − C)
- gap to target-only = A − T; share of gap closed: norm = (A − B)/(A − T), method = (B − C)/(A − T), total = (A − C)/(A − T)
- Shares are only meaningful when there is a drop (A − C > 0) or a gap (A − T > 0) clearly larger than noise. A share is
  marked **n/m** (not meaningful) when its denominator is ≤ 0, or ≤ the larger seed std of the two rows involved, or
  below 2 rounding steps of the JSON values (2e-5 for finance/finance_vol, whose per-seed metrics are stored to 5 decimals).

## Method: CORAL lambda=10

| domain | metric | A: SO-src | B: SO-tgt | C: method | T: target-only | total red. % = 100(A−C)/A | norm part A−B (share of A−C) | method part B−C (share of A−C) | gap A−T | gap closed: norm (A−B)/(A−T) | method (B−C)/(A−T) | total (A−C)/(A−T) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| energy | MAE | 802.88 | 561.61 | 554.78 | 524.44 | 30.9 | 241.27 (97.2%) | 6.84 (2.8%) | 278.44 | 86.6% | 2.5% | 89.1% |
| energy | RMSE | 1061.17 | 824.40 | 810.45 | 794.52 | 23.6 | 236.76 (94.4%) | 13.96 (5.6%) | 266.65 | 88.8% | 5.2% | 94.0% |
| climate | MAE | 2.945 | 1.673 | 1.698 | 1.674 | 42.3 | 1.272 (102.0%) | -0.025 (-2.0%) | 1.272 | 100.0% | -2.0% | 98.1% |
| climate | RMSE | 3.709 | 2.196 | 2.211 | 2.205 | 40.4 | 1.513 (101.0%) | -0.015 (-1.0%) | 1.504 | 100.6% | -1.0% | 99.6% |
| finance | MAE | 0.007170 | 0.007173 | 0.007177 | 0.007157 | -0.1 | -0.000003 (n/m 50.0%) | -0.000003 (n/m 50.0%) | 0.000013 | n/m -25.0% | n/m -25.0% | n/m -50.0% |
| finance | RMSE | 0.009617 | 0.009620 | 0.009627 | 0.009603 | -0.1 | -0.000003 (n/m 33.3%) | -0.000007 (n/m 66.7%) | 0.000013 | n/m -25.0% | n/m -50.0% | n/m -75.0% |
| finance_vol | MAE | 0.005100 | 0.005157 | 0.005207 | 0.005267 | -2.1 | -0.000057 (n/m 53.1%) | -0.000050 (n/m 46.9%) | -0.000167 | n/m 34.0% | n/m 30.0% | n/m 64.0% |
| finance_vol | RMSE | 0.006383 | 0.006410 | 0.006430 | 0.006460 | -0.7 | -0.000027 (n/m 57.1%) | -0.000020 (n/m 42.9%) | -0.000077 | n/m 34.8% | n/m 26.1% | n/m 60.9% |

## Method: AdaRNN-style lambda=10

| domain | metric | A: SO-src | B: SO-tgt | C: method | T: target-only | total red. % = 100(A−C)/A | norm part A−B (share of A−C) | method part B−C (share of A−C) | gap A−T | gap closed: norm (A−B)/(A−T) | method (B−C)/(A−T) | total (A−C)/(A−T) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| energy | MAE | 802.88 | 561.61 | 560.19 | 524.44 | 30.2 | 241.27 (99.4%) | 1.42 (0.6%) | 278.44 | 86.6% | 0.5% | 87.2% |
| energy | RMSE | 1061.17 | 824.40 | 811.97 | 794.52 | 23.5 | 236.76 (95.0%) | 12.43 (5.0%) | 266.65 | 88.8% | 4.7% | 93.5% |
| climate | MAE | 2.945 | 1.673 | 1.679 | 1.674 | 43.0 | 1.272 (100.5%) | -0.006 (-0.5%) | 1.272 | 100.0% | -0.5% | 99.6% |
| climate | RMSE | 3.709 | 2.196 | 2.203 | 2.205 | 40.6 | 1.513 (100.5%) | -0.007 (-0.5%) | 1.504 | 100.6% | -0.5% | 100.1% |
| finance | MAE | 0.007170 | 0.007173 | 0.007183 | 0.007157 | -0.2 | -0.000003 (n/m 25.0%) | -0.000010 (n/m 75.0%) | 0.000013 | n/m -25.0% | n/m -75.0% | n/m -100.0% |
| finance | RMSE | 0.009617 | 0.009620 | 0.009633 | 0.009603 | -0.2 | -0.000003 (n/m 20.0%) | -0.000013 (n/m 80.0%) | 0.000013 | n/m -25.0% | n/m -100.0% | n/m -125.0% |
| finance_vol | MAE | 0.005100 | 0.005157 | 0.005310 | 0.005267 | -4.1 | -0.000057 (n/m 27.0%) | -0.000153 (n/m 73.0%) | -0.000167 | n/m 34.0% | n/m 92.0% | n/m 126.0% |
| finance_vol | RMSE | 0.006383 | 0.006410 | 0.006490 | 0.006460 | -1.7 | -0.000027 (n/m 25.0%) | -0.000080 (n/m 75.0%) | -0.000077 | n/m 34.8% | n/m 104.3% | n/m 139.1% |

## Sanity check against the old slides (verified, not copied)

- energy MAE: A = 802.9 -> C (CORAL λ=10) = 554.8 MW (slides said roughly 803 -> 555)
- climate MAE: A = 2.945 -> C = 1.698 °C (slides said roughly 2.95 -> 1.70)

