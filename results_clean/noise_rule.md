# Single rule for "gain vs noise"

For a method M compared with the reference row R = 'source-only, target-train stats' on one (domain, metric): d = mean(M) − mean(R) over seeds; s_R = std of R; s_M = std of M (ddof=1). Verdict: |d| > max(s_R, s_M) → "exceeds both stds"; s_R < |d| ≤ s_M → "exceeds the target-stats std only"; |d| ≤ min(s_R, s_M) → "within both". The remaining case s_M < |d| ≤ s_R does not fit the fixed list and is printed as "within the target-stats std (exceeds method std)" so it is never silently relabelled. The verdict is about magnitude; the direction column says whether M is lower (better) or higher (worse) error. Paired per-seed differences use the same seed number in both rows (M seed s − R seed s); they are reported as extra information and do not change the verdict.

This is the only rule used in results_clean/ (headline, APPROVED_CLAIMS). Target test set, seeds 0-2.

| domain | method vs SO-tgt | metric | mean(M) | mean(R) | d = mean diff | paired diffs (s0; s1; s2) | paired mean | paired std | s_R (target-stats std) | s_M | max(s_R, s_M) | direction | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| energy | CORAL lambda=10 | MAE | 554.78 | 561.61 | -6.84 | -9.71; -4.17; -6.63 | -6.84 | 2.78 | 8.75 | 10.35 | 10.35 | lower error | within both |
| energy | CORAL lambda=10 | RMSE | 810.45 | 824.40 | -13.96 | -23.98; -5.17; -12.72 | -13.96 | 9.46 | 10.23 | 14.58 | 14.58 | lower error | exceeds the target-stats std only |
| energy | CORAL lambda=1 (sensitivity) | MAE | 561.10 | 561.61 | -0.51 | -3.55; -0.91; 2.92 | -0.51 | 3.25 | 8.75 | 7.42 | 8.75 | lower error | within both |
| energy | CORAL lambda=1 (sensitivity) | RMSE | 816.91 | 824.40 | -7.49 | -15.98; -1.92; -4.57 | -7.49 | 7.47 | 10.23 | 11.82 | 11.82 | lower error | within both |
| energy | AdaRNN-style lambda=10 | MAE | 560.19 | 561.61 | -1.42 | 0.94; -12.26; 7.06 | -1.42 | 9.87 | 8.75 | 2.00 | 8.75 | lower error | within both |
| energy | AdaRNN-style lambda=10 | RMSE | 811.97 | 824.40 | -12.43 | -15.93; -19.33; -2.03 | -12.43 | 9.16 | 10.23 | 1.09 | 10.23 | lower error | exceeds both stds |
| energy | target-only (labels, reference) | MAE | 524.44 | 561.61 | -37.17 | -43.57; -50.89; -17.06 | -37.17 | 17.80 | 8.75 | 10.02 | 10.02 | lower error | exceeds both stds |
| energy | target-only (labels, reference) | RMSE | 794.52 | 824.40 | -29.88 | -37.82; -41.53; -10.30 | -29.88 | 17.06 | 10.23 | 6.95 | 10.23 | lower error | exceeds both stds |
| climate | CORAL lambda=10 | MAE | 1.698 | 1.673 | 0.025 | 0.019; 0.061; -0.005 | 0.025 | 0.033 | 0.003 | 0.031 | 0.031 | higher error | exceeds the target-stats std only |
| climate | CORAL lambda=10 | RMSE | 2.211 | 2.196 | 0.015 | 0.008; 0.037; -0.000 | 0.015 | 0.020 | 0.009 | 0.010 | 0.010 | higher error | exceeds both stds |
| climate | CORAL lambda=1 (sensitivity) | MAE | 1.667 | 1.673 | -0.006 | 0.006; 0.001; -0.024 | -0.006 | 0.016 | 0.003 | 0.014 | 0.014 | lower error | exceeds the target-stats std only |
| climate | CORAL lambda=1 (sensitivity) | RMSE | 2.183 | 2.196 | -0.013 | -0.005; -0.010; -0.023 | -0.013 | 0.010 | 0.009 | 0.011 | 0.011 | lower error | exceeds both stds |
| climate | AdaRNN-style lambda=10 | MAE | 1.679 | 1.673 | 0.006 | 0.004; 0.005; 0.010 | 0.006 | 0.003 | 0.003 | 0.005 | 0.005 | higher error | exceeds both stds |
| climate | AdaRNN-style lambda=10 | RMSE | 2.203 | 2.196 | 0.007 | 0.012; 0.011; -0.001 | 0.007 | 0.007 | 0.009 | 0.008 | 0.009 | higher error | within both |
| climate | target-only (labels, reference) | MAE | 1.674 | 1.673 | 0.001 | -0.018; -0.007; 0.026 | 0.001 | 0.023 | 0.003 | 0.025 | 0.025 | higher error | within both |
| climate | target-only (labels, reference) | RMSE | 2.205 | 2.196 | 0.009 | -0.009; 0.003; 0.033 | 0.009 | 0.022 | 0.009 | 0.026 | 0.026 | higher error | within both |
| finance | CORAL lambda=10 | MAE | 0.007177 | 0.007173 | 0.000003 | -0.000010; 0.000030; -0.000010 | 0.000003 | 0.000023 | 0.000006 | 0.000029 | 0.000029 | higher error | within both |
| finance | CORAL lambda=10 | RMSE | 0.009627 | 0.009620 | 0.000007 | -0.000010; 0.000030; 0.000000 | 0.000007 | 0.000021 | 0.000010 | 0.000021 | 0.000021 | higher error | within both |
| finance | CORAL lambda=1 (sensitivity) | MAE | 0.007170 | 0.007173 | -0.000003 | 0.000000; 0.000000; -0.000010 | -0.000003 | 0.000006 | 0.000006 | 0.000010 | 0.000010 | lower error | within both |
| finance | CORAL lambda=1 (sensitivity) | RMSE | 0.009623 | 0.009620 | 0.000003 | 0.000000; 0.000000; 0.000010 | 0.000003 | 0.000006 | 0.000010 | 0.000006 | 0.000010 | higher error | within both |
| finance | AdaRNN-style lambda=10 | MAE | 0.007183 | 0.007173 | 0.000010 | 0.000020; 0.000010; 0.000000 | 0.000010 | 0.000010 | 0.000006 | 0.000012 | 0.000012 | higher error | exceeds the target-stats std only |
| finance | AdaRNN-style lambda=10 | RMSE | 0.009633 | 0.009620 | 0.000013 | 0.000030; 0.000020; -0.000010 | 0.000013 | 0.000021 | 0.000010 | 0.000031 | 0.000031 | higher error | exceeds the target-stats std only |
| finance | target-only (labels, reference) | MAE | 0.007157 | 0.007173 | -0.000017 | 0.000030; -0.000040; -0.000040 | -0.000017 | 0.000040 | 0.000006 | 0.000038 | 0.000038 | lower error | exceeds the target-stats std only |
| finance | target-only (labels, reference) | RMSE | 0.009603 | 0.009620 | -0.000017 | 0.000030; -0.000050; -0.000030 | -0.000017 | 0.000042 | 0.000010 | 0.000049 | 0.000049 | lower error | exceeds the target-stats std only |
| finance_vol | CORAL lambda=10 | MAE | 0.005207 | 0.005157 | 0.000050 | 0.000020; -0.000140; 0.000270 | 0.000050 | 0.000207 | 0.000203 | 0.000031 | 0.000203 | higher error | within the target-stats std (exceeds method std) |
| finance_vol | CORAL lambda=10 | RMSE | 0.006430 | 0.006410 | 0.000020 | 0.000020; -0.000100; 0.000140 | 0.000020 | 0.000120 | 0.000115 | 0.000010 | 0.000115 | higher error | within the target-stats std (exceeds method std) |
| finance_vol | CORAL lambda=1 (sensitivity) | MAE | 0.005180 | 0.005157 | 0.000023 | 0.000010; -0.000120; 0.000180 | 0.000023 | 0.000150 | 0.000203 | 0.000062 | 0.000203 | higher error | within both |
| finance_vol | CORAL lambda=1 (sensitivity) | RMSE | 0.006413 | 0.006410 | 0.000003 | 0.000010; -0.000090; 0.000090 | 0.000003 | 0.000090 | 0.000115 | 0.000029 | 0.000115 | higher error | within both |
| finance_vol | AdaRNN-style lambda=10 | MAE | 0.005310 | 0.005157 | 0.000153 | 0.000210; 0.000010; 0.000240 | 0.000153 | 0.000125 | 0.000203 | 0.000131 | 0.000203 | higher error | within the target-stats std (exceeds method std) |
| finance_vol | AdaRNN-style lambda=10 | RMSE | 0.006490 | 0.006410 | 0.000080 | 0.000140; -0.000010; 0.000110 | 0.000080 | 0.000079 | 0.000115 | 0.000082 | 0.000115 | higher error | within both |
| finance_vol | target-only (labels, reference) | MAE | 0.005267 | 0.005157 | 0.000110 | 0.000080; -0.000190; 0.000440 | 0.000110 | 0.000316 | 0.000203 | 0.000123 | 0.000203 | higher error | within both |
| finance_vol | target-only (labels, reference) | RMSE | 0.006460 | 0.006410 | 0.000050 | 0.000050; -0.000140; 0.000240 | 0.000050 | 0.000190 | 0.000115 | 0.000075 | 0.000115 | higher error | within both |

## finance_vol (and finance) statement, scoped

- **Source validation** (Nifty IT |log-return|, normalised): naive train-mean val MSE = 0.8051 (`experiments/bakeoff/finance_vol_val_baselines.json`); final TCN source-only val MSE per seed = 0.8180, 0.8251, 0.8182 (`final_3seed/tcn/runs/finance_vol_seed*.json`). No seed beats the naive val MSE; in the bake-off no backbone did (`finance_vol_val_baselines.json`). This is why finance_vol is a negative control (CLAUDE.md 2026-10-23).
- **Target test** (Nifty Pharma |log-return|): RMSE: naive 0.00678, persistence 0.00837, mean of last 5 0.00664, SO-tgt 0.00641 ± 0.00012, CORAL λ=10 0.00643 ± 0.00001. The target-test comparison is reported, not used to choose anything; the negative-control label rests on source validation.
- Raw finance, source validation: naive train-mean val MSE = 0.7984 (recomputed: mean(Y_val²) in source-normalised space; finance_vol recomputed the same way = 0.8051, JSON 0.8051); TCN val MSE per seed = 0.8053, 0.8105, 0.8048. No seed beats the naive val MSE.

