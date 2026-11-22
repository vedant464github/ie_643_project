# Bake-off: validation MSE relative to the naive validation MSE

- **Numerator:** each backbone's mean source-val MSE (normalised units) over seeds, from
  `experiments/bakeoff/summary.json`, unchanged.
- **Denominator:** the naive train-mean forecast's MSE on the same source-val windows. In z-space that forecast is 0,
  so the naive MSE is mean(Y_val²), recomputed by `results_clean/audit/audit_compute.py` from `prepare_pair` (in
  `audit_numbers.json` → `bakeoff`). The finance_vol value matches the stored
  `experiments/bakeoff/finance_vol_val_baselines.json` (0.80510).
- **Reading the ratio:** below 1 means the backbone beats the naive val MSE. The ± is the seed std divided by the same
  naive MSE.
- **Protocol** (from `experiments/bakeoff/selection_rule.json`): climate and finance_vol use 3 seeds and full
  training. Energy uses 2 seeds, every 4th training window and at most 15 epochs. The val set is always the full
  source val split. Raw finance was not in the bake-off.

| domain | naive val MSE (norm) | val windows |
|---|---|---|
| climate | 0.974819 | 610 |
| energy | 1.067650 | 9,785 |
| finance_vol | 0.805100 | 400 |

## Ratio per backbone and domain

| backbone | params (climate / energy) | climate | energy | finance_vol | mean (climate, energy) |
|---|---|---|---|---|---|
| gru | 9861 / 10488 | 0.0728 ± 0.0001 | 0.1226 ± 0.0037 | 1.0032 ± 0.0046 | 0.0977 |
| dlinear | 310 / 8112 | 0.0749 ± 0.0004 | 0.1054 ± 0.0001 | 1.0441 ± 0.0068 | 0.0902 |
| tcn | 34501 / 35128 | 0.0737 ± 0.0005 | 0.1071 ± 0.0051 | 1.0190 ± 0.0050 | 0.0904 |
| patchtst | 17797 / 19896 | 0.0780 ± 0.0019 | 0.1186 ± 0.0028 | 1.0115 ± 0.0053 | 0.0983 |

The last column is the unweighted mean of the climate and energy ratios. finance_vol is excluded there, following the
2026-10-23 amendment. All ratios are recomputable from the two sources above.

## Per-domain winners (lowest ratio)

| domain | winner | ratio | runner-up | ratio |
|---|---|---|---|---|
| climate | GRU | 0.0728 | TCN | 0.0737 |
| energy | DLinear | 0.1054 | TCN | 0.1071 |
| finance_vol | GRU | 1.0032 | PatchTST | 1.0115 |
| mean over climate and energy | DLinear | 0.0902 | TCN | 0.0904 |

Notes:
- **finance_vol:** every ratio is above 1, so no backbone beats the naive val MSE there. Its "winner" is the least-bad
  backbone. This is the reason the 2026-10-23 amendment excluded finance_vol from selection.
- **Ranks:** within a domain the denominator is the same for every backbone, so dividing does not change the per-domain
  ranks in `experiments/bakeoff/amendment.md`. The cross-domain mean of ratios is a different aggregate from the
  pre-registered mean rank.
- **DLinear vs TCN:** on mean ratio over climate and energy, DLinear and TCN differ by 0.0002. On mean rank they tie
  at 2.0. TCN was chosen by judgement, overriding the parameter-count tie-break (`experiments/bakeoff/amendment.md` §4).
