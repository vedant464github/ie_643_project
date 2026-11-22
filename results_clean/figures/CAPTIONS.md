# Figure captions

All figures: `results_clean/scripts/make_figures.py`, PNG, 200 dpi, white background. One colour per setting
everywhere (Okabe-Ito, colour-blind safe; checked with the dataviz palette validator): naive train-mean grey,
persistence / seasonal-naive reddish purple (second persistence variant hatched), source-only source stats orange
(dashed in line plots), source-only target-train stats sky blue (dotted), CORAL λ=10 blue (solid), CORAL λ=1 green,
AdaRNN-style λ=10 vermillion, target-only dark grey. Model curves (a, c, e) use the **seed-0** checkpoints only
(`experiments/final_3seed/tcn/checkpoints/*_seed0_source_only.pt`, `experiments/coral/<domain>/seed0.pt`), inference
only; recomputed seed-0 metrics reproduce the JSONs (`figures/_inference_check.txt`; energy differs by ≤ 0.0002 MW,
float summation order). Bar charts (b, d) use the 3-seed means/stds from `canonical.csv`.

**a_forecast_energy.png / a_forecast_climate.png / a_forecast_finance.png / a_forecast_finance_vol.png**
Actual target values vs one-step-ahead forecasts on the first contiguous span of the target test set
(energy: 168 consecutive hourly windows, 2017-06-20 09:00 to 2017-06-27 08:00; climate: 60 daily windows,
2025-01-25 to 2025-03-25; finance and finance_vol: 60 trading-day windows, 2025-02-06 to 2025-05-09). Each point is
the first horizon step (t+1) of one sliding test window, plotted at that step's timestamp, so the curve is a
rolling 1-step forecast; seed 0 only. Honest reading: at horizon step 1 the three models look close; the headline
differences are over all horizon steps (energy seed 0, MAE over the full test set: step 1 only 207 / 168 / 164 MW
for source stats / target stats / CORAL, all 24 steps 771 / 561 / 552 MW). For finance and finance_vol all three
model curves sit nearly on top of each other close to a constant, which is what a model that has learned roughly
the mean looks like.

**b_bars_energy.png / b_bars_climate.png / b_bars_finance.png / b_bars_finance_vol.png**
Target-test MAE of every setting as a percentage of the naive train-mean MAE (target-train stats) for that domain;
bars = mean over seeds 0-2, error bars = ±1 sample std (ddof=1) of the MAE expressed in the same percentage;
baselines are deterministic (no error bar). Dashed line = 100% (naive). Order and colours match canonical.md.
Lower is better. Target-only uses target labels and is a reference, not an adaptation method.

**c_energy_mae_by_hour.png**
Energy (COMED target test), seed 0: MAE in MW grouped by the hour of day of the forecast target, using all 24
horizon steps of every test window. Because the horizon is exactly 24 h, each window contributes exactly one
prediction to every hour bin, so each bin mixes all lead times equally. Normalising the target with target-train
statistics lowers the error at every hour; CORAL λ=10 is slightly lower than source-only with target stats at most
hours and slightly higher around 08:00-10:00. Seed 0 only; the 3-seed CORAL-vs-target-stats MAE difference
(−6.8 MW) is within both seed stds (noise_rule.md).

**d_all_domains_pct_naive.png**
One panel, all four domains: target-test MAE as % of each domain's naive train-mean MAE (mean ± std over seeds 0-2),
so domains with different units are on one scale. The persistence bar is the energy lag-24 seasonal-naive
(same hour yesterday) and the last-observed-value persistence elsewhere (lag 168 and mean-of-last-5 are in the
per-domain charts). Energy and climate models are far below naive; raw finance sits at ~100.7% (no model beats the
naive mean); finance_vol models are below the naive mean but not below the mean-of-last-5 baseline on MAE
(see canonical.md).

**e_scatter_coral_energy_climate.png**
Predicted vs actual on the target test set for CORAL λ=10, seed 0, every horizon step of every window
(energy 9,785 windows × 24 steps; climate 610 windows × 5 steps). Dashed line: y = x. Energy points are drawn with
high transparency because of their number. Both show regression towards the middle at the extremes
(high loads and the hottest days are under-predicted, the coldest days over-predicted).
