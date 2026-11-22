# Approved claims (poster and 2-page report)

Each claim is one sentence, carries its verified number, and names the row it comes from. Abbreviations for rows:
`canonical.csv` rows are named "domain / setting"; SO-src = source-only, source stats; SO-tgt = source-only,
target-train stats; all numbers are target-test, mean over seeds 0-2, std ddof=1, unless stated. Noise verdicts use
the single rule in `noise_rule.md` and only its three labels.

## Claims

1. **Headline (energy).** Adapting the source-trained TCN (target-train-stat normalisation + CORAL λ=10) lowers COMED
   target-test MAE from 802.88 to 554.78 MW, a 30.9% reduction (RMSE 1061.17 → 810.45 MW, 23.6%).
   *Rows: canonical.csv energy / SO-src, energy / CORAL lambda=10; headline.md energy MAE/RMSE.*

2. **Headline (climate).** The same adaptation lowers Delhi target-test MAE from 2.945 to 1.698 °C, a 42.3% reduction
   (RMSE 3.709 → 2.211 °C, 40.4%).
   *Rows: canonical.csv climate / SO-src, climate / CORAL lambda=10; headline.md climate.*

3. **Decomposition.** Almost all of that drop comes from normalising target inputs with target-train statistics: on
   energy 241.27 of 248.10 MW (97.2%) of the MAE drop is normalisation and 6.84 MW (2.8%) is CORAL; on climate
   normalisation accounts for more than the whole drop (1.272 °C) and CORAL adds 0.025 °C of error.
   *Rows: headline.md "Method: CORAL lambda=10", energy MAE and climate MAE (A−B, B−C).*

4. **Gap to target-only (energy).** The adapted model closes 89.1% of the energy MAE gap between SO-src and the
   label-trained target-only reference (86.6% from normalisation, 2.5% from CORAL) and remains 30.34 MW above
   target-only (554.78 vs 524.44 MW).
   *Rows: headline.md energy MAE (gap closed columns); canonical.csv energy / target-only.*

5. **CORAL vs the target-stats row, energy (noise rule).** On energy, CORAL λ=10 vs SO-tgt differs by −6.84 MW MAE,
   which is "within both" stds (8.75 and 10.35 MW), although it is lower on all three paired seeds (−9.71, −4.17,
   −6.63 MW); RMSE differs by −13.96 MW, which "exceeds the target-stats std only" (10.23 < 13.96 ≤ 14.58 MW).
   *Rows: noise_rule.md energy / CORAL lambda=10 / MAE and RMSE.*

6. **CORAL vs the target-stats row, climate (noise rule).** On climate, CORAL λ=10 is worse than SO-tgt: +0.025 °C MAE
   ("exceeds the target-stats std only") and +0.015 °C RMSE ("exceeds both stds").
   *Rows: noise_rule.md climate / CORAL lambda=10 / MAE and RMSE.*

7. **Pre-registered criteria.** The pre-registered success criterion (energy gain over the target-stats row larger
   than the seed std) is not met by CORAL λ=10 on MAE (|−6.84| ≤ 8.75 MW), consistent with CLAUDE.md amendment
   2026-10-25 (a).
   *Rows: noise_rule.md energy / CORAL lambda=10 / MAE; CLAUDE.md 2026-10-25 (a).*

8. **AdaRNN-style.** AdaRNN-style λ=10 reaches 560.19 MW MAE on energy (−1.42 MW vs SO-tgt, "within both") and
   811.97 MW RMSE (−12.43 MW, "exceeds both stds"); on climate it is +0.006 °C MAE vs SO-tgt ("exceeds both stds",
   worse) and +0.007 °C RMSE ("within both").
   *Rows: canonical.csv energy|climate / AdaRNN-style lambda=10; noise_rule.md AdaRNN rows.*

9. **Energy vs seasonal-naive.** The best simple energy baseline is same-hour-yesterday (lag 24, 860.85 MW MAE); the
   adapted model's MAE is 64.4% of it, while SO-src is 93.3% of it.
   *Rows: canonical.csv energy / seasonal-naive lag 24; energy / CORAL lambda=10 and SO-src (MAE % best baseline).*

10. **Climate vs persistence (small margin).** On climate, repeating the last observed temperature already gives
    1.733 °C MAE; the adapted model is only 0.035 °C better (1.698 °C, 98.0% of persistence; RMSE 2.211 vs 2.350 °C,
    94.1%), and SO-tgt is 1.673 °C (96.5%).
    *Rows: canonical.csv climate / persistence, climate / CORAL lambda=10, climate / SO-tgt (MAE/RMSE % best baseline).*

11. **Finance (raw log-returns) is a negative control.** On source validation no TCN seed beats the train-mean
    (val MSE 0.8048–0.8105 vs 0.7984), and on the target test every source-trained or adapted model sits at
    100.7–100.9% of the naive train-mean MAE (naive 0.007120 log-return units; target-only with labels 100.5%).
    *Rows: noise_rule.md "finance_vol (and finance) statement" (raw finance source val); canonical.csv finance rows (MAE % naive).*

12. **finance_vol is a negative control, scoped to source validation.** On Nifty IT |log-return| source validation no
    TCN seed beats the train-mean (val MSE 0.8180–0.8251 vs 0.8051), so finance_vol is reported as a negative
    control; on the target test the models' MAE is 102.6–106.8% of the mean-of-last-5 baseline (0.00497), and their
    RMSE is 96.1–97.7% of it (0.00664).
    *Rows: noise_rule.md finance_vol statement; canonical.csv finance_vol rows (MAE/RMSE % best baseline).*

13. **Units and normalisation rule.** All errors are reported in original units (MW, °C, log-return units) after
    inverting the z-score, with normalisation statistics from the training split only (source train, or target train
    for the target-stats rows, using target inputs only, never target labels).
    *Rows: canonical.md "Definitions"; src/data/prepare.py; CLAUDE.md Rules.*

14. **Post-hoc steps (disclose).** Three choices were made after seeing results: finance_vol was dropped from the
    backbone selection rule and TCN was picked over DLinear by judgment, overriding the parameter tie-break
    (2026-10-23); and the headline was reframed as source-only vs adapted with a new persistence/seasonal baseline and a
    single canonical table (2026-11-15 (g)), with no re-runs and no new training.
    *Rows: CLAUDE.md Amendments 2026-10-23 and 2026-11-15 (g); experiments/bakeoff/amendment.md.*

15. **λ sensitivity.** CORAL λ=1 (sensitivity check, not the pre-registered λ) gives 561.10 MW energy MAE (−0.51 MW vs
    SO-tgt, "within both") and 1.667 °C climate MAE (−0.006 °C, "exceeds the target-stats std only").
    *Rows: canonical.csv energy|climate / CORAL lambda=1; noise_rule.md CORAL lambda=1 rows.*

## Forbidden wording

- "within seed noise" as a blanket phrase. Use the noise-rule label for a specific (domain, method, metric).
- "ties the naive mean" (or "tie naive"). Say instead "at 100.7–100.9% of the naive train-mean MAE" (finance) with the row.
- Any claim not backed by a row above or in canonical.csv / headline.md / noise_rule.md.
- Also avoid: "CORAL improves energy" without the metric and the noise label, and "AdaRNN beats CORAL" (energy MAE
  560.19 vs 554.78 MW and RMSE 811.97 vs 810.45 MW; the methods were not compared under a pre-registered rule).

Where these phrases appear now (read-only scan, not fixed): report/overleaf/report.tex line 470 ("ties the naive"),
line 652 ("within noise"), line 659 ("within seed noise"); slides/IE643_Project_Slides.pptx slide 5 and its notes
("ties the naive"), slide 11 ("tie naive"), notes of slide 15 ("ties the naive"), slide 17 ("within seed noise");
slides/IE643_Project_Slides_v2.pptx slide 11 ("tie naive").
