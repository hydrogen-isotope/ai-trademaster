# v7: 5-day price forecast for every ranked NASDAQ stock (rules written before any v7 result, 2026-10-18)

## Why
v6 (adopted 2026-10-18) ranks stocks against each other. The user also wants each stock's own price forecast: where it
will be in 5 trading days, and how likely it is to be higher. v7 builds that from the parts that have been tested:
- **Market direction (QQQ):** no timing skill in v1–v5, in the user's fear rule, or in v4's AI-fear news. v7 therefore
  uses only a shrunk long-run drift for the market.
- **Stock vs peers:** v6's ridge model, which held up out of sample (final-test IC 0.059).
- **Size of moves:** the market's volatility (GJR-GARCH-t), the stock's beta, and its stock-specific volatility.

**Expectations, stated in advance:** well-calibrated ranges, and an up/down hit rate of about 51–54% per stock. That is
a small edge.

## Forecast for stock i at close t (horizon 5 trading days, log returns)
- **Market drift** μ_m: half the expanding mean of QQQ's daily log return since 2016, × 5.
- **Market variance** V_m: the sum of the next 5 expected daily variances from a GJR-GARCH(1,1)-t fitted to QQQ.
  - The fit is refitted on the first trading day of each month, using data up to that day.
  - Daily variances are filtered with that month's parameters.
- **Beta and stock-specific vol:** `beta252` and `idio_vol60` from the v6 feature panel. Missing values default to:
  beta = 1, and idio_vol = the universe median that day.
- **Tilt** τ: the v6 ridge model is refitted monthly (walk-forward, as in v6). Each month its in-sample predictions on
  the training rows (all of them before the refit day) are split into deciles. The tilt for decile d is the mean of the
  5-day log return vs the universe median among those training rows.
  - Today's prediction percentile picks the decile.
  - The tilt is multiplied by a shrink factor s.
- **Distribution:** Student-t with 5 degrees of freedom.
  - mean m = β·μ_m + s·τ;
  - standard deviation σ = √(β²·V_m + 5·idio²).
- **Recalibration:** symmetric recalibration (`daily_v2.recal_levels_sym`, the v3 method). It is pooled over all stocks,
  using each forecast's PIT from the last 250 trading days whose outcomes were known before the refit day. It is
  recomputed monthly, separately for each model.
- **Outputs:**
  - median price;
  - 80% and 95% ranges;
  - P(up) = 1 − calibrated CDF(0);
  - P(beats market).

## Models compared
- **RW:** m = 0 (random walk), same σ and recalibration method.
- **MKT:** m = β·μ_m (no tilt).
- **v7:** m = β·μ_m + s·τ.

## Periods
- **Development:** forecasts dated 2021-01-04 → 2024-12-31. The only choice is s ∈ {0.5, 0.75, 1.0}, picked by the
  lowest mean CRPS. Nothing else is tuned.
- **Seen period:** 2025-01-02 → latest complete. Reported once, labelled *seen*, because v6's final test already used
  it.
- **Forward shadow test (the real confirmation):** forecasts made daily from 2026-10-19, written to
  `private/reports/v7/shadow.jsonl` before outcomes exist and scored when they mature. After ≥ 60 forecast dates,
  v7 is **promoted** only if both hold:
  1. its mean CRPS is lower than RW's, with a 95% block-bootstrap CI (blocks of 5 dates) excluding 0;
  2. its pooled 80% coverage is within 75–85%.

  Until then the dashboard labels v7 *shadow*.

## Metrics (every model, every period)
- CRPS: pinball approximation on a 199-level grid; skill vs RW with a block-bootstrap CI over dates.
- Coverage of the 80% and 95% ranges.
- Direction hit rate: sign of the median vs sign of the realised return; binomial p vs 50%. Also reported vs the
  realised up-rate.
- Brier score of P(up), and its skill vs a constant equal to the past up-rate.
- All of the above split by v6 decile.

## Amendments
(none yet)

## Results
(not yet run)
