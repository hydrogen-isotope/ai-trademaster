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

**A (2026-10-19, at the user's request, before any 1-day or 20-day result). Horizons of 1 and 20 trading days.**
- **Model.** The 5-day model is unchanged in every way except the horizon h, which is now a parameter (h ∈ {1, 5, 20}).
  For each h:
  - the drift is shrunk the same way and scaled to h days;
  - the GJR-GARCH variance is summed over h steps;
  - the idiosyncratic variance is h · idio²;
  - the cross-stock ridge and its decile tilt table are fitted on h-day returns;
  - the symmetric recalibration uses h-day PITs.
- **Selection.** The tilt shrink is chosen by CRPS on development 2021–2024 among {0.5, 0.75, 1.0}. 2025–26 is reported
  as *seen*, as for 5 days.
- **Gate for showing a horizon in the app**, decided before looking:
  1. on development, mean CRPS is below RW's;
  2. 80% coverage is within 0.75–0.85 on development and on the seen period.

  A horizon that fails is not shown.
- **Forward test.** Each horizon has its own shadow log (`v7/h1/`, `v7/h20/`) and the same promotion rule as 5 days,
  with block bootstrap blocks of max(5, h) dates, because 20-day outcomes overlap.
- **App.** 1 day is the default view; 5 days and 20 days ("1M") are selectable.

**Amendment A results (2026-10-19), against the gate written above:**

| market | horizon | shrink | dev CRPS (RW) | dev 80% cov. | seen 80% cov. | dev / seen hit | gate |
|---|---|---|---|---|---|---|---|
| NASDAQ | 1 day | 1.0 | 0.01853 (0.01854) | 79.7% | 79.6% | 50.9% / 51.1% | **pass** |
| NASDAQ | 20 days | 1.0 | 0.08529 (0.08529, lower by < 1e-5) | 79.5% | 79.8% | 51.5% / 51.4% | **fail** (CRPS not below RW) |
| SGX | 1 day | 0.5 | 0.01003 (0.01001) | 79.9% | 80.1% | 52.8% / 52.8% | **fail** (CRPS not below RW) |
| SGX | 20 days | 0.5 | 0.04455 (0.04466) | 79.9% | 79.7% | 52.3% / 52.5% | **pass** |

- The app therefore shows 1 day and 5 days for NASDAQ, and 5 days and 1 month for SGX.
- Forecasts for the failed horizons are still written to their shadow logs, but are not shown.

## Results

### Development, 2021-01-04 → 2024-12-31 (956,649 stock-days with a direction)

| model | CRPS | vs RW (95% CI) | 80% cov | 95% cov | up/down hit | Brier skill vs constant |
|---|---|---|---|---|---|---|
| RW | 0.04247 | — | 79.5% | 94.8% | — | −0.0005 |
| MKT | 0.04253 | +5.3e-5 [+0.5e-5, +10.6e-5] (worse) | 79.5% | 94.8% | 49.2% | −0.0021 |
| v7 s=0.5 | 0.04247 | −0.1e-5 [−3.8e-5, +4.0e-5] | 79.5% | 94.8% | 50.8% | −0.0008 |
| v7 s=0.75 | 0.04246 | −1.3e-5 [−4.8e-5, +2.4e-5] | 79.5% | 94.8% | 51.1% | −0.0009 |
| **v7 s=1.0 (chosen)** | **0.04246** | −1.5e-5 [−5.1e-5, +2.2e-5] | 79.5% | 94.8% | **51.3%** (p ≈ 1e-147) | −0.0015 |

v7 s=1.0 by v6 decile (bottom → top):
- hit rate: 56.3, 53.2, 51.3, 48.6, 49.2, 49.9, 50.4, 50.6, 51.5, 52.3%;
- mean P(up) vs actual up-rate: 44.4/43.4% (bottom) … 55.4/52.2% (top).

### Seen period, 2025-01-02 → 2026-10-09 (already used by v6's final test; 414,395 stock-days)

| model | CRPS | vs RW (95% CI) | 80% cov | 95% cov | up/down hit | Brier skill |
|---|---|---|---|---|---|---|
| RW | 0.04740 | — | 80.2% | 95.4% | — | −0.0001 |
| MKT | 0.04742 | +2.6e-5 [−4.2e-5, +8.2e-5] | 80.2% | 95.3% | 49.8% | −0.0008 |
| v7 s=1.0 | 0.04734 | −5.6e-5 [−11.3e-5, +0.3e-5] | 80.2% | 95.4% | 51.3% (p ≈ 1e-59) | −0.0020 |

By decile:
- hit rate: 57.7% (bottom), then 51.0, 49.5, 48.9, 50.0, 50.7, 50.9, 50.6, 51.7, and 51.6% (top);
- bottom decile: P(up) 42.6% vs actual 42.1%.

### Reading
- **Ranges** are well calibrated in both periods: the 80% and 95% intervals cover 79.5–80.2% and 94.8–95.4% of
  outcomes.
- **Up/down:** 51.3% overall in both periods. That is statistically real but small. Almost all of the edge is in the
  bottom decile, where "down" was right 56–58% of the time and P(up) matched reality closely. Middle deciles are coin
  flips. The top decile is slightly overconfident (P(up) about 3 points too high).
- **Market drift alone (MKT)** did not help; in development it was significantly worse than zero drift.
- **Overall CRPS gain vs RW** is not yet significant: both CIs touch 0. Whether v7 is promoted is decided by the forward
  shadow test from 2026-10-19, as written above.
