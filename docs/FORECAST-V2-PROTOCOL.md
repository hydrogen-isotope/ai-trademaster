# 5-day forecast v2: test protocol (written before any v2 result was computed, 2026-10-17)

## Why v2
v1 (GJR-GARCH-t volatility, zero-mean random walk) passed the 250-day check for NVDA and AAPL. It failed for MU and
AMD: their prices rose 4.6× and 2.4× during the test year, and the ranges were too narrow on the upside. v1 also has
no view on direction. v2 adds two things, and only these two:
1. **A pooled cross-stock mean model.** One ridge regression trained on all ~90 stocks at once. Inputs per stock and
   day:
   - the last 1- and 5-day return (short-term reversal);
   - 1-, 3-, 6- and 12-month momentum, skipping the most recent week;
   - 20-day volatility and its ratio to the 6-month level;
   - distance from the 52-week high;
   - 5-day volume surprise;
   - the stock's 5- and 20-day return relative to its sector ETF (SMH for semiconductors, QQQ otherwise);
   - SPY's 5- and 20-day return.

   Every input is ranked across stocks each day. The target is the stock's 1..5-day return relative to the average
   stock, divided by its 20-day volatility. The market part of the forecast is SPY's trailing 5-year mean daily
   return × horizon, shrunk by half.
2. **Self-correcting ranges** (quantile recalibration). At each origin, the forecast quantiles are re-mapped using the
   probability integral transform of the model's own *past* forecasts whose outcomes are already known: the last 250,
   shrunk toward no change with weight n/(n+100).

Volatility stays the v1 GJR-GARCH(1,1)-t.

## Data
Alpaca daily bars (SIP, split-adjusted), 95 symbols, Oct 2016 – 16 Oct 2026, in `private/data/daily_universe`.
Known bias: the universe is today's large Nasdaq stocks (survivorship). That flatters momentum, so the cross-stock
results are optimistic.

## No foresight
- A model used at origin *t* is trained only on samples whose whole target window ends on or before *t*: a training
  row at day *s* is used only if *s + h ≤ t*.
- Features use closes up to *t*.
- Retraining happens every 21 trading days on an expanding window (at least 3 years).
- Recalibration uses only past forecasts whose outcome dates are ≤ *t*.

## Development vs final test
- **Development (all choices made here):** origins from 2020-01-02 to 2025-10-10. The only choices are the ridge
  penalty (1, 10, 100, 1000) and whether to use the mean model at all.
  - The penalty is picked by pooled rank correlation (IC) at h=5.
  - The mean model is adopted if, over development, its CRPS beats v1 (zero mean) on the 4 target stocks pooled.
- **Final test, run once:** the last 250 origins with complete 5-day outcomes (2025-10-13 to 2026-10-09). Note: v1
  results on this window were already seen. v2's settings are fixed from the development period, but this window is
  not perfectly "unseen". The truly clean test is the forward record from 19 Oct 2026 onward.

## Pass rules on the final test (same gate as v1, plus two new criteria)
1. **Calibration**, each of MU, NVDA, AMD and AAPL, every horizon 1–5: |80% coverage − 0.80| ≤ 0.06 and
   |95% coverage − 0.95| ≤ 0.04.
2. **Probabilistic accuracy:** pooled CRPS over the 4 stocks better than the random walk (skill > 0).
3. **Direction (cross-stock)**, ~90 stocks, non-overlapping 5-day windows: the share of stocks whose
   beat-or-lag-the-average call was right must be above 50%, with a one-sided t-test over dates at p < 0.05.

Every result is reported, pass or fail. If v2 fails a rule, the dashboard keeps showing which rule failed.
