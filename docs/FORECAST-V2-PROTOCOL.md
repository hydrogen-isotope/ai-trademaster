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

---

## Results (added after the run, 2026-10-17; nothing above was changed)
**Development (2020-01-02 → 2025-10-10)**
- Ridge penalty chosen: 10. The pooled IC was +0.0082 (t ≈ 0.5), and the cross-stock direction hit 49.9%: no
  detectable edge.
- The mean model did **not** beat zero mean on development CRPS, so per the protocol it is **not used**. The variant
  shipped is GJR-GARCH-t with self-correcting ranges ("v1cal").

**Final test (2025-10-13 → 2026-10-09, 250 origins)**

| Rule | Result |
|---|---|
| 1. Calibration, all 4 stocks, all horizons | **PASS** (MU 80% hit rate 0.76–0.79, AMD 0.75–0.79, NVDA 0.78–0.81, AAPL 0.79–0.82) |
| 2. Pooled CRPS skill vs random walk > 0 | **PASS** (+0.96%) |
| 3. Cross-stock direction > 50%, p < 0.05 | **FAIL** (50.56%, p = 0.38; the mean model's IC on the test was +0.040) |

80% hit rates per variant on the test, for comparison:
- **MU:** v1 0.71–0.74; with self-correcting ranges 0.76–0.79.
- **AMD:** v1 0.73–0.76; with self-correcting ranges 0.75–0.79.

**Reading.** The self-correcting ranges fixed the calibration failures. Direction remains unpredictable at this
horizon for these stocks. The median of the recalibrated forecast called the 5-day direction right 57–60% of the time
for MU, AMD and AAPL in the test year, but that is mostly a strong uptrend leaking into the recalibration. Treat it as
regime-dependent, not as an edge; it will lag if the trend turns.

---

## v3 amendment (written before any v3 result, 2026-10-17)
**Why.** A hold-out of 12–16 Oct 2026 with the shipped variant ("v1cal") put 10 of 20 closes inside the 80% range
(the old v1 managed 14 of 20). MU got 0 of 5. Its recalibration had learned "too low on the upside" from an uptrend
year, so it shifted the whole range up just before a −14.6% week. A shifting recalibration follows trends.

**v3 change (one change only).** The recalibration becomes **symmetric**. From past PIT values *u* (same window of 250,
same shrinkage n/(n+100), same no-foresight rule), take the distribution of |u − 0.5|. Each central interval of nominal
coverage *c* is widened or narrowed to the level *c′* = 2 · quantile(|u − 0.5|, c), keeping it centred on the model's
median. The median itself is never moved. Everything else is v1: GJR-GARCH-t, zero mean.

**Selection and pass rules.**
- Over the **development** period (2020-01-02 → 2025-10-10), whichever of "v1cal" and "v3sym" has the lower pooled
  CRPS on the 4 stocks is chosen.
- The chosen variant then faces **the same final-test rules as before**: calibration for each stock at every horizon,
  and pooled CRPS skill > 0. If it fails, the dashboard labels the failing stocks NOT VALIDATED.
- The 12–16 Oct hold-out is reported for information only. It is one week and cannot decide anything.
