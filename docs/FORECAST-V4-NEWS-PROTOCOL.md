# 5-day forecast v4: news and "priced in" (rules written before any v4 result, 2026-10-18)

## Question
Does news help predict a stock's next 5 trading days beyond what v3 already does, and does the price reaction so far
tell us whether the news is already priced in?

## Not tested here (already answered on 2016–2026 data, 90 stocks)
Trend-following and its acceleration/tapering variants show no reliable effect on the next 5 days.
- Rank correlation of past 1/5/20/60/250-day returns with the next 5 days: −0.019 to +0.021, no |t| > 2.
- After a 10-day fall of 6%+, the price fell further 44.6% of the time overall:
  - 45.5% when the fall was steepening;
  - 44.3% when it was flattening;
  - 45.2% for any stock on any day.
- After big moves, the size of the next move is larger (typical 5.2% vs 3.8%). v3 already uses that by widening its
  ranges.

## Data
- **News:** the Alpaca news API (Benzinga), headlines plus summaries, 2016 to 16 Oct 2026, for the 90-stock universe.
- **Prices:** `private/data/daily_universe` (Alpaca, split-adjusted).

## Scoring
FinBERT (`ProsusAI/finbert`, run on the CPU). Each item gets a sentiment score = P(positive) − P(negative), computed
on its headline plus summary (truncated to 128 tokens).

## Features per stock at close *t* (no foresight)
- **Timing:** an item belongs to close *t* if it was published after the previous close (16:00 ET) and at or before
  16:00 ET on day *t*. Weekend and holiday news rolls into the next trading close.
- **News level:**
  - `n1` = log(1 + item count) on day *t*;
  - `s1` = mean sentiment on day *t* (0 if no news);
  - `s5` = mean sentiment over days *t−4..t*.
- **Priced in:** the stock's return over the same window, in 20-day volatility units: `z1` for day *t*, `z5` for
  *t−4..t*.
  - `pi1 = s1 × z1`, `pi5 = s5 × z5`. Positive means the price already moved the way the news points.
  - `gap1 = s1 − tanh(z1)`: news that the price has not reflected.
- **Processing:** every feature is ranked across stocks each day, as in v2. Stocks with no news get the neutral value.

## Model
v4 = v3 (GJR-GARCH-t, symmetric recalibration) plus a mean from one pooled ridge on the news features, trained like v2
(target = the 1..5-day return relative to the average stock, in volatility units; retrained every 21 days on an
expanding window; a training row is used only if its outcome is known by the origin).

## Development vs final test
- **Development:** origins from 2017-01-03 to 2025-10-10.
  - The ridge penalty is chosen from {1, 10, 100, 1000} by pooled development IC at h = 5.
  - **v4 is adopted only if**, over development, (a) its pooled IC has t > 2 *and* (b) its pooled CRPS on MU, NVDA,
    AMD and AAPL is lower than v3sym's.
- **Final test, run once:** the same last 250 origins as v2/v3 (2025-10-13 to 2026-10-09). The same rules apply:
  - calibration for each stock at every horizon;
  - pooled CRPS skill > 0 vs the random walk;
  - cross-stock direction > 50% at p < 0.05.

  These results are reported whatever the outcome. If v4 was not adopted, v3 stays live.
- **Hold-out 12–16 Oct:** reported for information only.
