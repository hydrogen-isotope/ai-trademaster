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

---

## Amendment A (2026-10-18, still before any v4 result): trend, ETF pressure, macro, nonlinear model
The user asked for three layers. All are added before any v4 number is computed.

**Trend and acceleration.** For each stock, in 20-day volatility units:
- slope over 3, 5 and 10 days, and the change between the last 3 days and the 3 before (steepening vs flattening);
- the curve (5-day slope minus 10-day slope);
- each of these split into its down part and its up part, so falls and rises can behave differently.

**ETF and forced-selling pressure (from Alpaca daily bars).**
- **The stock relative to its sector ETF** (SMH for semis, QQQ otherwise), over 1, 5 and 20 days.
- **Leveraged-ETF rebalancing pressure:**
  - funds: SOXL/SOXS for semis; TQQQ/SQQQ for everything else; MUU for MU and NVDL for NVDA where Alpaca has them;
  - pressure = leverage × (leverage − 1) × the day's return of the underlying, times the fund's dollar volume
    relative to its 60-day average;
  - summed over 1 and 5 days.
- **Breadth stress:** the share of the 90 stocks closing at a 20-day low, and the share with volume ≥ 2× normal on a
  down day (proxies for forced or margin selling; true margin data is not public).

**The bigger picture.**
- **Treasury yields** (FRED DGS2 and DGS10, US Treasury, public domain): the level, the 5- and 20-day change, and the
  2s10s curve.
- **Bonds and fear:**
  - TLT 5- and 20-day returns;
  - VIXY (VIX-futures ETF) 5-day return, as a fear proxy (the VIX index itself is not used, for licensing reasons);
  - SPY 20-day realised volatility and SPY drawdown from its 250-day high.
- **News** (FinBERT, as above), plus market-wide daily theme counts and theme sentiment. Themes are matched with fixed
  keyword lists written here:
  - AI: "AI bubble", "AI spending", "AI capex", "AI fear", "AI stocks";
  - rates: "rate hike", "Fed", "Treasury yield", "bond yield", "inflation";
  - deleveraging: "margin call", "deleverag", "liquidation", "forced selling", "leveraged ETF";
  - export: "export control", "chip ban", "tariff".

**Stress regime.** A day is "stress" if SPY is ≥ 8% below its 250-day high or SPY's 20-day realised volatility is in
its top 20% (measured on data up to that day). Results are reported separately for stress and calm days.

**Model.** The pooled ridge is replaced by a small neural network: 2 hidden layers of 32 units, weight decay, 5 seeds
averaged, early-stopped on the last 20% of each training window. Everything else is unchanged: same targets,
retraining every 21 days on an expanding window, no-foresight rules, v3 ranges around the predicted mean, and the same
adoption and final-test rules as above. The network size is fixed here and will not be tuned on the test year.

**Hardware.** CPU. The GPU is not needed for this size.

---

## Results (added after the overnight run, 2026-10-18; nothing above was changed)
**Data.** 329,865 news items (2016 to 16 Oct 2026), all scored by FinBERT on the CPU. 45 inputs: 22 per stock and 23
market-wide. 685 of 2,563 days were classed as stress.

**Selection.** Weight decay 0.01, chosen by development IC (all three choices gave IC ≈ +0.009). Search settings:
2 seeds, retraining every 63 days. The final run used the registered 5 seeds, retraining every 21 days.

**Adoption (development, 2017-01-03 → 2025-10-10): NOT ADOPTED. v3 stays live.**
- Pooled IC +0.0066, t = 0.65. Rule (a) needs t > 2: **fail**.
- Pooled CRPS ×1e4: v4 256.268 vs v3 256.289. Rule (b): technically lower, but far too small to matter.

**Final test (2025-10-13 → 2026-10-09), reported for completeness**

| Rule | Result |
|---|---|
| Calibration, all 4 stocks | pass (it is v3's ranges) |
| Pooled CRPS skill vs random walk | −0.32%: **fail** |
| Cross-stock direction | 49.58%, p = 0.66: **fail** |

v4 − v3 test CRPS: −0.545 ×1e4, 95% block-bootstrap CI −1.20 to +0.10, which includes zero.

**Stress vs calm.**
- **Development, stress days (137 windows):** direction hit 51.46% (one-sided p = 0.013), IC t = 0.6.
- **Development, calm days:** 49.93%.
- **Test year:** too few stress windows to measure (fewer than 5).
- This is a secondary split among many, so p = 0.013 is not strong evidence on its own. It is worth following in the
  forward record.

**Which layer carries signal (extra check, not used for adoption):**

| Inputs | Development IC (t) | Test IC (t) |
|---|---|---|
| Trend / acceleration only | +0.011 (1.2) | +0.059 (2.7) |
| — of which, stress days in development | −0.006 | — |
| ETF + macro only | +0.006 (0.5) | −0.000 (0.0) |
| News only | +0.007 (1.0) | +0.026 (1.1) |
| Full model, seed set 2 | +0.011 | +0.036 |
| Full model, seed set 3 | +0.011 | +0.037 |

The seed-set results are stable. Trend-only looked good in the test year but not over 2017–2025, and in stress periods
its sign was opposite to "steepening falls continue". A one-year result that the nine development years do not
support is not adopted.

**Hold-out 12–16 Oct, closes inside the 80% range** (information only): identical for v3 and v4.

| | MU | NVDA | AMD | AAPL |
|---|---|---|---|---|
| v3 and v4 | 1/5 | 3/5 | 4/5 | 5/5 |
