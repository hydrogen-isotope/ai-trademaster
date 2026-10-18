# SGX: cross-stock ranker and 5-day price forecasts for Singapore stocks (rules written before any SGX result, 2026-10-18)

## Why
The user wants SGX stocks alongside NASDAQ. This applies the adopted NASDAQ methods (v6 ranker, v7 price forecasts) to
SGX unchanged, so the result shows whether they carry over to a smaller, thinner market.

## Data (Eulerpool API, the user's key)
- **Prices:** daily OHLCV, adjusted for corporate actions (checked: no gap at DBS's April 2024 1-for-10 bonus issue).
  History from 2000.
- **Symbols:** the 583 SGX tickers (`.SI`) of Singapore-domiciled companies currently in Eulerpool's screener.
- **SURVIVORSHIP CAVEAT, stated in advance:**
  - The list contains only companies listed today. Eulerpool keeps history for delisted names (e.g. SPH until
    2022-05-12), but provides no list of them, and renamed tickers (e.g. Sembcorp Marine) are not found.
  - Stocks that delisted, were taken private or failed before today are therefore missing. That makes every backtest
    number optimistic, most for low-priced small caps.
  - All SGX results are labelled **"survivor-only universe"**.
  - Foreign-domiciled SGX listings are also missing, because the screener lists by domicile.
- **Market series:** Eulerpool has neither the STI index nor the STI ETF (ES3). The market is a **synthetic STI
  proxy**: each day, the average return of the 30 stocks with the largest 20-day traded value (as of the previous day),
  weighted by that traded value.
- **No fundamentals, industries or news.** Industry codes are missing for 88% of names, and there is no point-in-time
  fundamentals source or SGX headline source. This is price/volume only, the plan's phase 1.
- Some listings are quoted in USD/CNY/MYR rather than SGD. Returns are in each listing's own currency, and the universe
  filter uses traded value in the quote currency (a small distortion, noted).

## Universe on day t (point in time)
- Price ≥ 0.20 (quote currency).
- Among the **top 150** by 20-day average traded value, using data up to day t only.

## Features, target and models
- **Features:** the 14 v6 price/volume features, computed by the same GPU/CPU feature code; betas are against the STI
  proxy. Each is ranked across the universe daily.
- **Target:** the 5-day forward log return minus the universe median (as v6).
- **Models:**
  - B1: 12-1 momentum;
  - B2: 1-week reversal;
  - **R: ridge, α = 10 (the adopted v6 model type; the only SGX candidate);**
  - walk-forward, refit monthly, 5-day embargo, training from 2003-01-02.

## Periods
- **Development:** 2016-01-04 → 2022-12-30.
- **Final test:** 2023-01-03 → latest complete, run **once**.

## Metrics
As in v6:
- daily rank IC with Newey-West t;
- top-vs-bottom decile hit rate;
- weekly long-short portfolio;
- long-only top decile vs the STI proxy.

Costs are **30 bps per side** (wider spreads than NASDAQ). Borrow is 1%/yr, and also reported at 10%/yr. **Retail
shorting of SGX stocks is limited, so the long-only result is the headline.**

## Adoption (candidate R, final test)
All three must hold:
1. IC > 0.02 with Newey-West t > 2;
2. IC above the better of B1 and B2;
3. net long-short Sharpe > 0.5.

If any fails, SGX rankings are shown as "not adopted".

**Stated in advance:**
- About 150 stocks means much less breadth than NASDAQ's ~1,000, so t-stats will be lower for the same edge.
- Published anomalies are often weaker outside the US.
- Survivor-only data inflates results.

## Price forecasts (v7 method)
- **v7 SGX:**
  - drift from the STI proxy;
  - GJR-GARCH-t on the STI proxy;
  - β and idiosyncratic volatility;
  - the ridge tilt with shrink s ∈ {0.5, 0.75, 1.0}, chosen on development by CRPS;
  - symmetric recalibration.
- **Reported:** development and final-test calibration, CRPS vs RW, and up/down hit rate.
- **Forward shadow test:** from the first forecast date, with the v7 promotion rule (≥ 60 dates; CRPS beats RW with a
  CI excluding 0; 80% coverage within 75–85%).

## Amendments
(none yet)

## Results
(not yet run)
