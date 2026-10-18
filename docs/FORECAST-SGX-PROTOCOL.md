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

**A (2026-10-18, data cleaning before any model result).** Building the panel and the STI proxy exposed three data
problems. The fixes:
1. **One-day spikes:** a close more than 50% away from both its neighbours, in opposite directions, is treated as a
   bad tick and removed (4,309 stock-days).
2. **Consolidation-distorted tickers dropped (12):** these are tickers whose adjusted history exceeds 100 in quote
   currency (5F7, 5JS, AUE, AYV, BAI, BJW, CJN, E6R, PRH, QZG, V3M, XCF). Large share consolidations scale old prices
   up, but not volumes, so their past traded value is wildly inflated. That corrupted the proxy and the liquidity-based
   universe. The panel is now 530 tickers.
3. **STI proxy definition:**
   - membership: the top 30 stocks priced ≥ 1 by **one-year** average traded value (as of the previous day);
   - weights: traded value, capped at 15% each;
   - moves beyond 30% in a day are left out;
   - the index averages **simple** returns (averaging log returns had added a downward drag of about half the
     variance per day).

   It tracks the STI's turning points (2003 low, 2007 peak, 2009 low, 2020 low, recovery), with a level of 0.98 by
   2026-10-16 vs about 1.7 for the real STI. The lower long-run return reflects its composition: no
   foreign-domiciled heavyweights like Jardines, and capped bank weights.

   It is used for betas, residual momentum, market volatility and the small drift term. **Labelled "STI proxy",
   not STI.**

**B (2026-10-18, before any result of this strategy is produced). Forward-only test of SGX 12-1 momentum (long-only).**

The ranker's final test hinted that plain 12-1 momentum, long-only, beat the STI proxy (+8.5%/yr, IR 0.47). That
number was not the pre-registered candidate, so it is **not** evidence. Momentum is tested **forward only**, on months
that have not happened yet:
- **Formation:** at the close of the last SGX trading day of each month, starting **2026-10-30** (the first month-end
  after this amendment).
- **Universe:** stocks priced ≥ 0.20 whose 20-day average traded value is in the top 150 at formation, using data up
  to the formation close only. Same panel and cleaning as amendment A.
- **Signal:** 12-1 momentum, the log price change from 252 to 21 trading days before formation. It needs both closes
  and at least 200 valid days in between.
- **Portfolio:** the top 10% of the universe by the signal, at least 10 stocks, equal weight, held until the next
  formation close. A stock that stops trading is valued at its last close.
- **Costs:** 30 bps per side on the weight that changes at each rebalance (the first month counts as fully bought).
- **Benchmark:** the STI proxy over the same holding period.
- **Logging:** picks are written to `private/reports/sgx/momentum_shadow.jsonl` with the formation date and the time
  they were written. The rule is deterministic and committed here, so a pick list depends only on data up to its
  formation close.
- **Measured monthly:** net return, benchmark return, excess = net − benchmark, and turnover.
- **Promotion rule, after ≥ 12 matured months:** mean monthly excess > 0 with t > 2 (plain t over months), **and** a
  cumulative excess > 0. Until 36 months:
  - if, after 12 months, the mean excess is ≤ 0, the strategy is **dropped**;
  - otherwise it stays "forward test" until the rule passes, or 36 months end without passing (then **dropped**).
- **Stated in advance:**
  - 12 months is too few to prove a modest edge. A t > 2 needs a large edge, so the likely outcome is still
    "inconclusive" after 12 months.
  - The data stays survivor-only, but forward months suffer no survivorship bias: stocks are picked from what is
    listed at formation.
- **Display:** the Rankings page (SGX view) shows the current provisional picks (as if the month ended at the latest
  close) and the forward record, labelled **"forward test, unproven"**. Months formed before 2026-10-30 are never
  counted.

## Results (survivor-only universe; STI proxy)

### Ranker, development 2016-01-04 → 2022-12-30 (704,906 universe stock-days)

| model | IC (t) | top > bottom | L/S net /yr | L/S Sharpe | turnover | long-only vs STI proxy /yr |
|---|---|---|---|---|---|---|
| B1 momentum 12-1 | +0.028 (4.0) | 59% | +5.9% | 0.23 | 21% | −8.1% |
| B2 reversal 1w | +0.071 (13.1) | 61% | −20.6% | −0.84 | 86% | −16.5% |
| R ridge | +0.097 (17.1) | 66% | −0.7% | −0.04 | 76% | −14.9% |
| placebo | +0.001 | | | | | |
| R, features 1 day older | +0.066 (9.9) | 61% | −10.1% | −0.55 | 57% | −15.7% |

### Ranker, final test 2023-01-03 → 2026-10-09 (run once)

| model | IC (t) | top > bottom | L/S Sharpe | at 10% borrow | long-only vs STI proxy /yr (IR) |
|---|---|---|---|---|---|
| B1 momentum 12-1 | +0.027 (3.1) | 53% | −0.02 | −0.46 | **+8.5% (0.47)** |
| B2 reversal 1w | +0.046 (6.0) | 50% | −3.36 | −3.80 | −21.8% (−1.43) |
| **R ridge (candidate)** | **+0.057 (6.7)** | 51% | **−3.09** | −3.55 | −21.3% (−2.33) |

**Adoption:** the IC passes and beats the baselines, but net Sharpe is −3.09 < 0.5. **The SGX ranker is NOT adopted.**

Reading:
- The SGX IC is real but comes mostly from short-term reversal, i.e. bid-ask bounce in thinly traded stocks. The edge
  halves with one day of delay, and it turns over 76% of the portfolio each week, which 30 bps costs wipe out.
- Plain 12-1 momentum was the only profitable long-only signal in the test (+8.5%/yr vs the proxy). It was not the
  candidate and is **not** adopted after the fact. It would need its own pre-registered, forward test.

### Price forecasts (v7 method)

| period | model | CRPS | vs RW (95% CI) | 80% cov | 95% cov | up/down hit |
|---|---|---|---|---|---|---|
| dev 2016–22 | RW | 0.02144 | — | 80.0% | 94.9% | — |
| dev 2016–22 | **v7 s=0.5 (chosen)** | 0.02141 | −2.8e-5 [−4.2e-5, −1.5e-5] | 79.9% | 94.9% | 52.3% |
| dev 2016–22 | v7 s=1.0 | 0.02149 | +5.0e-5 (worse) | 79.9% | 94.8% | 52.1% |
| test 2023–26 | RW | 0.02101 | — | 79.9% | 94.9% | — |
| test 2023–26 | v7 s=0.5 | 0.02102 | −1.4e-6 [−1.9e-5, +1.6e-5] | 79.9% | 94.9% | 52.3% |

The ranges are well calibrated in both periods. Up/down is 52.3% in both. The accuracy gain over a random walk is
significant in development but not in the test. The SGX v7 forecasts go into the forward shadow test (separate log)
and are shown labelled "shadow".
