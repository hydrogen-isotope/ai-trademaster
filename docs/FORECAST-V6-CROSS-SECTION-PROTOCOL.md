# v6: cross-stock NASDAQ ranker (rules written before any v6 result, 2026-10-18)

## Why
v1–v5 forecast each stock on its own. For direction, every version was close to a coin flip, as the research
literature predicts for single-stock timing. Professional and academic systems instead rank **all** stocks against
each other every day, combine many individually weak signals, and judge the result as a long-short portfolio after
costs:
- Gu, Kelly & Xiu, *RFS* 2020;
- Microsoft Qlib (on CSI300, LightGBM's IC is 0.045 and a linear model's is 0.040);
- Jensen, Kelly & Pedersen, *JF* 2023.

Gains reported in the literature come mostly from microcaps and vanish after costs (Avramov, Cheng & Metzker, *Mgmt
Sci* 2023). So v6 uses a liquid universe and net-of-cost results. A realistic target is an IC of 0.02–0.05; v6 is not
expected to "call" single stocks.

## Data (NASDAQ only)
- **Prices:** daily split-adjusted SIP bars from 2016-01-04 for every NASDAQ-listed symbol Alpaca knows, active **and
  delisted** (to avoid survivorship bias). No delisting returns are available: a stock that delists within the horizon
  is scored to its last close. This is a known limitation.
- **Universe on day t:** operating companies, price ≥ $5, and the top 1,000 by 20-day average dollar volume, all using
  data up to t only.
  - Operating companies exclude names matching the ETF / fund / warrant / right / unit / note / preferred / SPAC
    filter in `universe_us.py`.
  - When EDGAR industry codes are available, SIC 6726 (investment trusts) and 6770 (blank checks) are also excluded.
- **Market series:** QQQ.
- **Fundamentals:** SEC EDGAR company facts, each value usable from the trading day after its filing date. If EDGAR
  data is not ready when the development run starts, development runs without it. Fundamentals may then be added
  **once**, by a dated amendment written before the final test, and the development numbers are re-run with them.

## Features at close t (computed on the GPU, `src/Trademaster.Gpu/Shaders/features.hlsl`, CPU-parity tested)
- **Price:** mom5, mom21, mom63, mom126, mom12_1, vol20, vol60, dist_252_high, volume_surprise, log_dollar_vol20,
  beta252, resid_mom12_1, max_ret21, idio_vol60.
- **Industry** (2-digit SIC; only with EDGAR): ind_mom21, ind_mom126, ind_breadth21, ind_leader_mom5,
  mom21_minus_ind.
- **Fundamental** (only with EDGAR): earnings_yield, book_to_market, sales_to_price, gross_profitability, roe,
  asset_growth, accruals, share_issuance, leverage, sue, days_since_filing.
- **Market regime** (same value for every stock): QQQ 21-day return, QQQ 20-day vol, 10-year yield 21-day change.
  These are given to the nonlinear models only.

Every stock-level feature is converted to its cross-sectional percentile rank inside the universe that day. Missing
ranks are set to 0.5.

## Target
- **Primary:** the 5-day forward log return from close t to close t+5, minus the universe median of the same quantity
  on day t. Models are trained on its cross-sectional rank.
- **Secondary (reported only):** the same at 21 days.

## Models (walk-forward, refit on the first trading day of each month)
- **Training rows:** all universe stock-days from 2017-01-03 (after the 252-day warm-up) whose 5-day target window
  ends **before** the refit day. That embargo means no overlapping labels.
- **Validation (for early stopping only):** the last 12 months of the training rows.

The models:
- **B1:** rank of mom12_1 (momentum).
- **B2:** minus the rank of mom5 (1-week reversal).
- **R:** ridge regression on the ranks, α = 10 (fixed).
- **G:** LightGBM regression with Qlib's published Alpha158 settings: learning_rate 0.2, max_depth 8, num_leaves 210,
  subsample 0.8789, colsample_bytree 0.8879, lambda_l1 205.7, lambda_l2 580.98, early stopping 50 rounds on
  validation, at most 1,000 rounds.
- **M:** MLP 2 × 64 ReLU trained on the GPU (HLSL), Adam lr 1e-3, weight decay 1e-4, batch 4,096, at most 20 epochs
  with early stopping on validation (patience 3), averaged over 10 seeds.
- **E (v6 candidate):** the average of the daily cross-sectional ranks of G and M.

## Periods
- **Development:** predictions dated 2021-01-04 → 2024-12-31. All choices, and the stop gate, use only this.
- **Final test:** predictions dated 2025-01-02 → the last date with a complete 5-day target. Run **once**, after
  development is frozen.

## Metrics (for every model)
- **Daily rank IC:** Spearman correlation of prediction vs the target across universe stocks. Report the mean, and its
  t-stat with a Newey-West lag of 4 (5-day targets overlap).
- **Decile spread:** mean target return of the top 10% minus the bottom 10%.
- **Weekly portfolio:** rebalanced every 5th trading day; long the top 10%, short the bottom 10%, equal weight.
  - Costs: 10 bps per side on traded weight, plus 1%/yr borrow on the short side.
  - Report net annualised return, Sharpe (×√52), turnover, max drawdown, and the share of weeks where the top decile
    beats the bottom decile.
- **Long-only top decile** vs QQQ: net return and information ratio.

## Gates
- **Stop gate (development, after R):** if the best development IC among B1, B2 and R is < 0.015, v6 stops there and
  results are reported, rather than building G and M.
- **Adoption (final test, model E):** all three must hold:
  1. IC > 0.02 with Newey-West t > 2;
  2. IC above the better of B1 and B2 over the same period;
  3. net long-short Sharpe > 0.5.

  Otherwise v6 is reported as **not adopted**.

## Checks
- **Placebo:** targets shuffled across stocks within each day give an IC of ≈ 0 (|IC| < 0.005).
- **Timing check:** features lagged one extra day must not raise the IC. A large rise would reveal look-ahead.
- **GPU/CPU parity:** `trademaster gpu-features --check` on the full panel; differences are reported.

Any change to these rules after a v6 result is seen is written below as a dated amendment, with the reason.

## Amendments

**A (2026-10-18, after the development checkpoint, before G/M and before any final-test result).**
The development diagnostics showed that most long-short return comes from shorting high-volatility small and mid caps.
These are often hard to borrow, at far more than 1%/yr. The final test therefore **also reports**, for every model:
- the long-short result with **10%/yr borrow** on the short side;
- the **long-only** top decile vs QQQ (already in the metrics), shown as the conservative headline;
- the long-short result split into its long and short legs.

The adoption rule is unchanged. A model that passes only because of the 1%/yr borrow assumption is described as such.

## Development checkpoint (2026-10-18; price features only, EDGAR not yet available)

**Setup:** panel of 2,713 days × 6,501 NASDAQ symbols, 2,580,894 universe stock-days; development
2021-01-04 → 2024-12-31.

| model | IC | NW t | IC > 0 | L/S net /yr | L/S Sharpe | max DD | top > bottom | turnover | long-only vs QQQ /yr |
|---|---|---|---|---|---|---|---|---|---|
| B1 momentum 12-1 | +0.029 | 3.0 | 57% | +35% | 1.09 | 40% | 59% | 17% | −16% |
| B2 reversal 1w | +0.007 | 1.1 | 51% | +2% | 0.06 | 43% | 54% | 85% | −39% |
| **R ridge** | **+0.068** | **8.1** | 67% | +57% | 1.87 | 36% | 62% | 42% | −10% |
| placebo (shuffled) | +0.002 | 1.5 | — | — | — | — | — | — | — |
| R, features 1 day older | +0.066 | 7.6 | 66% | +52% | 1.68 | 39% | 61% | 38% | −10% |

**Stop gate:** passed. Placebo ≈ 0; one extra day of lag barely changes the IC, so there is no sign of look-ahead.

**Where the edge comes from (diagnostics, development only):**
- **Strongest single features:**
  - low volatility / lottery: vol60 −0.080, idio_vol60 −0.077, max_ret21 −0.071;
  - distance from the 52-week high: +0.061;
  - momentum: resid_mom12_1 +0.030, mom12_1 +0.029.

  These are the documented low-risk / MAX / 52-week-high anomalies.
- **Every year is positive:** IC 0.075 (2021), 0.061 (2022), 0.051 (2023), 0.084 (2024).
- **Most of the long-short return is the short leg.** The bottom-decile stocks (speculative, high-volatility small and
  mid caps) fell every year. Annualised arithmetic mean returns:

  | year | short leg | long leg | QQQ |
  |---|---|---|---|
  | 2021 | −16% | +21% | +28% |
  | 2022 | −90% | −21% | −39% |
  | 2023 | −37% | +19% | +48% |
  | 2024 | −83% | +21% | +28% |

  The long leg trails QQQ in up years.
- **Caveat:** such names are often hard to borrow, at far more than the protocol's 1%/yr. The long-short Sharpe
  therefore overstates what is achievable. The long-only result (−10%/yr vs QQQ) is the conservative view.

## Results
(final test not yet run)
