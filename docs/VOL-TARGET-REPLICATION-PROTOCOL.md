# Volatility targeting: replication on other index ETFs (rules written 2026-10-21, before their data was downloaded)

## Why
The QQQ timing search (`QQQ-TIMING-PROTOCOL.md`) found volatility targeting consistently positive on QQQ 2000–2018,
but the selected design missed the deflated-Sharpe gate (0.927 vs 0.95), so QQQ's 2019–2026 period stays closed.
Rather than relax that gate, this study tests **one fixed rule, no search**, on other markets never used for timing
here. If the effect is real it should show up there too; one rule means no trial correction is needed.

## Rule (fixed from the QQQ selection, scaled to each asset)
- Exposure decided from data through close t, traded at close t + 1, as in the QQQ study.
- Volatility estimate: annualised standard deviation of the last 60 daily log returns.
- Target: 0.9 × the asset's own realised volatility over the previous 1260 trading days (the QQQ selection, 25%, was
  0.9 × QQQ's 27.8% development volatility). Exposure = min(1, target / estimate): no leverage, no trend filter.
- Same costs (1 bp per traded dollar, 0.10 band), cash earns the T-bill rate.

## Assets (decided now)
SPY, DIA, IWM, MDY, EFA, EEM and the nine original sector SPDRs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY): 15
equity ETFs, adjusted daily prices from Twelve Data. QQQ is excluded. An asset enters once it has 1260 + 60 days of
history.

## Test
For each asset and day, the hedged return h = (design − T-bill) − b × (asset − T-bill), with b the asset's beta
estimated over the same period. Each day, the average h over the assets present.
- **Primary (pass/fail): 2019-01-02 → the latest complete day.** Pass if the mean daily average h > 0 with Newey-West
  t ≥ 2 (5 lags).
- Also reported: 2000–2018 (assets as available), per-asset alpha, t, CAGR, Sharpe and worst drawdown vs holding the
  asset, and how many assets have positive alpha.

## Consequence
If it passes, a paper slice running this rule on QQQ (and only then) is justified as an experiment; QQQ's own 2019+
period remains unopened under the first study. If it fails, nothing changes.
