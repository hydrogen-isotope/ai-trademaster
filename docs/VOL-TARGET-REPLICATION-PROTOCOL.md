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

## Results (added 2026-10-21 after the single run; nothing above was changed)
EEM's history at Twelve Data starts in 2008, so it enters in 2013 (after the 1260-day warm-up); duplicate rows in the
download are removed on load.

| period | assets | average hedged return | Newey-West t | assets with positive alpha |
|---|---|---|---|---|
| **2019-01-02 → 2026-10-19 (primary)** | 15 | **−0.29%/yr** | **−0.19** | 7 / 15 |
| 2004 → 2018 (history) | 15 | +2.38%/yr | +2.36 | 14 / 15 |

**Fails.** The rule worked broadly before 2019 (14 of 15 ETFs, alpha +1.3% to +4.1%/yr) and not at all since: alpha
near zero everywhere, and a lower CAGR than holding on all 15 ETFs (e.g. SPY +13.6% vs +17.2%, XLK +20.7% vs +28.1%).
What it still does reliably is **cut drawdowns** (SPY −23% vs −34%, sector ETFs typically 10–15 points smaller), but
since 2019 that protection cost as much return as it saved: the 2020 and 2022 falls recovered quickly, and the rule
was still holding less when they did. Consistent with the later literature (Cederburg et al. 2020: volatility-managed
portfolios do not reliably beat their unmanaged versions out of sample).

**Consequence:** no paper slice. QQQ's own 2019+ period stays unopened under the first study.
