# 5-day forecast v5: options market + earnings events (rules written before any v5 result, 2026-10-18)

## Why
v1–v4 found no reliable 5-day direction signal in prices, ETFs, rates or news for large US stocks. A direct test of
"fear → falling stocks keep falling" over 2016–2026 found the opposite: the next-5-day fall rate was 39–43% under fear
conditions, vs 45% baseline. v5 adds information that often *leads* prices: the options market. It also adds earnings
events, because post-earnings drift is one of the best-documented return patterns.

## New data
**Options** (Alpaca; daily bars of listed contracts, available from Feb 2024). For each stock, all **monthly** expiries
(third Friday), calls and puts, strikes within ±30% of the stock price at the time.

**Earnings events**, from the stored Alpaca/Benzinga news (2016+). An item is an earnings item if its headline matches
any of these (case-insensitive):
`Q[1-4] (earnings|results)`, `reports Q[1-4]`, `earnings beat`, `earnings miss`, `beats estimates`, `misses estimates`,
`EPS of`, `quarterly results`.

## New features per stock at close *t* (no foresight)
**Options**, from the monthly expiry closest to 30 calendar days out (at least 7). Implied volatility is solved from the
option's daily close with Black-Scholes, using the 2-year Treasury yield and no dividend. Only contracts with volume > 0
that day are used.
- `iv_atm`: the mean IV of the call and put with strike nearest the stock close.
- `iv_rv`: `iv_atm` minus the 20-day realised volatility (annualised), i.e. fear priced in vs actual.
- `iv_chg5`: the 5-day change in `iv_atm`.
- `skew`: IV of the put nearest 90% of spot, minus IV of the call nearest 110% of spot (crash protection demand).
- `pc_vol`: log((put volume + 1) / (call volume + 1)) over the chosen expiry.
- `pc_vol_z`: `pc_vol` minus its own 60-day mean.

**Earnings**
- `earn_day`: 1 if an earnings item is assigned to close *t*.
- `earn_age`: trading days since the last earnings close (capped at 60).
- `earn_react`: the stock's return relative to its sector ETF on the last earnings close, in vol units.
- `earn_sent`: the mean FinBERT sentiment of that earnings close's items.
- `earn_drift = earn_react × max(0, 1 − earn_age/20)`: the post-earnings drift window.

All features are ranked across stocks each day. Missing values (no liquid options that day, no earnings yet) take the
neutral value.

## Model and periods
v5 = the v4 network (2 × 32, the v4 weight-decay choice, 0.01, 5 seeds, early stopping) on v4 features + the 11 new
features, with v3 ranges around its mean.
- **Development:** origins from 2024-04-01 (options history needs a warm-up) to 2025-10-10.
- **Final test:** the same last 250 origins as v2–v4 (2025-10-13 → 2026-10-09).

The development window is short (about 385 days). That is a limitation, and it is stated here in advance.

## Adoption and test rules (unchanged)
- **v5 is adopted only if**, over development, (a) its pooled cross-stock IC has t > 2 *and* (b) its pooled CRPS on
  MU, NVDA, AMD and AAPL is lower than v3sym's over the same period.
- **Final-test rules:** calibration for each stock at every horizon; pooled CRPS skill > 0 vs the random walk;
  cross-stock direction > 50% at p < 0.05.

**Also reported, not used for adoption:**
- IC of the options features alone and of the earnings features alone;
- earnings-only IC over the full 2017–2026 history (available because news goes back to 2016);
- the 12–16 Oct hold-out.

## Also added: "your rule" live tracker
A dashboard signal predicts "falls further" for any stock that is down 5%+ over 5 days while the VIXY 5-day change is
in its top 20% (thresholds from past data only). It is scored live against the model from 19 Oct 2026. It is never
used in the model.

## Amendment B (2026-10-18, still before any v5 result)
A check of the earnings matcher (6,732 of 329,865 items matched) showed preview articles being matched, e.g. "Ahead Of
Q1 Earnings" and "Likely To Report Higher Q4 Earnings". Items whose headline also contains any of `ahead of`,
`likely to`, `preview`, `expect`, `what to watch`, `could`, `will report` or `set to report` are **not** earnings
items. Nothing else changes.

## Amendment C (2026-10-18, still before any v5 result)
A check of MU's detected earnings days showed other companies' earnings items tagged with many tickers (e.g. 13 and
23 Jul 2026). An earnings item now counts only for its symbols if it is tagged with **at most 3** tickers. Nothing else
changes.

---

## Results (added after the complete run, 2026-10-18; nothing above was changed)
The first run was cut off by the session ending and produced no result. It was rerun from scratch, completed in
31 min, and its result file was checked as complete.

**Inputs:** 56 (v4's 45 + 6 options + 5 earnings). Options cover 94% of stock-days in the test year; 3,136 earnings
days were found.

**Adoption (development, 2024-04-01 → 2025-10-10): NOT ADOPTED. v3 stays live.**
- Pooled IC −0.018 (t = −0.66). Rule (a) needs t > 2: fail.
- Pooled CRPS ×1e4: v5 279.44 vs v3 279.82.

**Final test (2025-10-13 → 2026-10-09)**
- Calibration: pass for all 4 stocks.
- CRPS skill vs random walk: −0.37%, fail.
- Direction: 50.50%, p = 0.33, fail.
- **v5 − v3 test CRPS: −1.31 ×1e4, 95% CI −2.61 to −0.11.** This is the first variant whose test-year accuracy beats
  v3 with a CI that excludes zero. It was not predicted by development, and several variants have now been tested
  on this same year, so it is treated as a lead to confirm going forward, not as proof.

**Extra checks (not used for adoption)**

| Inputs | Development IC (t) | Test IC (t) |
|---|---|---|
| Options only | +0.011 (0.8) | −0.008 (−0.2), hit 48.8% |
| Earnings only | −0.002 (−0.1) | +0.060 (1.9), hit 52.2% |
| Earnings only, 2017–2026 | +0.012 (1.5), hit 50.6% | — |

**Hold-out 12–16 Oct, closes inside the 80% range:** v3 and v5 identical (MU 1/5, NVDA 4/5, AMD 4/5, AAPL 5/5).

**Next.** Run v5 as a *shadow* model from 19 Oct 2026: forecasts logged daily next to v3 and scored as outcomes
arrive, without being shown as the main forecast. Promotion rule, fixed now: after ≥ 60 trading days, v5 replaces v3
only if its forward pooled CRPS is lower and the block-bootstrap 95% CI of the difference excludes zero.
