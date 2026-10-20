# QQQ timing: how much to hold, not which stocks (rules written 2026-10-21, before any result)

## Why
Every stock-picking design so far lags QQQ or beats it by an amount that cannot be told from luck. The published
evidence that has survived out of sample points elsewhere: **volatility managing** (hold less when recent volatility is
high, more when it is low; Moreira & Muir 2017) and **trend filtering** (step out when the index is below a long moving
average; Faber 2007, Hurst-Ooi-Pedersen 2017). Both act on one instrument, so costs are small. This study asks whether
either, alone or together, earns a reliable risk-adjusted edge over holding QQQ, with the option of modest leverage
(the user accepts aggressive risk on the paper account).

## 1. Data
QQQ daily open/high/low/close, adjusted for splits and dividends (Twelve Data, `adjust=all`, from 1999-03-10), saved to
`private/data/timing/`. Cash rate: 3-month T-bill (FRED DTB3, annual %, forward-filled over missing days). A day whose
volume is below 10% of its trailing 20-day median is treated as incomplete and dropped (guards against a partial last
bar).

## 2. Exposure rule (decided from data through close t, traded at close t + 1: one full day of delay)
- **Volatility estimate:** σ̂_t = annualised standard deviation of the last w daily log returns, w ∈ {20, 60}.
- **Volatility target** σ* ∈ {15%, 20%, 25%, 30%}: raw exposure = σ*/σ̂_t, capped at L ∈ {1, 1.5, 2}.
- **Trend filter** ∈ {none, 100-day, 200-day}: when close t is below its simple moving average of that length, exposure
  is 0 (cash).
- Also: trend filter only, at fixed exposure L ∈ {1, 1.5, 2} while the trend is up.
- **Grid:** 4 targets × 2 windows × 3 caps × 3 trend settings = 72, plus 2 trend lengths × 3 levels = 6 → **78 designs,
  all counted.** (Plain buy-and-hold is the benchmark, not a trial.)

## 3. Execution and costs
- Position changes only when the new target exposure differs from the drifted exposure by more than 0.10 (moves to or
  from 0 always trade).
- Cost 1 bp per traded dollar.
- Uninvested cash earns the T-bill rate; borrowed money (exposure > 1) costs the T-bill rate + 1.5% a year. (Stated
  limitation: the Alpaca paper account pays nothing on cash and its margin rate may differ; the live slice will be
  judged as run.)

## 4. Measure
Daily excess returns over the T-bill: r_d (design) and r_q (QQQ). Regression r_d = a + b·r_q + e.
**Edge = annualised alpha a**; its strength is the appraisal ratio a/σ(e) and the Newey-West t (5 lags).
Reported for every period: CAGR, volatility, Sharpe, worst drawdown, average exposure, turnover, cost drag, alpha,
beta, and the same for QQQ.

## 5. Selection (development 2000-01-03 → 2018-12-31)
The design with the highest development alpha t-statistic. **Gate:** the Deflated Sharpe Ratio of its daily appraisal
ratio across the 78 trials (their appraisal-ratio variance, the series' skew and kurtosis) must be ≥ 0.95, else the
study stops and reports.

## 6. Final check (2019-01-02 → the latest complete day), once
**Pass:** alpha > 0 with Newey-West t ≥ 2. Also reported, not part of the pass: whether its CAGR beat QQQ and whether
its worst drawdown was smaller. Stated limitation: 2019–2026 QQQ prices were seen as a benchmark in earlier studies, but
no timing rule was ever tested on them.

## Live
Only a passing design becomes a paper slice (exposure set from the daily close, traded near the next close), compared
with holding QQQ. If none passes, nothing changes.
