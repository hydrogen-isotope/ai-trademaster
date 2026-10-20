# Dip-and-bounce intraday study (rules written 2026-10-21, before any result)

## Question
Buy a dip below the day's average price with a limit order, sell the bounce with a limit order: does it make money after
costs on NVDA, MU and QQQ? The user's target is 1% of capital a day; the study also reports what the rule earns per day
on the capital it uses.

## Data
Alpaca 5-minute bars (SIP, split-adjusted), 2016-01 → 2026-10-19, `private/data/intraday5/`. Regular session only.

## Rule (per stock, at most one trade per day)
- **VWAP** = running volume-weighted average of the bars' typical price (high + low + close) / 3 from 09:30.
- **Entry:** from the 10:00 bar to the 15:00 bar, a buy limit sits at VWAP × (1 − d), using VWAP up to the previous
  bar. It fills only if a bar trades **through** it (bar low < limit × (1 − 0.0005)), at the limit price. This
  conservative fill rule stands in for queue position and adverse selection.
- **Exit:** a sell limit at entry × (1 + u), filled if a later bar's high > the target (at the target). Otherwise the
  position is sold at the day's last close. Nothing is held overnight.
- **Grid (12 per stock):** d ∈ {0.5%, 1%, 1.5%, 2%} × u ∈ {0.5%, 1%, 2%}.

## Costs (round trip)
1 bp base (limit orders pay no spread; fees and occasional slippage at the close), also reported at 3 and 5 bps.

## Periods, selection and success (decided now)
- Development: 2016-01-04 → 2022-12-30. For each stock, the setting with the highest mean net return per **day**
  (trades × mean per trade / days) among those with development t ≥ 2 (t over trades). None → the stock is dropped.
- Hold-out: 2023-01-03 → 2026-10-19, run once, selected settings only.
- **Success per stock:** hold-out mean net per trade > 0 with t ≥ 2 at the base cost, and > 0 at 3 bps.
- Reported for context: the share of days with a trade, the hit rate, and the mean return per trading day on the
  capital committed. The user's 1%-a-day target is compared with that.
