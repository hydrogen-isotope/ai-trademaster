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

## Results (added 2026-10-21 after the complete run; nothing above was changed)
Development 2016–22, net per trade at 1 bp round trip (t over trades), dip d below VWAP / bounce target u:

| stock | d = 0.5% | d = 1% | d = 1.5% | d = 2% | selected |
|---|---|---|---|---|---|
| NVDA | −20 to −24 bps (t −4 to −6), 1,274 trades | −15 to −20 (t −3) | −12 to −17 (t −2) | −3 to −22 (t −0.4 to −1.7), 276 trades | none |
| MU | −21 to −25 (t −5 to −7), 1,333 trades | −16 to −21 (t −3 to −4) | −14 to −18 (t −2) | −0.4 to −11 (t −0.1 to −1.0), 274 trades | none |
| QQQ | −8 to −14 (t −2 to −4), 668 trades | −6 to −12 | +7 to +9 (t < 1), 91 trades | **+31 to +63 (t 3–6), 35 trades** | d 2%, u 1% |

**Hold-out 2023–26, QQQ d 2% / u 1%: 4 trades, −71 bps per trade (t −0.7). Fails.**

**No setting works.** Buying intraday dips in NVDA and MU **loses money, significantly** (t down to −7): a stock that has
fallen well below its day's average tends to keep falling, and a limit order is filled exactly then. The only positive
cell was QQQ after a 2% intraday drop, which happened 35 times in seven years (mostly the 2020 crash rebound) and 4
times since, losing. Against the user's target of 1% a day: the best NVDA and MU settings earned about 0 per day on the
capital used, before any chance of 1%.
