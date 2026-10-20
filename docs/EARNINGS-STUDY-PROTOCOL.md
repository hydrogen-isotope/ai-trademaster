# Earnings reaction study (rules written 2026-10-20, before any result)

## Question
Is there a tradable pattern around earnings reports in NASDAQ stocks, after realistic trading costs? This decides
whether an automated earnings strategy is worth building (stage 2) and paper-testing (stage 3).

## Data
- **Reports:** Eulerpool `calendar/earnings-by-symbol` (`private/data/earnings/by_symbol/`): report date, time (when
  given), actual and estimated EPS. 67,083 reports in 2016–2026 have both.
- **Prices:** `private/data/us/panel.npz` (daily open, high, low, close, volume, split-adjusted; Alpaca SIP).
  QQQ is the market.
- **Eligible events:** the stock's close ≥ $5 and 20-day median dollar volume ≥ $5M on the last close before the
  reaction day; the reaction day and every day in the holding window have prices.

## Definitions
- **Reaction day R:** the report date for before-open or during-market reports, the next trading day for after-close
  reports. Unknown time (most reports): whichever of the report date and the next trading day has the larger absolute
  opening gap. *Stated limitation:* that choice looks at the gap itself, so it is checked by a robustness run on
  known-time reports only.
- **Gap:** log(open_R / close_{R−1}). **σ:** standard deviation of the stock's daily log returns over the 60 trading
  days ending at close_{R−1}. **Gap size** = |gap| / σ.
- **Surprise:** (actual − estimate) / max(|estimate|, 0.05). Terciles are formed **within each calendar quarter of
  the development period only** and then applied as fixed cut-offs to the hold-out.
- **Market-adjusted return:** the stock's log return minus QQQ's over the same window. Both raw and adjusted are
  reported; rules are judged on **adjusted** returns.
- **Costs (one-way):** 10 bps at the open, 5 bps at the close, 3 bps for an exit at a later close. Every rule is also
  reported at 2× costs.

## Candidate rules (the complete menu; nothing else is selected)
Each trade is one event; the sign is +1 (long) or −1 (short). Also reported: the long-only half of each rule.
- **H1, day-0 gap (open_R → close_R):** sign = d·sign(gap), d ∈ {+1 continuation, −1 reversal}, only when gap size ≥ g,
  g ∈ {1, 2, 3}. 6 rules.
- **H2, surprise on day 0 (open_R → close_R):** sign = sign(surprise), only for the top and bottom surprise terciles,
  either (a) always or (b) only when sign(gap) = sign(surprise). 2 rules.
- **H3, drift after the reaction (close_R → close_{R+k}):** sign = sign(surprise), top and bottom terciles, k ∈ {5, 20},
  either (a) always or (b) only when the day-0 return close_{R−1} → close_R has the surprise's sign. 4 rules.

## Periods
- **Development:** reaction days 2016-01-04 → 2022-12-30. All selection happens here.
- **Hold-out:** reaction days 2023-01-03 → 2026-10-16. Run **once**, after selection is fixed, on the pre-selected rules
  only.

## Statistics
Per rule: number of trades, mean net adjusted return per trade, hit rate, and a t-statistic **clustered by reaction
day** (average the trades of each day first, then t over days). Annual figures assume trades are equally weighted.

## Selection (development) and success (hold-out), decided now
- **Selection:** for each of H1, H2 and H3, the rule with the highest mean net adjusted return per trade, provided its
  development t ≥ 2. A hypothesis with no rule at t ≥ 2 is dropped.
- **Success, per selected rule:** on the hold-out, (1) the mean net adjusted return is > 0 with t ≥ 2, and (2) it is
  still > 0 at 2× costs. Reported separately for the long-only half, since the paper account is long-only.
- 12 rules are compared in development, so a hold-out pass is needed before anything is built; a development-only
  result is not evidence.

## Outputs
`python/trademaster_ml/earnings_study.py` (event table and evaluation), `private/reports/earnings_study/` (events,
development table, selection, hold-out result). Results are added to the end of this file without changing anything
above.

## Results (added 2026-10-20 after the complete run; nothing above was changed)
Run as a workflow of 7 Claude Sonnet agents: one built the events and evaluation code, three tested H1–H3 on
development, and three independent reviewers tried to refute each result and recomputed it with their own scripts.

**Events:** 18,550 in development and 13,442 in the hold-out. Only 0.17% have a known report time, so the
known-time robustness run is nearly empty (5–31 trades per rule) and cannot test the unknown-time rule.

**Development (2016–22), net market-adjusted return per trade, t clustered by reaction day:**

| rule | trades | net (bps) | t | at 2× costs | long-only net |
|---|---|---|---|---|---|
| H1 continuation, gap ≥ 1σ / 2σ / 3σ | 10,861 / 6,283 / 3,808 | −30.0 / −26.7 / −25.3 | −2.04 / −2.16 / −0.27 | −45 / −42 / −40 | −51 / −56 / −49 |
| H1 reversal, gap ≥ 1σ / 2σ / 3σ | 10,861 / 6,283 / 3,808 | 0.0 / −3.3 / −4.7 | −0.92 / −0.26 / −1.84 | −15 / −18 / −20 | −23 / −34 / −27 |
| H2 surprise on day 0, (a) / (b) | 11,648 / 7,955 | −14.6 / −21.8 | −1.47 / −1.26 | −30 / −37 | −29 / −27 |
| H3 drift 5 days, (a) / (b) | 11,643 / 7,212 | +1.5 / +0.6 | −2.02 / −1.05 | −6.5 / −7.4 | −36 / −28 |
| H3 drift 20 days, (a) / (b) | 11,625 / 7,205 | +7.2 / +7.5 | −1.58 / −0.86 | −0.8 / −0.5 | −141 / −107 |

**Selection: no rule has development t ≥ 2 in the right direction, so H1, H2 and H3 are all dropped. The hold-out was
not opened.** All three reviewers reproduced the numbers and could not overturn the conclusion:
- H1: no rule turns positive with t ≥ 2 under other reaction-day definitions either (report day; next day). The
  max-gap rule peeks at the next open, so "continuation loses" is not a reliable finding on its own.
- H2: gross is about +0.4 bps per trade; the 15 bps round trip is the whole loss.
- H3: 20-day drift is +7 bps per trade before the clustered t is considered, and below zero at 2× costs.

**Reading.** On daily data, with costs at the open of 10 bps one-way, none of the standard earnings patterns is
tradable in NASDAQ stocks in 2016–22. Earnings moves are large but their direction after the open is not predictable
from the gap or the EPS surprise. An automated earnings strategy is not built on this evidence. A finer test would need
report times (to trade the true first reaction) and intraday quotes; neither is in the current data.
