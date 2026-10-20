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
