# Engineered strategy: one design search on a shared back-test engine (rules written 2026-10-21, before any result)

## Why
Earlier slices each tested one idea with its own code, so every test was another chance for a fluke. This study builds
one portfolio from the inputs that held up (the ranking's loser side, risk forecasts, the overnight drift) plus the
user's attention idea, searches a fixed grid of designs on development data with a correction for the number of tries,
and opens the final period once.

## 1. Data (point in time: everything known at close t)
v6 panel and universe (survivorship-free, about 900 NASDAQ stocks a day), daily open and close, QQQ. A stock's missing
return after it leaves the data (e.g. delisting) counts as 0 (stated limitation).

## 2. Signals at close t
- **Loser score:** the LightGBM ranking's walk-forward prediction (monthly refits; labels end before each refit),
  recomputed from 2019 on the current panel. Cross-sectional percentile r_G.
- **Attention:** percentile of `volume_surprise` (today's volume vs its normal level). The historical news archive is
  no longer on disk, so news counts are not used.
- **Risk:** `vol20` (20-day realised volatility); earnings: a report can be priced in on day t+1
  (earnings.reaction_events, as in v7 amendment B).
- **Overnight:** used through the holding mode below.

## 3. Portfolio (weights set at close t)
- A share α in QQQ; the rest (1 − α) in a stock sleeve.
- Sleeve: the universe minus the stocks with r_G in the bottom q; weight ∝ vol20^(−γ) × (1 + λ·(2·r_att − 1)) ×
  (0.5 if earnings can be priced in at t+1 and e = halve), normalised to sum to 1 − α.

## 4. Execution and costs
- **Day mode:** trade at the close (closing auction); hold close → close; a stock's position is only changed when its
  target differs from its drifted weight by more than 20% of the target (entries and exits always trade).
- **Overnight mode:** buy the sleeve at the close, sell at the next open; flat during the day (QQQ share held
  throughout).
- Costs per one-way traded dollar: stocks 5 bps, QQQ 1 bp.

## 5. Grid and selection (development 2019-01-02 → 2024-12-31)
α ∈ {0, 0.5} × q ∈ {0, 0.1, 0.2, 0.3} × γ ∈ {0, 0.5, 1} × λ ∈ {0, 0.5, 1} × e ∈ {keep, halve} × mode ∈ {day,
overnight}: **288 designs, every one counted.**
- Daily excess return = portfolio net return − QQQ close-to-close return.
- Selected: the highest development information ratio (annualised mean / standard deviation of daily excess).
- Trial correction: the Deflated Sharpe Ratio (Bailey & López de Prado) of the selected design, using the 288 trials'
  information-ratio variance and the selected series' skew and kurtosis. **The final period is opened only if
  DSR ≥ 0.95.** Otherwise the study stops and reports.

## 6. Final check (2025-01-02 → the latest complete day), once
**Pass:** mean daily excess > 0 with t ≥ 2 (Newey-West, 5 lags) and the worst drawdown of cumulative excess better than
−25%. Stated limitation: 2025–26 was already used by the v6 final test and by the adaptive back-test today, so it is
not untouched; that is why the trial correction is applied before it, and the bar is t ≥ 2.
Reported for every period: annual excess, information ratio, tracking error, worst excess drawdown, turnover and cost
drag. Before the final check, an independent reviewer audits the code for look-ahead.

## Live
Only a passing design becomes a paper slice, compared with holding QQQ.

## Amendment A (2026-10-21, after the development search, before the final period was opened)
**What the search showed (development only):** all 288 designs lag QQQ (best: −5.6%/yr, IR −0.78; DSR 0.000, final
stays closed). The cause is the sleeve's equal weighting: over 2019–2024 QQQ returned +22.1%/yr and the equal-weighted
universe +2.2%/yr, a gap no stock signal in the grid can close.
**Change:** a new grid dimension, the sleeve's base weight: `equal` (the 288 designs already run) or `size`, a
weight ∝ the stock's 20-day dollar volume (`exp(log_dollar_vol20)`, a close proxy for market value; capped at 10% of
the sleeve, the excess spread pro rata), multiplied by the same risk, attention and earnings factors. 288 more designs.
**Selection and gate unchanged, over all 576 designs:** the highest development IR; the Deflated Sharpe Ratio counts
576 trials; the final period opens only if DSR ≥ 0.95.

## Results (added 2026-10-21 after the complete search; the final period was not opened)
**576 designs on development 2019–2024.** The 288 equal-weight designs all lag QQQ. With size weighting (amendment A)
the best designs beat QQQ by +1.0% to +2.3%/yr, but none reliably:

| selected design | excess/yr | IR | Newey-West t | worst excess drawdown | tracking error | turnover/yr | cost/yr |
|---|---|---|---|---|---|---|---|
| 50% QQQ + size-weighted sleeve minus the bottom 30% by the ranking, risk-sized (γ 0.5), day mode | +1.0% | +0.32 | +0.79 | −7.7% | 3.1% | 8× | 0.4% |

**Deflated Sharpe Ratio 0.000 (gate 0.95): the final period stays closed, as written.** Even without the trial
correction, t = 0.8 is far from the t ≥ 2 the final check requires. (The DSR's benchmark is inflated by the 144
overnight designs, whose costs make them very negative; the conclusion does not depend on it.)

What the grid shows, one choice at a time (size-weighted, day mode):
- **Loser filter helps, steadily:** dropping the bottom 0 / 10 / 20 / 30% by the ranking gives +0.9 / +1.9 / +2.2 /
  +2.3%/yr (no QQQ core) and +0.4 / +0.7 / +1.0 / +1.3%/yr (50% QQQ). Consistent with the v6 final test: the ranking's
  information is in the losers. Small, and not significant on its own over six years.
- **Attention hurts:** tilting toward unusual volume lowers the excess (λ 0 / 0.5 / 1: +1.9 / +0.5 / −1.9%/yr), in line
  with the evidence that attention-grabbing stocks underperform after the attention.
- **Halving positions before earnings** slightly hurts (+1.9 → +0.8%/yr).
- **Overnight-only holding is ruined by costs:** −30.6%/yr (buying and selling the whole sleeve every day).

**Reading.** No design earns a reliable edge over QQQ. The one consistent, sensible effect is the loser filter, worth
roughly 1–2% a year before it can be told apart from luck. Nothing is promoted; a paper slice of the selected design
would be an experiment, not a tested strategy.
