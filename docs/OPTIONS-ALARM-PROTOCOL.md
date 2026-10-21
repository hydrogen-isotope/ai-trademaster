# NVDA & MU options alarm: back-test and a learned exit model (rules written 2026-10-21, before the data was downloaded)

## Question
The live slice (one slice, whole paper account) buys NVDA and/or MU at 9:35 New York when the first 5 minutes closed
up, and sells on an **options alarm**: at-the-money implied volatility (IV) 8% above the day's first reading, or the
put/call volume ratio 1.5× the first reading (≥ 1.0, ≥ 500 contracts). Does the alarm improve the slice's trades,
and does a model trained on stock and options bars (on the GPU) exit better?

## 1. Data (Alpaca; options history starts February 2024)
- NVDA and MU 5-minute stock bars (SIP), 2024-02-01 → 2026-10-20, regular hours.
- Each day, each stock: option contracts on the nearest expiry at least 5 days away, strikes within ±5% of the 9:35
  price; their 5-minute bars (trade-based close and volume).
- **ATM IV at a bar:** the mean of the call and put IV (Black-Scholes, 4% rate) at the strike nearest the stock's bar
  close, from each option's latest traded bar close in the last 30 minutes (stale beyond that: no reading).
- **Put/call ratio at a bar:** cumulative put volume ÷ cumulative call volume since the open, over those contracts.
- A day with no option data for a stock is skipped for that stock (both variants, so the comparison stays matched).

## 2. Trades (the live slice's rules)
- **Entry:** if the 9:30–9:35 bar closes above its open, buy at the 9:35 bar's open.
- **Variant A (no alarm):** at the 15:50 bar's open, sell if at or above cost; otherwise keep, then sell at the first
  bar open at or above cost, at −4% from cost, or at the 15:50 bar of the third session; never add.
- **Variant B (alarm):** as A, plus: from the 9:40 bar on, the day's first IV / put-call reading is the base; at each
  bar close, if the alarm condition holds, sell at the next bar's open (even at a loss). Each held day takes a new base.
- Cost 3 bps per side. One trade per stock per day; a stock still held from earlier days is not bought again.

## 3. Grid and selection
IV jump ∈ {5%, 8%, 12%, 20%} × put/call rule ∈ {off, on (1.5×, ≥ 1.0, ≥ 500 contracts)} = 8 alarm variants.
- **Measure:** per trade, net return of B − net return of A (the alarm's value), mean and t clustered by entry day.
- **Development 2024-02-01 → 2025-06-30:** the variant with the highest mean B − A.
- **Final 2025-07-01 → 2026-10-20, once:** pass if its mean B − A > 0 with t ≥ 2. The live setting (8%, put/call on) is
  reported for both periods whatever is selected.
- Also reported: A and B's mean return per trade, hit rate, holding time, and how often the alarm fires.

## 4. Learned exit model (GPU)
- **Target:** the stock's return over the next 30 minutes (6 bars), from each 5-minute bar close 9:40 → 15:20.
- **Inputs (known at the bar close):** the last 12 bar returns and volume ratios, time of day, return since the open,
  ATM IV level and change since the base, put/call ratio and its change, call-minus-put IV, the stock (NVDA/MU).
- **Model:** a small neural network (2 hidden layers, 64 units), trained on development data only: 2024-02 → 2025-03
  to fit, 2025-04 → 2025-06 to stop early and to choose the exit threshold (one of the 10th / 20th / 30th
  percentiles of its predictions). Fixed seed, then frozen.
- **Exit rule C:** as A, plus sell at the next bar's open when the prediction is below the chosen threshold.
- **Final, once:** report the prediction's rank correlation (IC) with the realised 30-minute return, and C − A per
  trade. Pass if C − A > 0 with t ≥ 2. Runs on the GPU (DirectML) if available, otherwise on the CPU; the result
  does not depend on which.

## Live
Whatever passes replaces the current alarm rule on the paper slice; if nothing passes, the slice keeps the rule as
it is (an untested experiment, as the user asked), and the result is shown on the Auto-trade page.

## Amendment A (2026-10-21, before any option or model result was seen): a larger model on much more data, on the GPU
The user asked for a larger model trained on more data (the earlier small models found almost nothing). Section 4 is
replaced by:
- **Data:** 5-minute bars (SIP, split-adjusted) 2019-01-02 → 2026-10-20 for the 100 most-traded NASDAQ stocks of each
  year (top 100 by 20-day dollar volume on the year's first trading day, from the v6 panel, so the universe is point in
  time; a stock is used only in its own years), plus NVDA, MU and QQQ.
- **Target:** the stock's return over the next 30 minutes (6 bars) from each bar close 9:40 → 15:20, divided by its
  trailing 20-day 5-minute volatility (training only; trading uses the sign and rank).
- **Inputs (known at the bar close):** the stock's last 48 bar returns and volume ratios (vs its 20-day average for that
  time of day), return since the open and since the prior close, the same for QQQ, time of day, day of week. No options
  inputs (options history covers only 2024 onward and only two stocks; the alarm keeps those).
- **Model sizes, all counted:** small (2 × 64), medium (3 × 256), large (4 × 1024) fully connected networks, plus a large
  sequence model (1-D convolution over the 48 bars, 4 layers × 256 channels). Adam, fixed seed, early stopping.
  - Fit 2019-01 → 2024-12; **validation 2025-01-02 → 2025-06-30** picks the size (highest rank correlation, IC) and the
    exit threshold (10th / 20th / 30th percentile of predictions).
- **Final 2025-07-01 → 2026-10-20, once, for the chosen model:**
  - IC over all stocks (reported, with t over days);
  - **exit rule C** (as A, plus sell at the next bar's open when NVDA/MU's prediction is below the threshold);
  - **entry rule D** (as A, but buy at 9:35 only if the first 5 minutes closed up *and* the prediction is above 0).
  - Pass for each: mean (C or D) − A per trade > 0 with t ≥ 2.24 (two rules: Bonferroni at 5%).
- Trained on the RX 7800 XT through DirectML. Stated limitation: the 2025-07 → 2026-10 period was already used in
  earlier daily studies, but no 5-minute model was ever evaluated on it.
