# Adaptive swing rule: buy the strongest at the open, sell into strength (rules written 2026-10-21, before any result)

## Question
The user's plan for the last 20 sessions before a holiday: at the market open buy what looks most promising, sell when
it has gone up enough, otherwise keep holding (with a stop and a time limit). Does this beat simply holding QQQ?

## Back-test (daily data; the live slice uses the Qwen + models signal, which has no history)
- **Picks:** each trading day t, the 3 stocks with the highest LightGBM ranking score at close t−1 (its out-of-sample
  walk-forward predictions, `private/reports/v6/pred_G_{dev,test}.npy`). Bought at the open of day t.
- **Exit, whichever comes first:**
  - take-profit: a day's high ≥ entry × (1 + u) → sold at that price;
  - stop: a day's low ≤ entry × 0.95 → sold at that price (if both happen on the same day, the stop is assumed first);
  - time limit: sold at the close of the 5th session (day t + 4).
- **u ∈ {1%, 1.5%, 2%, 3%}.** Cost 5 bps each way (10 bps round trip).
- **Benchmark per trade:** QQQ bought at the same open and sold at the same exit day's close.
- **Development 2021-01-04 → 2024-12-31:** u = the value with the highest mean net excess return per trade over QQQ.
- **Final check 2025-01-02 → 2026-10-09, once:** **pass** if the mean net excess per trade is > 0 with t ≥ 2 (t over
  trades, clustered by entry day).
- Also reported: hit rate, average days held, the share of trades ending at the target, stop and time limit.

## Live slice (whatever the back-test says, as a 20-session paper experiment the user asked for)
"Adaptive" replaces the Intraday and NVDA & MU slices (20% together). Picks: the 3 stocks with the highest Qwen + models
1-day chance of a rise among the latest evaluations (BUY signals first); exits as above with the chosen u. Results are
compared with holding QQQ, as for every slice.
