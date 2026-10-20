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

## Results (added 2026-10-21 after the complete run; nothing above was changed)
The LightGBM walk-forward picks were recomputed on the current data panel (the saved prediction files no longer lined
up with it after the universe changed). Excess return per trade over QQQ bought at the same open and sold on the same
exit day, after 10 bps round-trip costs; t over entry days:

| take-profit u | development 2021–24 | final check 2025–26 | beat QQQ | days held | exits: target / stop / time |
|---|---|---|---|---|---|
| 1% | −0.42% (t −8.9) | −0.35% (t −4.9) | 52% / 58% | 1.8 / 1.9 | 76% / 17% / 7% |
| 1.5% | −0.39% (t −7.6) | −0.35% (t −4.4) | 54% / 59% | 2.2 | 68% / 20% / 12% |
| 2% | −0.37% (t −6.5) | −0.28% (t −3.2) | 55% / 59% | 2.5 | 61% / 22% / 17% |
| **3% (chosen)** | **−0.36% (t −5.7)** | **−0.23% (t −2.3)** | 52% / 54% | 3.0 | 48% / 26% / 25% |

**Fails.** The rule beats QQQ on more than half of its trades, but loses on average and significantly, in both periods.
Capping the winners at the target while the stop takes full −5% losses gives many small gains and fewer, larger losses;
on top, the top-ranked stocks lag QQQ (as the v6 final test found for long-only). Note: the back-test assumes the stop
is hit first when both happen on the same day, which is conservative; with the target first the loss would be smaller
but, at these t-values, still negative.
