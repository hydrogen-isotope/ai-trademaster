# v8: six specialised stage models, with a local LLM reading the news (rules written before any v8 result, 2026-10-18)

## Why
The user's view is that news, such as the current AI anxiety, should move the forecasts. Our earlier news inputs could
not test this fairly: FinBERT sentiment and keyword themes (v4). Published evidence says GPT-4-class models reading
headlines predicted next-day returns, while BERT-class models did not (Lopez-Lira & Tang). v8 tests a local LLM news
stage inside a staged design:

1. news;
2. price;
3. fundamentals;
4. industry;
5. market and volatility;
6. combiner.

## Look-ahead control for the LLM
The model is Qwen2.5-7B-Instruct (Q4_K_M, run locally with llama.cpp). Its training data reportedly ends around
2024-06-30. Anything learned from news is therefore **tested only on headlines published from 2024-07-01 onward**. No
headline before that date is used to fit or test any news-based weight.

## Stage definitions (all walk-forward and point-in-time; refit on the first trading day of each month)
- **S1 News.** Each headline (title only) goes to the LLM with a fixed prompt, committed in `llm_news.py` before
  scoring. The answers come back as single tokens, scored from token probabilities:
  - (a) direction for each tagged ticker: P(+), P(0), P(−), giving direction = P(+) − P(−);
  - (b) new information vs recap/preview/opinion: P(new);
  - (c) AI-related: P(yes);
  - (d) about the whole market or economy rather than one company: P(yes).

  A headline counts from its timestamp. Up to 16:00 New York time it counts for that day's close; after 16:00 it counts
  for the next trading day.

  Per stock-day:
  - `news_dir` = the sum of direction × P(new) over the last 5 trading days, divided by √(number of items);
  - `news_count`.

  Per day:
  - `ai_anxiety` = −(mean direction over AI-related items), weighted by P(AI) × P(new);
  - `mkt_news` = mean direction over market-wide items, weighted by P(market) × P(new).
- **S2 Price**, **S3 Fundamentals**, **S4 Industry.** Each stage is a ridge (α = 10) on its own feature family's ranks,
  using the v6 feature panel; the output is a score.
- **S5 Market:** the QQQ GJR-GARCH-t variance (as in v7) and the market drift. The drift becomes news-tilted only if
  test C passes.
- **S6 Combiner:** a ridge (α = 1) on the cross-sectional ranks of the stage scores. It is fitted on each stage's
  previous out-of-sample walk-forward scores, so it never sees in-sample stage fits. Its output feeds the v7 price
  distribution in place of the v6 tilt.

## Tests
Metric definitions follow v6 and v7: daily rank IC with Newey-West t, and a block bootstrap (blocks of 5 days).

- **A. Staging vs one ridge (no news):** S2–S4 + S6 vs v6 R.
  - Report development 2021–2024, and 2025–26 labelled *seen*.
  - Pass: the development IC is higher, and the seen-period IC is not lower by more than 0.005.
- **B. LLM news adds to the stock ranking.** S1–S4 + S6 vs S2–S4 + S6, with predictions 2025-01-02 → latest.
  - The combiner is fitted only on stage outputs from 2024-07 onward.
  - News coverage is limited to the scored universe (the top ~300 by dollar volume, plus market and AI items).
  - Pass: the IC difference is > 0 with a 95% block-bootstrap CI excluding 0.
- **C. News predicts the market.** Daily origins 2024-07-01 → latest; `ai_anxiety` and `mkt_news` vs QQQ's next 5-day
  log return.
  - Report the IC and hit rate for each index, with Newey-West t (lag 4).
  - Pass, for at least one index: |t| > 2, and the IC has the same sign in both halves of the window.
  - If it passes, S5's drift becomes a ridge of the next 5-day QQQ return on the passing index, refitted monthly on data
    from 2024-07 onward. Otherwise S5 keeps the shrunk long-run drift.
- **D.** Passing components go into the forward shadow test from the first trading day after they pass, with the same
  promotion rule as v7.

**Expectations, stated in advance:** B is plausible but uncertain for a 7B model on large caps. C is unlikely to pass.
The market prices widely read news quickly, and the window is about 570 days.

## Amendments

**A (2026-10-18, after the 200-headline hand check and before any v8 news result).** What the hand check showed:
- Direction labels were sensible.
- The LLM's "new vs old" question did not discriminate. It marked about 100% of headlines as new, including
  "What's Going On With Broadcom Stock?" and "Earnings Scheduled For …".
- Speed was 0.30 s per headline, about 12 h for the window.

Changes:
- **Recap / template filter.** P(new) is replaced by a fixed list of Benzinga template patterns (`RECAP` in
  `llm_news.py`: "What's Going On With", "Why … Shares Are Trading …", "Stocks Moving", "Whale Activity", "Earnings
  Scheduled For", "Options Activity", "Stocks To Watch", "Preview", 52-week high/low, P/E or short-interest
  "Insights", "Top N", "N … Stocks", and similar). Matching headlines (18% of the fetched sample) are not scored and
  get weight 0. Every other headline gets weight 1.
- **One topic question.** The AI and market questions are merged: "mainly about artificial intelligence, the overall
  market or economy, or other?"; the answer word is "other", because "neither" is two tokens in Qwen's vocabulary and can't be read from the first token
  - `ai_anxiety` uses P(ai).
  - `mkt_news` uses P(market).
- **Direction** is asked for at most 2 tagged tickers per headline.

Nothing else changes.

**B (2026-10-18, before any news result). The user chose rolling news only; no historical back-scoring.** Headlines are
fetched and scored each day for the last 7 calendar days and deleted after 10 days. Tests B and C (historical) are
therefore dropped. The news stage is judged **only by a forward shadow test**, which also removes any LLM look-ahead.

The weights are fixed now, as priors, because they cannot be estimated without history:
- **Stock news score** `news_dir` (stock i, close t): the sum of direction (P(good) − P(bad)) over its non-recap headlines
  in the 5 trading days ending at t (same timestamp rule as above), divided by √count, clipped to [−2, 2].
- **Market direction:** for headlines with P(ai) + P(market) ≥ 0.5, one extra question: good, neutral or bad news for
  the overall US stock market.
  - **Market news index** S: the mean of that direction, weighted by P(ai) + P(market), over the same 5 days. S = 0 if
    there are fewer than 10 such headlines.
  - **AI-anxiety** (display only): minus the same mean over AI headlines.
- **News-adjusted forecast ("v7+news"):** the v7 mean + β·(1.0%·S) + 0.25%·`news_dir` (5-day log return), with v7's
  standard deviation and recalibration. That gives at most ±1% from the market and ±0.5% from stock news.
- **Forward shadow test:** both v7 and v7+news are logged daily. After ≥ 60 forecast dates, v7+news is promoted only if
  both hold:
  1. its mean CRPS is lower than v7's, with a 95% block-bootstrap CI excluding 0;
  2. its up/down hit rate is ≥ v7's.

  Until then the dashboard labels it "news-adjusted (unproven)".

**C (2026-10-18, before any result of it). Per-stock Qwen analysis that advises the forecast range.** At the user's
request, the lean comes from a slower, per-stock Qwen analysis that also sees the quantitative stages. The rules are
fixed now:
- **Input per stock (one prompt):**
  - S2–S4 facts as numbers: 1-week, 1-month and 12-1 momentum with their percentile ranks; 60-day volatility;
    distance from the 52-week high; NASDAQ fundamentals (earnings yield, sales/price, gross profitability, earnings
    surprise, days since filing); industry 1-month and 6-month moves; rank score and decile;
  - the v7 model-only median and 80% range;
  - the market news index;
  - cross-stock context: up to 5 same-industry peers with the strongest 5-session news, with Qwen's earlier read;
  - up to 40 of the stock's own headlines from the last 30 days (recap templates removed). SGX has no headline source
    and gets facts and peers only.
- **Output:** JSON enforced by a schema, with reasons first (≤ 3 positives, ≤ 3 negatives), then `lean` (strong_down,
  down, neutral, up, strong_up), `confidence` (low, medium, high) and `risk` (normal, elevated). An unparsable reply
  is treated as neutral, normal, low.
- **Mapping:**
  - the model median m shifts by k·σ, where σ is the stock's v7 5-day standard deviation; k = ±0.30 for strong,
    ±0.12 for up/down, 0 for neutral;
  - k is multiplied by 0.5 for low confidence, 1.0 for medium, 1.25 for high;
  - elevated risk multiplies σ by 1.25;
  - quantiles and P(up) use v7's recalibration levels for that date.
- **Logging:** every advised forecast is written with its model-only twin, before outcomes exist, to
  `private/reports/evals/<market>/advice_shadow.jsonl`.
- **Promotion**, after ≥ 60 forecast dates and ≥ 300 stock-forecasts, requires all three:
  1. mean CRPS lower than model-only, with a 95% block-bootstrap CI (blocks of 5 dates) excluding 0;
  2. an up/down hit rate ≥ model-only's;
  3. pooled 80% coverage within 75–85%.

  Otherwise the advice weights are set to 0.
- **Which stocks:**
  - any stock the user clicks;
  - at app start (GPU, user present): the top and bottom 25 by rank score, plus the top and bottom 10 by model P(up).
- Results are cached per stock and close date.

**D (2026-10-18, after the amendment C hand check and before any advised forecast was logged or scored).** The first
prompt (8 heavy-news NASDAQ stocks) failed the hand check:
- Qwen answered "neutral, low" for MU, AVGO, CRWV, NVDA and SNDK, despite a week of mostly bad news for each. The
  prompt's line "use neutral with low confidence when the information is thin, old or mixed" dominated.
- Its "reasons" were copied headlines, including a peer's headline cited as a reason for the stock itself.

Changes (wording only; inputs, schema, mapping, logging and promotion are unchanged):
- Each reason must be one sentence in Qwen's own words, saying what happened and why it matters for the stock's price.
  Headlines may not be copied, peer news counts only if it clearly affects the stock, and recent days weigh more.
- "Pick a direction when the recent news clearly leans one way; use neutral only when the evidence is balanced or there
  is almost no news."

The hand-check outputs from the first prompt were moved out of the shadow log (`evals/handcheck1/`); they are not part
of the forward test.

Also found during the hand check: with only `-ngl 0`, the Vulkan llama.cpp build still sends prompt batches to the GPU.
"CPU" runs now pass `--device none --no-op-offload` (true CPU: about 160 prompt tokens/s instead of about 800).

## Results

### Test A: staging vs one ridge (no news), run 2026-10-18

| model | dev IC (t) | dev top>bottom | dev L/S Sharpe | seen IC (t) | seen top>bottom | seen L/S Sharpe |
|---|---|---|---|---|---|---|
| v6 R (one ridge) | +0.0673 (8.3) | 66% | 1.93 | +0.0588 (5.0) | 67% | 2.83 |
| S2 price alone | +0.0679 (8.1) | 62% | 1.87 | +0.0612 (5.3) | 70% | 3.27 |
| S3 fundamentals alone | +0.0401 (6.0) | 59% | 1.04 | +0.0368 (3.3) | 55% | 0.60 |
| S4 industry alone | +0.0070 (1.1) | 53% | −0.15 | +0.0147 (1.2) | 50% | 0.02 |
| **S6 combiner of S2–S4** | **+0.0680 (7.8)** | 61% | 1.41 | **+0.0614 (5.0)** | 67% | 2.76 |

**Test A: PASS** by the written rule: the development IC is higher (+0.0680 vs +0.0673), and the seen IC is not lower
(+0.0614 vs +0.0588).

The gain is negligible, though, and the combiner's development portfolio is weaker (Sharpe 1.41 vs 1.93). Splitting by
stage adds no information; the price stage carries almost all of it. The combiner's weights at the last refit were
price +0.055, fundamentals +0.018, industry +0.001.

### Tests B and C
(pending: news back-scoring)
