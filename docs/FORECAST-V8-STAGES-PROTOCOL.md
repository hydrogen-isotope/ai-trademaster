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
(none yet)

## Results
(not yet run)
