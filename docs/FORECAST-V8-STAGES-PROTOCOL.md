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

**E (2026-10-18, at the user's request, before any advised forecast matured). Qwen first, then our models; a
research signal; three views.** This replaces the lean → k·σ mapping of amendment C. Inputs, schema-enforced reasons,
the cache and the logging rule are unchanged.
- **Qwen's probability.** After the JSON verdict, one more question in the same conversation: "what is the chance
  (0 to 100) that the price is higher in 5 trading days than at the last close? Answer with a whole number only."
  - Qwen writes numbers digit by digit, so the first answer token is the tens digit.
  - P_qwen is the expectation over that token's digit probabilities (digit d counts as 10d + 5).
  - A first version asked "rise or fall?" (one word). On the first hand-check stock it just repeated the verdict with
    about 100% certainty (MU: P(rise) = 0.00), so it was replaced before any other stock was run. That output is in
    `evals/handcheck3/`.
- **Post-processing by our models.** P_total = P_model + w · (P_qwen − 0.5), clipped to [0.05, 0.95]:
  - P_model is the v7 model-only P(up);
  - w is a fixed shrinkage, because Qwen has no track record. It is set once from the hand check's raw P_qwen spread,
    which involves no outcomes: w = 0.12 / (90th percentile of |P_qwen − 0.5| over the 10 hand-check stocks), capped at
    0.5.
  - After ≥ 60 matured dates, w may be re-fitted once, by a logistic regression of the up/down outcome on
    (P_qwen − 0.5) with offset logit(P_model). It is then frozen again and reported as a new amendment.
- **Distribution.** The v7 shape and recalibration levels are kept, with the median shifted so that the calibrated
  P(up) equals P_total. Elevated risk still widens σ by 25%.
  - The headline band is the 50% range; the 80% and 95% ranges stay as outer bands.
  - The ranges are not narrowed: they matched their nominal coverage in testing.
- **Views** (all logged per stock and close):
  - *Total*: the above;
  - *Qwen only*: P_qwen without shrinkage, on the same shape;
  - *Model only*: v7.
- **Research signal** (not advice), from P_total: **buy** if ≥ 0.58, **sell** if ≤ 0.42, otherwise none.
- **Forward test.** The promotion rule of amendment C applies to *Total* vs *Model only*. Also reported:
  - *Qwen only*'s CRPS and hit rate;
  - each signal's hit rate (buy: the 5-day return > 0; sell: < 0) with a binomial 95% CI.
- The amendment C/D outputs (12 stocks) were moved to `evals/handcheck2/` and are not part of the forward test.
- **Hand check (10 NASDAQ stocks, CPU, no outcomes).**
  - P_qwen took only three values, one per verdict: 45% for "down" (MU, AVGO, CRWV, SNDK), 55% for "neutral"
    (INTC), and 64–66% for "up" (NVDA, AAPL, TSLA, AMD, COST).
  - So it is graded only by its verdict, and it leans up: "up" sits 15 points above 50, "down" only 5 below.
  - The 90th percentile of |P_qwen − 0.5| is 0.155. By the rule, w = 0.12 / 0.155 = 0.77, so **w = 0.5** (the cap).
  - Resulting signals: buy for NVDA, AAPL, TSLA and AMD (60–62%); none for the rest; no sell.
  - The asymmetry is a known weakness. The 60-day re-fit may add an intercept to correct it.
  - These outputs are in `evals/handcheck4/` and are not part of the forward test.

**F (2026-10-19, with v7 amendment A, before any 1-day or 20-day Qwen result). Qwen at three horizons.**
- In the same conversation after its verdict, Qwen is asked for the chance (0–100) that the price is higher in 1, 5 and
  20 trading days. These are three questions, each read the same way as in amendment E.
- Each horizon's P_total = P_model(h) + 0.5 · (P_qwen(h) − 0.5), clipped to [0.05, 0.95], using that horizon's v7 shape
  and recalibration levels.
- The same buy/sell thresholds apply (0.58 / 0.42).
- Every horizon is logged to its own shadow file (`advice_shadow.jsonl` for 5 days; `advice_shadow_h1.jsonl`,
  `advice_shadow_h20.jsonl`).
- The same promotion rule applies per horizon. A horizon whose v7 model failed its gate is not shown.

**G (2026-10-19, at the user's request, before any result). Product knowledge, tested A/B against the current
analysis.**
- **Company profile**, built once per company and rebuilt when it is more than 120 days old.
  - Source text:
    - NASDAQ: the "Business" section (Item 1) of the latest 10-K filed on SEC EDGAR, or Item 4 of a 20-F;
    - otherwise, and for SGX: Eulerpool's company description.
  - Qwen turns the text into schema JSON: up to 6 products with an approximate revenue share; up to 6 customers,
    suppliers and competitors (with a US ticker where Qwen gives one); up to 5 sales drivers; a summary of at most
    300 characters.
  - The source and the filing date are stored with the profile. The text is only ever older than the forecast, so
    there is no look-ahead.
- **Variant B (product knowledge).** The amendment C–F prompt, plus the profile, plus related-company news.
  Related-company news means the last 5 sessions' headlines (at most 3 each, recap templates removed, up to 6
  companies) for the profile's customers, suppliers and competitors that map to a ticker. Everything after the prompt
  is unchanged: the verdict, the chance per horizon, w = 0.5, and the signals.
- **Variant A.** The current analysis, unchanged.
- **Both run on every evaluation.** Both are logged before outcomes exist: A to the existing files, B to
  `advice_shadow_prof*.jsonl`.
- **The page shows B**, labelled "with product knowledge (A/B test)", and A in the comparison table.
- **Decision** after ≥ 60 matured dates and ≥ 300 forecasts per horizon:
  - B replaces A only if its mean CRPS is lower than A's, with a 95% block-bootstrap CI excluding 0, and its up/down
    hit rate is ≥ A's;
  - otherwise the page goes back to A.

  Each variant also keeps its own amendment E test against model-only.
- **Addendum (same day, after looking at profile quality only, before any outcome).**
  - *Source text.* Qwen gets the first 3,000 characters of the Business section, then the most specific sentences
    (named companies, relationship words, revenue shares), up to 17,000 characters in total. These come from the
    Business section first; the rest of the filing may fill at most 35%. Taking the first 12,000 characters missed AMD's
    foundries (TSMC, GlobalFoundries) and its console customers (Sony, Microsoft).
  - *Profile instructions.* Name only real companies, never categories. Give revenue shares only where stated or well
    known.
  - *Clean-up (no model).* Categories ("PC OEMs", "Hyperscale cloud providers"), the company itself, and duplicates are
    dropped.
  - *Tickers.* A strict name match against SEC's ticker list (shares only, never notes or ETNs, e.g. Credit Suisse
    AG's GLDI). A name that is itself a ticker ("AMD") is used as is. A short list maps brand and unit names to their
    listed parents (AWS → AMZN, Google Cloud → GOOGL).
  - *Existence check.* Only for companies that are neither named in the filing nor have a current US ticker, i.e.
    ones that came from Qwen's (2024) memory.
    - The check uses market data: Eulerpool's search finds the company's shares and their trades in the last 30 days.
      Shares with no trades mean delisted, and the company is removed (Credit Suisse).
    - Only if Eulerpool does not know the company at all, Qwen is asked yes/no and the company is removed at
      P(no) ≥ 0.6 (Qimonda).
    - An earlier one-shot "which no longer exist?" question removed real companies (Sandisk, re-listed in 2025;
      Powerchip) and missed Credit Suisse, so it was replaced before any outcome.

**H (2026-10-19, before any advised forecast matured; the first 1-day outcome is the US close of 2026-10-19).
Qwen's up-lean, and profile data.**
- **Observed (no outcomes involved).** Over 40 five-day evaluations of stocks chosen evenly from both ends of the
  ranking, Qwen said "up" 24 times and "down" 12 times. Its mean chance of a rise was 0.56 against the model's 0.50.
- **Symmetric question.** After the verdict, Qwen is asked both the chance (0–100) that the price is *higher* and the
  chance that it is *lower* in h trading days, each read as in amendment E.
  P_raw = P_higher / (P_higher + P_lower). This removes the tendency to agree with the direction a question suggests.
- **Re-centring.**
  - Qwen's average lean over its recent evaluations is removed: P_qwen = P_raw − (mean(P_raw) − mean(P_model)).
  - The window is the last 200 evaluations of that horizon (all stocks, before this one). It uses the market's own
    window once it has ≥ 30 evaluations, otherwise all markets pooled, otherwise no shift.
  - Qwen then decides only how a stock compares with the rest; the market-wide direction comes from the tested model.
    No outcome is used.
  - P_total = P_model + 0.5 · (P_qwen − 0.5) as before. P_raw, the shift and P_qwen are all logged.
- **Profile data (variant B only).**
  - **Revenue split** from Eulerpool's segment data, which comes from the company's XBRL filings. It uses the latest
    fiscal period that ended at least 75 days before the forecast close and whose shares sum to 90–110%, plus the year
    before. These numbers replace Qwen's revenue shares.
  - **Peers** from Eulerpool's peer list are added to the competitors, with the same clean-up and ticker rules. For SGX
    peers, the prompt gets their 5-session price moves from our SGX data, since SGX has no news source.
  - **Reverse links.** If company X's profile names Y as a customer (supplier), Y's profile gains X as a supplier
    (customer), marked "named by X".
  - **Rebuild** when a newer annual report than the profile's source exists (checked at most weekly), instead of after
    120 days.

**I (2026-10-19, 07:36 New York, before any advised forecast matured). Variant B's profiles are researched by
Claude Haiku.**
- **Why.** In a 20-company pilot, Haiku research found named relationships with evidence that the local 7B model
  missed: NVDA's foundries and assemblers, PACCAR's engine and transmission suppliers, ASML's lens supplier, Roku's
  named competitors. Some of its evidence was stale (e.g. Singapore Airlines aircraft retired years ago), so recency
  rules are added.
- **Coverage.** The 300 NASDAQ stocks with the highest 20-day traded value (as of the 2026-10-16 close) and all 119
  ranked SGX stocks.
- **Research rules.** Haiku subagents, about 10 companies each, start from the inputs (filing excerpt, Eulerpool
  description, segment data, peers) and use web search (at most 6 searches per company). Every relationship needs a
  named company, ≤ 25 words of evidence, a source URL and a source date. Sources must be dated 2023 or later, except
  the company's latest annual report. No prices, recent news, ratings or opinions.
- **Checks** (unchanged from G/H): size limits, categories, the company itself, duplicates, strict ticker matching,
  delisting by market data; the revenue split comes from Eulerpool segment data; reverse links come from all stored
  profiles.
- **Variant B** uses the checked Haiku profile where one exists, otherwise the amendment G/H Qwen profile; the page
  says which. The B shadow logs written with Qwen profiles before this amendment (a few test stocks) are moved to
  `evals/<market>/b_qwen_profiles/` and are not part of the forward test. B's forward test starts from this
  amendment.
- **Refresh.** Haiku profiles are re-researched when a newer annual report exists (as in H), or after 180 days.

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
