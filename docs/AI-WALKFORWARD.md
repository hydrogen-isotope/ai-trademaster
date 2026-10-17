# B1 v2 — walk-forward evaluation (2026-10-17, CPU only)

> **Correction (later the same day):** the "net" figures below come from a simplified per-rebalance cost model that **undercounts trading costs**. A full portfolio simulation in `trademaster_ml.research` (dollar-neutral, flat at every close, costs on every weight change) puts **every model net negative** on this 5-minute data. The pre-selected strategy came out flat on a 15-session hold-out (t = 0.06). See [AI-RESEARCH.md](AI-RESEARCH.md). The rank-IC findings below still stand.

**Setup**
- **Universe:** 88 large NASDAQ stocks plus SMH, SOXX, QQQ, SPY and XLK. 60 sessions of 5-minute bars from Yahoo (unofficial; personal research only).
- **Features** (point-in-time): r1, r3, r6, r12, vol12, residual vs reference ETF, volume surprise, time of day, sector-relative 30-min momentum, distance from session open, position in the session range.
- **Target:** each stock's forward return **minus the average forward return of all stocks** at that bar. This is a cross-sectional (relative-value) target; market direction is removed.
- **Walk-forward:** 5 folds. Each trains on all earlier sessions, picks the model on the next 5 sessions and tests on the following 5. That gives 25 out-of-sample sessions. B1 is a 3-seed ensemble per fold.
- **Trading test:** long the top quintile and short the bottom quintile by prediction, rebalanced once per horizon (no overlapping holds). Costs are **1.5 bps one-way** per changed name, which is optimistic for some names. No market impact, no borrow cost.

`python -m trademaster_ml.train_wf --hist private/data/hist2 --out private/models/b1/v2 [--horizon 6]`

## Results (pooled out-of-sample)
| Horizon | Model | Rank IC (t, overlap-adjusted) | Gross L/S (bps/rebalance) | Turnover | **Net (bps/rebalance)** | Net per fold |
|---|---|---|---:|---:|---:|---|
| 10 min | B0 | +0.019 (3.4) | +0.61 | 0.47 | **−0.80** | 1 of 5 positive |
| 10 min | Ridge | +0.013 (1.9) | −0.05 | 0.51 | −1.58 | 0 of 5 |
| 10 min | B1 | +0.002 (0.3) | −0.35 | 0.29 | −1.23 | 1 of 5 |
| 30 min | B0 | +0.026 (2.8) | +2.94 | 0.79 | **+0.57** | 3 of 5 |
| 30 min | Ridge | +0.023 (1.6) | +5.09 | 0.67 | **+3.09** | 3 of 5 |
| 30 min | B1 | +0.015 (1.3) | +2.74 | 0.46 | **+1.35** | **5 of 5** |

## What this does and doesn't show
- There is a small, consistent **short-term cross-sectional reversal**: stocks that ran ahead of their sector tend to give some back. At 10 minutes it doesn't survive costs. At 30 minutes all three models are net positive on paper, and B1 is positive in every fold.
- **Not enough evidence to trade.** It's 25 days of one regime, the t-stats are 1.3–2.8, costs are optimistic, there's no market impact or borrow cost, and the data is unofficial.
- **Survivorship bias:** the universe is today's large caps.
- **Selection bias:** the 30-minute horizon was chosen after seeing the 10-minute result.
- Confirming it needs months of data, a licensed feed, a fresh untouched hold-out period, and realistic spread and slippage from quotes rather than bars.
- Next AI steps: longer history, a hysteresis or thresholded entry rule to cut turnover, a calibrated σ for sizing, and running B1 inside the engine on a 5-minute cadence.

ONNX exports: `private/models/b1/v2` (10 min) and `private/models/b1/v2-h6` (30 min). Both pass the C# parity test.
