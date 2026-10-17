# AI research with a locked hold-out (2026-10-17, CPU only)

Harness: `python -m trademaster_ml.research`
1. Walk-forward on the **development** period only. Ridge and a 3-seed B1 ensemble are retrained per fold; B0 is rule-based.
2. Every **model × trading rule** combination is simulated as an intraday portfolio: dollar-neutral, gross exposure 1, flat at every close, costs charged on every weight change. The best combination by development Sharpe is pre-selected, and the full grid is reported.
3. The **hold-out** is evaluated once, for that pre-selected combination only.

- **Target:** each stock's forward return minus the cross-sectional mean of all stocks.
- **Universe:** 88 large NASDAQ stocks plus 5 ETFs.
- **Data:** Yahoo (unofficial; personal research only).
- **Costs:** 1.5 bps one-way, with a 3 bps sensitivity check.
- **Trading rules tested:** quintile long/short; hysteresis (enter top/bottom 10% or 20%, exit at 30% or 50%); signal-weighted with smoothing 0.1 or 0.3.
- **Models tested:** B0, ridge, B1 (μ), B1 (μ/σ).

## Results
| Data | Development OOS | Hold-out (locked) | Pre-selected | Hold-out net | Hold-out Sharpe (t) | At 3 bps |
|---|---|---|---|---:|---:|---:|
| 5-min bars, 30-min horizon | 15 sessions | 15 sessions | ridge + quintile | **+0.7 bps/day** | +0.24 (t 0.06) | −21.3 bps/day |
| 1-hour bars, 1-hour horizon, 2.9 years | 320 sessions | **120 sessions (~6 months)** | B1 μ/σ + hysteresis(0.1/0.3) | **−6.3 bps/day** | −2.01 (t −1.39) | −13.5 bps/day |

In development, **all 20 combinations were net negative** on both data sets. Best development Sharpe: −1.1 (hourly) and −3.5 (5-minute).

### Where the signal is: gross edge vs break-even cost
| Data | Strongest gross (development) | Gross bps/day | Turnover/day | Break-even cost (one-way) |
|---|---|---:|---:|---:|
| 1 h | B0 + quintile | +3.82 | 5.17 | **0.74 bps** |
| 1 h | B1 + quintile | +2.99 | 4.46 | 0.67 bps |
| 5 min | ridge + quintile | +14.26 | 14.06 | **1.01 bps** |
| 5 min | B0 + quintile | +12.82 | 17.42 | 0.74 bps |

### B1 uncertainty calibration
σ is about 4–5× too small: the standardized residual has std 4.1 (5-min) and 4.7 (hourly) against an ideal 1.0. Only 34–41% of outcomes fall within 1σ, against an ideal 68%. B1's σ must not be used for sizing until it's recalibrated (for example by scaling on validation data).

## Conclusions
1. **There's no tradable edge from public bar data with taker execution at 1.5 bps.** That holds over 2.9 years, across three model families, five trading rules and a 6-month locked hold-out. Short-term cross-sectional reversal is real (B0 rank IC about +0.01 to +0.03, positive gross), but costs exceed it.
2. **The deciding variable is execution cost, not prediction.** Break-even is about 0.6–1.0 bps one-way, below the half spread of many names. It would take *passive* execution (resting limit orders that earn the spread), which is the spec's "orders resting at the market with cutoff prices" design. That needs **quote and trade data**, a queue/fill model and adverse-selection measurement. Bars can't answer it.
3. **B1 adds nothing over the one-line B0 rule** on this data. A bigger network isn't the bottleneck; information and execution are.
4. **Correction to the earlier write-up:** the "+3 bps net at 30 min" in [AI-WALKFORWARD.md](AI-WALKFORWARD.md) came from an over-simple cost model and doesn't hold under full portfolio simulation.

## Recommended next steps
- **Execution research:** get licensed quote/trade (L1) history for a few names, then measure spread capture, fill probability and adverse selection for resting limit orders. Fold that into the simulated exchange.
- **Different information:** order-book imbalance or trade flow (needs quote data), earnings/news events, overnight and daily horizons.
- **Calibrate σ before any sizing,** and keep every future idea behind this harness and a fresh hold-out.
- **Forward test:** freeze a model and log its predictions live from Monday, with no retraining. That gives a hold-out nobody has seen.
