# Real-data check: B0 baseline on NASDAQ, 12–16 Oct 2026

The first test of the B0 baseline on real data: **predictions only, no orders**. CPU only.

- **Data:** Yahoo Finance chart API, 1-minute bars, regular session, Mon 12 Oct 09:30 → Fri 16 Oct 16:00 ET. Free and unofficial, for personal research only; not a licensed feed.
- **Method** (`tools/eval_bars.py`): point-in-time. At each minute, B0 sums the stock's residual against its reference ETF over the last 30 minutes and predicts 10-minute return = −0.5 × that sum. Beta comes from an EWMA (300-bar half-life) using only past bars. Overnight gaps are excluded. Parameters are the engine defaults, **not fitted** to this data.
- **Refresh:** live every 10 minutes on 16 Oct (`tools/live_watch.ps1`).

| Stock | Ref | Scored | Corr (5 d) | Direction right | RMSE model vs "no change" (bps) |
|---|---|---:|---:|---:|---|
| MU | SMH | 1,751 | −0.22 | 46% | 43.9 vs 34.4 |
| NVDA | SMH | 1,751 | −0.05 | 51% | 22.4 vs 18.8 |
| AMD | SMH | 1,751 | +0.02 | 48% | 38.5 vs 35.5 |
| AAPL | QQQ | 1,751 | +0.05 | 48% | 19.3 vs 14.9 |

Per-session correlations ranged from −0.33 to +0.28, with no stable sign except MU, which was negative on 4 of 5 days.

## Conclusions
- **B0 has no usable edge on these stocks.** It was worse than predicting zero on all four, and direction accuracy was at or below a coin flip.
- MU's residual against SMH tended to *continue* rather than revert over 10–30 minutes. That's the opposite of B0's assumption. Don't flip the sign and trade it: overlapping 10-minute windows overstate significance, it's 5 days of one stock, and costs aren't included.
- B0 can't see market-wide moves (for example the broad semiconductor selloff after 14:40 on 16 Oct). It only models moves relative to the ETF.
- NVDA is a large part of SMH, so its residual against SMH is partly against itself. A better reference for it would be an equal-weight basket or a different index.

## Next
B1 (cross-stock neural model, trained on the CPU) must beat "no change" and B0 on held-out real sessions before any paper trading. It will need more history (weeks of 1-minute bars), market-wide features, and a licensed data source for anything beyond research.
