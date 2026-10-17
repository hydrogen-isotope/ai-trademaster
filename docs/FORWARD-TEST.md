# Frozen-model forward test (real NASDAQ data, predictions only)

**Purpose:** the only fully clean out-of-sample test. Models are frozen *before* the session. Every prediction is written to disk before its outcome exists, and is scored only after the outcome bar completes. Nothing is retrained during the test. No orders are sent.

## What's frozen (`private/models/forward/frozen-2026-10-17/`, SHA-256 in `manifest.json`)
| Model | Source | Predicts |
|---|---|---|
| **B1** | `private/models/b1/v2-h6` (3-seed attention ensemble, ONNX, CPU) | next 30 min (6 × 5-min bars) return minus the stock-universe average |
| **Ridge** | `private/models/research/5m-h6` final weights | same target |
| **B0** | rule: −0.5 × 30-min residual vs reference ETF | same target |

The logger re-checks every hash at start and refuses to run if any artifact changed. To test a different model, freeze a **new** set; never edit this one.

## How it runs
```
python -m trademaster_ml.forward live      (or double-click tools\forward-live.cmd, which also starts the dashboard)
```
- Can be started any time. It waits for the next session's 09:30 ET open: **Mon 19 Oct 2026 = 21:30 Singapore time**. Weekends and listed holidays are skipped; early-close days aren't handled yet.
- After each completed 5-minute bar (+25 s) it fetches the latest bars for the 93 symbols and computes point-in-time features. It then logs the B1, ridge and B0 predictions for every valid stock to `private/data/forward/live-<date>/predictions.jsonl`, and scores predictions whose 30-minute outcome has completed into `outcomes.jsonl`.
- It updates `summary.json` after every poll, and the dashboard page **http://127.0.0.1:5080/forward** refreshes every 15 s:
  - per-model rank IC (with overlap-adjusted t);
  - direction hit rate;
  - long/short quintile P&L gross and net of 1.5 bps one-way (exit of the last position included);
  - cumulative net curve, IC per bar, the latest top and bottom predictions, and the frozen hashes.
- It keeps the PC from idle-sleeping only while running (Windows `SetThreadExecutionState`, released on exit). No settings are changed, and it's CPU only.
- It stops at 16:10 ET (04:10 Singapore time).

## Rehearsal
`python -m trademaster_ml.forward replay --date 2026-10-16` runs the identical pipeline over Friday's session (`private/data/forward/replay-2026-10-16`). Result: B1 rank IC +0.064, ridge −0.037, B0 −0.026, over 73 scored bars. That session was already inside the research data, so this is a **pipeline check, not evidence**.

## Reading the results
- **One day proves nothing.** Rank IC per bar is noisy (±0.1 or more is normal). Judge only after weeks of sessions. The research suggests |IC| ≈ 0.01–0.03 would be the realistic best case.
- **The net P&L line assumes taker costs.** Per [EXECUTION-RESEARCH.md](EXECUTION-RESEARCH.md), passive execution might lower them, but that isn't calibrated yet.
- **Data:** Alpaca Market Data API, IEX real-time feed. Prices are real time; volumes cover IEX only, but every volume feature is a ratio to the same feed's recent volume. Missing symbols in a poll are logged in `errors` and masked. Needs a free Alpaca key ([MARKET-DATA.md](MARKET-DATA.md)); without one the logger stops with a message. Sessions before 19 Oct 2026 used Yahoo data.
