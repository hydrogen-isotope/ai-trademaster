# Market data for the dashboard

The markets page (`trademaster app`, or `tools\AI-Trademaster.cmd`) shows a watchlist, charts, quotes and trades.
It is **read-only**: it never places orders.

## Why not Yahoo Finance?
Yahoo has no official public API. Its Terms of Service (section 2(d)) forbid accessing or collecting data from its
services "using any automated means" without Yahoo's express prior permission. So the dashboard does not use Yahoo.

The research scripts used Yahoo's chart endpoint until 17 Oct 2026 and now use Alpaca too:

| Script | What it downloads | Alpaca feed |
|---|---|---|
| `fetch.py` | 1m / 5m / 1h bars for research (1h = session-aligned 09:30, 10:30, …) | `sip`, all exchanges, up to 15 min ago |
| `fetch1m.py` | 1-minute bars, merged into the growing `private/data/hist_1m_30d` | `sip` |
| `forward.py live` | the last 7 days of 5-minute bars, every 5 minutes during the session | `iex`, real time |
| `signals_live.py live` | today's 1-minute bars at each 30-minute checkpoint | `iex`, real time |

Bars are stamped with their start time and filtered to 09:30–16:00 ET, which is the same shape as the Yahoo data
before. Older files in `private/` that came from Yahoo stay where they are. Offline checks:
`python python/tests/test_alpaca.py`.

## Live data: Alpaca (free)
[Alpaca](https://alpaca.markets) Market Data API v2, free Basic plan, via its official, documented API:
- real-time quotes, trades and bars from the **IEX exchange** (one exchange, so prices are real but volumes only
  cover IEX);
- historical bars from **all US exchanges** (SIP), except the most recent 15 minutes;
- 200 requests per minute. The dashboard caches responses so that polling every 1–2 s stays well under this.

Check Alpaca's current plan terms yourself when you sign up. Free tiers change.

### Setup (you do this; the app never asks for your password)
1. Create a free Alpaca account and generate **API keys** for market data in the Alpaca dashboard. Paper-trading
   keys work for market data.
2. Give the keys to the app in **one** of these two ways:
   - environment variables `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`; or
   - a file `private/config/alpaca.json`. The `private/` folder is git-ignored, and the pre-commit and pre-push
     scanners block keys:
     ```json
     { "key_id": "YOUR_KEY_ID", "secret_key": "YOUR_SECRET", "feed": "iex" }
     ```
3. Restart the app. The top bar then reads **LIVE · ALPACA** instead of **DEMO · RECORDED DATA**.

The keys stay on this PC. They are sent only to `data.alpaca.markets` and are never shown in the browser or the logs.

## Without a key: demo mode
The app replays the recorded 1-minute data in `private/data/hist_1m_30d`: the latest session, looping, at 30× speed.
It is clearly labelled **DEMO** and has no bid/ask quotes.

## Later: Interactive Brokers
IBKR market data (TWS API, paid subscription) can be added as another provider behind the same `IMarketData`
interface (`src/Trademaster.Cli/MarketData.cs`). That is the natural next step once a broker account exists.
